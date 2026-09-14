#!/usr/bin/env python
"""EXP01 + EXP02: logistic regression vs LightGBM, then calibration.

    python scripts/train_baseline.py --config configs/models/boosting.yaml
    python scripts/train_baseline.py --config configs/models/boosting.yaml --set features.feature_set=with_lc_grade

Trains both models on the same features, scores them on the out-of-time window, and then compares
calibration methods. The out-of-time window is touched exactly once, at the end.

Two feature sets are worth running:

* ``primary`` -- excludes LendingClub's grade, sub-grade, interest rate and instalment. "Can we
  underwrite from primary evidence?"
* ``with_lc_grade`` -- includes them. "Can we beat the incumbent scorecard?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.evaluation.metrics import (  # noqa: E402
    compare_models,
    compute_metrics,
    gains_table,
    reliability_table,
)
from credit_risk.features.build import FeatureBuilder, FeatureSpec  # noqa: E402
from credit_risk.models.baseline import LogisticBaseline, LogisticConfig  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel, SearchSpace  # noqa: E402
from credit_risk.models.calibration import compare_calibrations  # noqa: E402
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    load_prepared,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.baseline")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/models/boosting.yaml")
    parser.add_argument("--skip-logistic", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    y_train, y_valid, y_test = (
        d["default"].astype(int).to_numpy() for d in (train, valid, test)
    )
    LOG.info(
        "train %s | valid %s | test %s | default rates %.4f / %.4f / %.4f",
        f"{len(train):,}", f"{len(valid):,}", f"{len(test):,}",
        y_train.mean(), y_valid.mean(), y_test.mean(),
    )

    # -- features: fitted on train, guard measured against valid --------------
    fcfg = config.get("features", {})
    spec = FeatureSpec(
        feature_set=str(fcfg.get("feature_set", "primary")),
        include_vintage_sensitive=bool(fcfg.get("include_vintage_sensitive", True)),
        include_late_additions=bool(fcfg.get("include_late_additions", False)),
        include_joint=bool(fcfg.get("include_joint", False)),
        include_derived=bool(fcfg.get("include_derived", True)),
        availability_shift_tolerance=float(fcfg.get("availability_shift_tolerance", 0.05)),
        max_categorical_levels=int(fcfg.get("max_categorical_levels", 40)),
    )
    builder = FeatureBuilder(spec).fit(train, valid)
    X_train, X_valid, X_test = (builder.transform(d) for d in (train, valid, test))
    builder.save(paths.artifacts / f"{run_name}_features.pkl")
    save_table(builder.describe_dropped(), paths, f"{run_name}_dropped_features", index=False)

    scores: dict[str, np.ndarray] = {}
    metrics = {}

    # -- EXP01a: logistic baseline -------------------------------------------
    if not args.skip_logistic:
        lcfg = LogisticConfig(
            C=float(config.get_path("model.C", 0.1)),
            class_weight=config.get_path("model.class_weight", None),
        )
        logistic = LogisticBaseline(
            builder.state.numeric_features + builder.state.indicator_features,
            builder.categorical_features,
            lcfg,
        ).fit(X_train, y_train)
        scores["logistic"] = logistic.predict_proba(X_test)
        metrics["logistic"] = compute_metrics(y_test, scores["logistic"])
        LOG.info("logistic  | %s", metrics["logistic"].summary_line())
        logistic.save(paths.artifacts / f"{run_name}_logistic.pkl")
        save_table(logistic.coefficients().head(40), paths, f"{run_name}_logistic_coefficients",
                   index=False)

    # -- EXP01b: LightGBM ------------------------------------------------------
    bcfg = BoostingConfig(
        learning_rate=float(config.get_path("model.learning_rate", 0.03)),
        num_leaves=int(config.get_path("model.num_leaves", 31)),
        min_child_samples=int(config.get_path("model.min_child_samples", 200)),
        subsample=float(config.get_path("model.subsample", 0.8)),
        colsample_bytree=float(config.get_path("model.colsample_bytree", 0.8)),
        reg_alpha=float(config.get_path("model.reg_alpha", 0.1)),
        reg_lambda=float(config.get_path("model.reg_lambda", 1.0)),
        n_estimators=int(config.get_path("model.n_estimators", 3000)),
        early_stopping_rounds=int(config.get_path("model.early_stopping_rounds", 100)),
        class_weight=config.get_path("model.class_weight", None),
    )
    search = None
    if bool(config.get_path("search.enabled", True)) and not args.skip_search:
        search = SearchSpace(n_trials=int(config.get_path("search.n_trials", 15)))

    boosting = BoostingModel(builder.categorical_features, bcfg).fit(
        X_train, y_train, X_valid, y_valid, seed=seed, search=search
    )
    p_valid = boosting.predict_proba(X_valid)
    scores["lightgbm"] = boosting.predict_proba(X_test)
    metrics["lightgbm"] = compute_metrics(y_test, scores["lightgbm"])
    LOG.info("lightgbm  | %s", metrics["lightgbm"].summary_line())
    boosting.save(paths.artifacts / f"{run_name}_boosting.pkl")
    save_table(boosting.feature_importance().head(40), paths, f"{run_name}_importance", index=False)
    if boosting.result and boosting.result.search_trials:
        save_table(pd.DataFrame(boosting.result.search_trials), paths, f"{run_name}_search",
                   index=False)

    save_table(compare_models(metrics), paths, f"{run_name}_exp01_models")

    # -- EXP02: calibration ----------------------------------------------------
    methods = tuple(config.get_path("calibration.methods", ["identity", "platt", "isotonic"]))
    calibration = compare_calibrations(y_valid, p_valid, y_test, scores["lightgbm"], methods)
    save_table(calibration.table(), paths, f"{run_name}_exp02_calibration")
    for name, calibrator in calibration.fitted.items():
        if name != "identity":
            calibrator.save(paths.artifacts / f"{run_name}_calibrator_{name}.pkl")

    best_method = min(
        (m for m in calibration.per_method if m != "identity"),
        key=lambda m: calibration.per_method[m].brier,
    )
    np.save(paths.artifacts / f"{run_name}_pd_test.npy", calibration.calibrated_test[best_method])
    save_table(
        reliability_table(y_test, calibration.calibrated_test[best_method]),
        paths, f"{run_name}_reliability", index=False,
    )
    save_table(gains_table(y_test, scores["lightgbm"]), paths, f"{run_name}_gains", index=False)

    # -- figures ---------------------------------------------------------------
    if not args.no_figures:
        from credit_risk.evaluation.plots import (
            gains_chart, reliability_diagram, roc_curves, save, score_distribution,
        )
        curves = {n: (y_test, s) for n, s in scores.items()}
        curves["lightgbm (calibrated)"] = (y_test, calibration.calibrated_test[best_method])
        save(reliability_diagram(curves), paths.results_figures / f"{run_name}_reliability.png")
        save(roc_curves({n: (y_test, s) for n, s in scores.items()}),
             paths.results_figures / f"{run_name}_roc.png")
        save(gains_chart(y_test, scores["lightgbm"]), paths.results_figures / f"{run_name}_gains.png")
        save(score_distribution({n: s for n, s in scores.items()}),
             paths.results_figures / f"{run_name}_scores.png")

    # -- written verdicts ------------------------------------------------------
    lines = [
        f"# {run_name}: EXP01 and EXP02",
        "",
        f"Feature set: **{spec.feature_set}** ({len(builder.feature_names)} features, "
        f"{len(builder.state.dropped)} dropped by the availability guard and constancy checks).",
        f"Out-of-time test window: {len(test):,} loans, observed default rate {y_test.mean():.4f}.",
        "",
        "## EXP01 - does non-linear modelling improve default prediction?",
        "",
        compare_models(metrics).round(5).to_markdown(),
        "",
    ]
    if "logistic" in metrics:
        delta = metrics["lightgbm"].roc_auc - metrics["logistic"].roc_auc
        lines.append(
            f"**Verdict.** LightGBM moves out-of-time ROC-AUC by {delta:+.4f} against logistic "
            f"regression ({metrics['logistic'].roc_auc:.4f} -> {metrics['lightgbm'].roc_auc:.4f}). "
            + (
                "That is a small margin for the extra complexity, and the logistic scorecard "
                "remains defensible."
                if abs(delta) < 0.01
                else "That is a material margin and justifies the non-linear model."
            )
        )
    lines += [
        "",
        "## EXP02 - does calibration improve the reliability of PD estimates?",
        "",
        calibration.table().round(5).to_markdown(),
        "",
        f"**Verdict.** {calibration.verdict()}",
        "",
        "Note the ROC-AUC column: it barely moves. That is the whole point. Calibration is a "
        "monotone transform of the score, so every ranking metric is blind to it, while the "
        "probabilities the decision layer multiplies by an exposure become materially more "
        "accurate. A model that ranks well but is mis-calibrated will systematically mis-state "
        "expected loss.",
    ]
    write_report(lines, paths, f"{run_name}_exp01_exp02", source)
    save_json(
        {
            "feature_set": spec.feature_set,
            "n_features": len(builder.feature_names),
            "models": {k: v.to_dict() for k, v in metrics.items()},
            "calibration": {k: v.to_dict() for k, v in calibration.per_method.items()},
            "best_calibration": best_method,
        },
        paths, f"{run_name}_exp01_exp02",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
