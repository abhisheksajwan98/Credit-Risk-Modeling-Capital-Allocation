#!/usr/bin/env python
"""EXP10: does disagreement between a tabular and a graph-augmented model flag unreliable PDs?

    python scripts/analyze_disagreement.py --config configs/experiments/exp10_disagreement.yaml

Two corrections relative to the first version of this experiment, both of which changed what the
number means:

1. **The alternative model differs in features only.** Previously it also used different
   hyper-parameters (1000 trees / 50 early-stopping rounds against the baseline's search) and was
   always isotonic-calibrated while the baseline used whichever calibrator won on Brier. So
   "disagreement" partly measured those choices. Here the baseline is *refitted* under the
   identical configuration and calibrator, and only the feature set changes.

2. **Disagreement is measured on the log-odds scale.** An absolute probability difference is
   mechanically larger in the middle of the PD range, so ``|p_a - p_b|`` partly restates the PD
   level rather than measuring disagreement. ``|logit(p_a) - logit(p_b)|`` is scale-free.

As in EXP08, the marginal quintile table is descriptive and the conditional-on-PD table is the
evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.decision.blindspot import (  # noqa: E402
    conditional_signal_table,
    marginal_signal_table,
)
from credit_risk.evaluation.metrics import compute_metrics  # noqa: E402
from credit_risk.features.build import FeatureBuilder, FeatureSpec  # noqa: E402
from credit_risk.features.financial import add_derived_features  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel  # noqa: E402
from credit_risk.models.calibration import ProbabilityCalibrator  # noqa: E402
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

LOG = get_logger("scripts.exp10")

QUINTILE_LABELS = ("Q1_agree", "Q2", "Q3", "Q4", "Q5_disagree")
_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1.0 - p))


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp10_disagreement.yaml")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = add_derived_features(load_prepared(paths).reset_index(drop=True))
    cohort_path = paths.data_processed / "cohort_features.parquet"
    if not cohort_path.exists():
        raise SystemExit(
            "cohort_features.parquet not found. Run:\n"
            "  python scripts/build_graph.py --config configs/graph/cohort_graph.yaml"
        )
    cohort = pd.read_parquet(cohort_path)
    if len(cohort) != len(df):
        raise SystemExit(
            f"cohort features have {len(cohort):,} rows but the prepared table has {len(df):,}. "
            "Re-run build_graph.py against the current processed data."
        )
    df = pd.concat([df, cohort.set_index(df.index)], axis=1)

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    y = df["default"].astype("float32").to_numpy()
    y_train, y_valid, y_test = (y[masks[s]] for s in ("train", "valid", "test"))

    cohort_columns = tuple(c for c in cohort.columns if c in df.columns)
    match_calibrator = bool(config.get_path("disagreement.match_baseline_calibrator", True))
    scale = str(config.get_path("disagreement.scale", "logodds"))

    # Identical configuration for both arms; only `extra_numeric` differs.
    shared = BoostingConfig(
        n_estimators=int(config.get_path("model.n_estimators", 2000)),
        early_stopping_rounds=int(config.get_path("model.early_stopping_rounds", 100)),
        learning_rate=float(config.get_path("model.learning_rate", 0.03)),
        num_leaves=int(config.get_path("model.num_leaves", 31)),
        min_child_samples=int(config.get_path("model.min_child_samples", 200)),
    )

    def fit_arm(label: str, extra: tuple[str, ...]) -> np.ndarray:
        builder = FeatureBuilder(FeatureSpec(feature_set="primary", extra_numeric=extra)).fit(
            train, valid
        )
        X = {k: builder.transform(d) for k, d in (("tr", train), ("va", valid), ("te", test))}
        model = BoostingModel(builder.categorical_features, shared).fit(
            X["tr"], y_train, X["va"], y_valid, seed=seed
        )
        calibrator = ProbabilityCalibrator("isotonic").fit(y_valid, model.predict_proba(X["va"]))
        scores = calibrator.transform(model.predict_proba(X["te"]))
        LOG.info("%-22s | %s", label, compute_metrics(y_test, scores).summary_line())
        return scores

    # The baseline is refitted here rather than loaded from disk, precisely so that the two arms
    # share hyper-parameters and calibrator. `boosting_pd_test.npy` is the production PD and is
    # used elsewhere; it is not the right comparator for an ablation.
    LOG.info("fitting both arms under an identical configuration (features are the only difference)")
    p_tabular = fit_arm("tabular only", ())
    p_graph = fit_arm("tabular + graph", cohort_columns)

    if scale == "logodds":
        disagreement = np.abs(_logit(p_tabular) - _logit(p_graph))
    else:
        disagreement = np.abs(p_tabular - p_graph)
    np.save(paths.artifacts / "disagreement_test.npy", disagreement)

    # The production PD is what a decision would actually consume, so reliability is assessed
    # against it rather than against the ablation arm.
    production_path = paths.artifacts / "boosting_pd_test.npy"
    p_reference = np.load(production_path) if production_path.exists() else p_tabular

    n_bins = int(config.get_path("analysis.n_bins", 5))
    marginal = marginal_signal_table(
        y_test, p_reference, disagreement, n_bins=n_bins,
        labels=QUINTILE_LABELS if n_bins == 5 else None,
    )
    save_table(marginal, paths, f"{run_name}_bins", index=False)

    conditional = conditional_signal_table(
        y_test, p_reference, disagreement,
        n_strata=int(config.get_path("analysis.n_strata", 10)),
        min_per_half=int(config.get_path("analysis.min_per_half", 200)),
    )
    save_table(conditional.table, paths, f"{run_name}_conditional", index=False)

    verdict = conditional.verdict(f"Disagreement ({scale} scale)")
    LOG.info(verdict)
    keep = conditional.sign_test_p < 0.05 and conditional.mean_gap_difference > 0

    # For reference: how much did the probability-scale version confound with PD?
    prob_disagreement = np.abs(p_tabular - p_graph)
    prob_corr = float(pd.Series(prob_disagreement).corr(pd.Series(p_reference), method="spearman"))

    lines = [
        "# EXP10: model disagreement as a blind-spot signal",
        "",
        "Two LightGBM models were fitted under an **identical** configuration and calibrator, "
        f"differing only in whether the {len(cohort_columns)} graph cohort features were "
        f"available. Disagreement is `|logit(p_tabular) - logit(p_graph)|` on the "
        f"{len(test):,} out-of-time test borrowers.",
        "",
        f"Measured on the **{scale}** scale, disagreement correlates with the PD level at "
        f"Spearman {conditional.spearman_with_pd:+.3f}. On the raw probability scale the same "
        f"quantity correlates at {prob_corr:+.3f} -- which is why the log-odds scale is used: an "
        "absolute probability difference is mechanically larger in the middle of the PD range and "
        "so partly restates the PD rather than measuring disagreement.",
        "",
        "## Marginal view (descriptive only)",
        "",
        marginal.round(4).to_markdown(index=False),
        "",
        "## Conditional view (the evidence)",
        "",
        conditional.table.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"**Decision: {'KEEP' if keep else 'REJECT'}.**",
        "",
        "> **Correction.** The first version of this experiment compared the production PD "
        "against an alternative model that differed in hyper-parameters *and* calibrator *and* "
        "features, using an absolute probability difference. The disagreement it measured was "
        "therefore partly a measure of those choices and partly a restatement of the PD level. "
        "See `docs/AUDIT_REPORT.md`.",
    ]
    write_report(lines, paths, f"{run_name}", source)
    save_json(
        {
            "scale": scale,
            "match_baseline_calibrator": match_calibrator,
            "n_cohort_features": len(cohort_columns),
            "spearman_with_pd_logodds": conditional.spearman_with_pd,
            "spearman_with_pd_probability": prob_corr,
            "marginal": marginal.to_dict(orient="records"),
            "conditional": conditional.table.to_dict(orient="records"),
            "n_strata": conditional.n_strata,
            "n_strata_positive": conditional.n_strata_positive,
            "mean_gap_difference": conditional.mean_gap_difference,
            "sign_test_p": conditional.sign_test_p,
            "odds_ratio_per_sd": conditional.odds_ratio_per_sd,
            "decision": "KEEP" if keep else "REJECT",
            "tabular_metrics": compute_metrics(y_test, p_tabular).to_dict(),
            "graph_metrics": compute_metrics(y_test, p_graph).to_dict(),
        },
        paths, f"{run_name}",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
