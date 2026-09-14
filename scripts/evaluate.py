#!/usr/bin/env python
"""EXP07 plus the trust layer: drift, explainability and a fairness slice.

    python scripts/evaluate.py --config configs/models/boosting.yaml

Consumes the artefacts written by ``train_baseline.py`` and reports:

* **EXP07 drift.** Feature and prediction drift measured on the 2016-2018 window, which is carried
  through the pipeline *without labels* precisely because that is the production situation --
  features and scores arrive immediately, outcomes arrive years later. Calibration drift is
  measured separately on the vintages that have matured.
* **Explainability.** SHAP attribution and human-readable reason codes for a stratified sample of
  decisions.
* **Fairness.** Error and approval rates across observable proxy groups, with the limits of that
  exercise stated rather than glossed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.evaluation.fairness import fairness_report  # noqa: E402
from credit_risk.explainability.reasons import risk_band  # noqa: E402
from credit_risk.explainability.shap_explain import (  # noqa: E402
    explain_boosting,
    sample_explanations,
    save_importance,
)
from credit_risk.features.build import FeatureBuilder  # noqa: E402
from credit_risk.models.boosting import BoostingModel  # noqa: E402
from credit_risk.models.calibration import ProbabilityCalibrator  # noqa: E402
from credit_risk.monitoring.drift import (  # noqa: E402
    calibration_drift,
    feature_drift,
    monitoring_summary,
    prediction_drift,
)
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

LOG = get_logger("scripts.evaluate")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/models/boosting.yaml")
    parser.add_argument("--model-run", default="boosting", help="Run name of the trained model.")
    parser.add_argument("--threshold", type=float, default=0.15, help="Cut-off for decision rates.")
    parser.add_argument("--shap-rows", type=int, default=4000)
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    stem = args.model_run
    needed = [
        paths.artifacts / f"{stem}_features.pkl",
        paths.artifacts / f"{stem}_boosting.pkl",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing artefacts: " + ", ".join(p.name for p in missing) + "\nRun:\n"
            "  python scripts/train_baseline.py --config configs/models/boosting.yaml"
        )

    builder = FeatureBuilder.load(needed[0])
    model = BoostingModel.load(needed[1])
    calibrator_path = paths.artifacts / f"{stem}_calibrator_platt.pkl"
    if not calibrator_path.exists():
        calibrator_path = paths.artifacts / f"{stem}_calibrator_isotonic.pkl"
    calibrator = (
        ProbabilityCalibrator.load(calibrator_path)
        if calibrator_path.exists()
        else ProbabilityCalibrator("identity")
    )

    df = load_prepared(paths)
    splits = {s: df[df["split"] == s] for s in ("train", "valid", "test", "monitor")}
    features = {k: builder.transform(v) for k, v in splits.items() if len(v)}
    scores = {k: calibrator.transform(model.predict_proba(v)) for k, v in features.items()}
    for k, v in scores.items():
        LOG.info("%-8s n=%7d mean PD %.4f", k, len(v), float(v.mean()))

    # =====================================================================
    # EXP07 - drift
    # =====================================================================
    numeric = builder.state.numeric_features
    categorical = builder.categorical_features

    reports: dict[str, object] = {}
    lines = [f"# {run_name}: EXP07, explainability and fairness", ""]

    if "monitor" in features and len(features["monitor"]):
        fdrift = feature_drift(
            features["train"], features["monitor"], numeric, categorical,
            reference_label="train (2010-2014H1)", current_label="production (2016-2018)",
        )
        pdrift = prediction_drift(scores["train"], scores["monitor"])
        save_table(fdrift.frame, paths, f"{run_name}_exp07_feature_drift", index=False)
        save_json(pdrift.to_dict(), paths, f"{run_name}_exp07_prediction_drift")
        reports["feature_drift"] = fdrift.summary()
        reports["prediction_drift"] = pdrift.to_dict()
    else:
        fdrift = pdrift = None
        LOG.warning("no monitoring window present; skipping production drift")

    labelled = pd.concat([splits[s] for s in ("train", "valid", "test")])
    labelled_scores = np.concatenate([scores[s] for s in ("train", "valid", "test")])
    labelled = labelled.assign(_pd=labelled_scores)
    cdrift = calibration_drift(labelled, "_pd", period_column="issue_period", freq="Y")
    save_table(cdrift, paths, f"{run_name}_exp07_calibration_drift", index=False)

    lines += [
        "## EXP07 - how do model and policy degrade under drift?",
        "",
        "### Calibration drift on matured vintages",
        "",
        cdrift.round(4).to_markdown(index=False) if len(cdrift) else "_no matured vintages_",
        "",
    ]
    if len(cdrift) >= 2:
        first, last = cdrift.iloc[0], cdrift.iloc[-1]
        lines.append(
            f"Observed/expected moves from {first['observed_expected']:.3f} ({first['vintage']}) "
            f"to {last['observed_expected']:.3f} ({last['vintage']}), while ROC-AUC moves "
            f"{first['roc_auc']:.4f} -> {last['roc_auc']:.4f}."
        )
        lines.append(
            "Note which one moves first. Ranking power is comparatively stable across vintages "
            "while the *level* of predicted risk drifts. A monitoring regime that watches only "
            "AUC would see nothing wrong; the expected loss the business plans against would "
            "already be systematically wrong. This is the operational version of the EXP02 point."
        )
    if fdrift is not None and pdrift is not None:
        lines += [
            "",
            "### Production drift (2016-2018, no mature outcomes)",
            "",
            "```",
            monitoring_summary(fdrift, pdrift, cdrift),
            "```",
            "",
            "Top drifting features:",
            "",
            fdrift.frame.head(12).round(4).to_markdown(index=False),
            "",
            "Approval-rate impact at fixed cut-offs:",
            "",
            "| cut-off | change in approval rate |",
            "|---|---|",
        ]
        lines += [f"| {k} | {v:+.2%} |" for k, v in pdrift.approval_rate_shift.items()]

    # =====================================================================
    # Explainability
    # =====================================================================
    test = splits["test"]
    shap_result = explain_boosting(model, features["test"], max_rows=args.shap_rows, seed=seed)
    save_importance(shap_result, paths.results_tables / f"{run_name}_shap_importance.csv")

    n_shap = len(shap_result.values)
    pd_sample = scores["test"][:n_shap]
    decisions = np.where(pd_sample < args.threshold, "Approve", "Decline")
    exposures = np.where(
        pd_sample < args.threshold,
        pd.to_numeric(test["loan_amnt"], errors="coerce").to_numpy()[:n_shap],
        0.0,
    )
    explanations = sample_explanations(shap_result, pd_sample, decisions, exposures, n=6, seed=seed)

    lines += [
        "",
        "## Explainability",
        "",
        "Global attribution (mean |SHAP|, top 15):",
        "",
        shap_result.global_importance().head(15).round(5).to_markdown(index=False),
        "",
        "### Sample decisions",
        "",
        "Each reason below is derived from a SHAP contribution computed for that specific "
        "applicant. Nothing is generated from the prediction alone.",
        "",
    ]
    for explanation in explanations:
        lines += ["```", explanation.render(), "```", ""]
    lines += [
        "SHAP explains **the model**, not the borrower and not the world. If the model has "
        "learned a proxy, SHAP reports the proxy faithfully. The contributions are local to each "
        "prediction and do not support causal claims such as 'reducing utilisation by 10 points "
        "would lower your PD by X'.",
    ]

    # =====================================================================
    # Fairness
    # =====================================================================
    y_test = test["default"].astype(float).to_numpy()
    tables, fairness_text = fairness_report(test, y_test, scores["test"], threshold=args.threshold)
    for column, table in tables.items():
        save_table(table, paths, f"{run_name}_fairness_{column}", index=False)

    lines += ["", "## Fairness slice", "", "```", fairness_text, "```", ""]
    if tables:
        first_column = next(iter(tables))
        lines += [
            f"Largest groups by `{first_column}`:",
            "",
            tables[first_column].head(10).round(4).to_markdown(index=False),
        ]

    # =====================================================================
    if not args.no_figures:
        from credit_risk.evaluation.plots import (
            drift_bars, save, score_distribution, shap_summary_bar, vintage_calibration,
        )
        save(shap_summary_bar(shap_result.global_importance()),
             paths.results_figures / f"{run_name}_shap.png")
        if len(cdrift):
            save(vintage_calibration(cdrift),
                 paths.results_figures / f"{run_name}_calibration_drift.png")
        if fdrift is not None:
            save(drift_bars(fdrift.frame), paths.results_figures / f"{run_name}_feature_drift.png")
            save(score_distribution({k: v for k, v in scores.items() if k in ("train", "monitor")},
                                    title="PD distribution: train vs production"),
                 paths.results_figures / f"{run_name}_prediction_drift.png")

    write_report(lines, paths, f"{run_name}_exp07_trust", source)
    save_json(
        {
            "drift": reports,
            "calibration_drift": cdrift.to_dict(orient="records"),
            "top_shap": shap_result.global_importance().head(20).to_dict(orient="records"),
            "risk_bands": {
                band: int((np.vectorize(risk_band)(scores["test"]) == band).sum())
                for band in ("Very low", "Low", "Medium", "High", "Very high")
            },
        },
        paths, f"{run_name}_exp07_trust",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
