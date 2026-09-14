#!/usr/bin/env python
"""EXP08: does autoencoder reconstruction error identify borrowers whose PD is untrustworthy?

    python scripts/analyze_blind_spots.py --config configs/experiments/exp08_blindspots.yaml

An autoencoder is fitted on the **training** feature distribution and never sees the target. A
borrower it reconstructs badly is one the training population does not describe well, which is a
reasonable prior for "the PD model is less trustworthy here" -- and a claim to be tested.

Two tests are reported, and only the second is evidence:

* **Marginal.** Model performance by anomaly quintile. Descriptive. If the anomaly score
  correlates with the PD level, this table will show calibration degrading merely because a
  boosted model is already less well calibrated at high PD.
* **Conditional.** Within each PD decile, split at the median anomaly score and compare the
  calibration gap. This isolates whether the signal adds anything the PD does not already say.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from credit_risk.decision.blindspot import (  # noqa: E402
    conditional_signal_table,
    marginal_signal_table,
)
from credit_risk.features.build import FeatureBuilder  # noqa: E402
from credit_risk.models.autoencoder import TabularAutoencoder, TabularAutoencoderConfig  # noqa: E402
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

LOG = get_logger("scripts.exp08")

QUINTILE_LABELS = ("Q1_normal", "Q2", "Q3", "Q4", "Q5_anomalous")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp08_blindspots.yaml")
    parser.add_argument("--model-run", default="boosting")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    y_test = test["default"].astype(int).to_numpy()

    builder_path = paths.artifacts / f"{args.model_run}_features.pkl"
    pd_path = paths.artifacts / f"{args.model_run}_pd_test.npy"
    for path in (builder_path, pd_path):
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found. Run:\n"
                "  python scripts/train_baseline.py --config configs/models/boosting.yaml"
            )

    # The autoencoder must see exactly the features the PD model saw, or "anomalous" would
    # describe a different representation from the one whose reliability is in question.
    builder = FeatureBuilder.load(builder_path)
    X_train, X_valid, X_test = (builder.transform(d) for d in (train, valid, test))
    p_test = np.load(pd_path)
    if len(p_test) != len(test):
        raise SystemExit(
            f"{pd_path.name} has {len(p_test):,} rows but the test split has {len(test):,}"
        )

    ae_config = TabularAutoencoderConfig(
        hidden_dims=list(config.get_path("autoencoder.hidden_dims", [64, 32])),
        latent_dim=int(config.get_path("autoencoder.latent_dim", 16)),
        epochs=int(config.get_path("autoencoder.epochs", 100)),
        patience=int(config.get_path("autoencoder.patience", 10)),
        learning_rate=float(config.get_path("autoencoder.learning_rate", 1e-3)),
        batch_size=int(config.get_path("autoencoder.batch_size", 256)),
        dropout=float(config.get_path("autoencoder.dropout", 0.1)),
        seed=seed,
    )
    n_numeric = X_train.select_dtypes(include=[np.number]).shape[1]
    n_dropped = X_train.shape[1] - n_numeric
    LOG.info(
        "autoencoding %d numeric features (%d categorical columns are not encoded)",
        n_numeric, n_dropped,
    )

    autoencoder = TabularAutoencoder(input_dim=n_numeric, config=ae_config)
    autoencoder.fit(X_train, X_valid)
    anomaly = autoencoder.score_anomalies(X_test)
    autoencoder.save(paths.artifacts / f"{run_name}_autoencoder.pt")
    np.save(paths.artifacts / "anomaly_test.npy", anomaly)

    # --- marginal (descriptive) ---------------------------------------------
    n_bins = int(config.get_path("analysis.n_bins", 5))
    marginal = marginal_signal_table(
        y_test, p_test, anomaly, n_bins=n_bins,
        labels=QUINTILE_LABELS if n_bins == 5 else None,
    )
    save_table(marginal, paths, f"{run_name}_bins", index=False)

    # --- conditional (the actual evidence) ----------------------------------
    conditional = conditional_signal_table(
        y_test, p_test, anomaly,
        n_strata=int(config.get_path("analysis.n_strata", 10)),
        min_per_half=int(config.get_path("analysis.min_per_half", 200)),
    )
    save_table(conditional.table, paths, f"{run_name}_conditional", index=False)

    verdict = conditional.verdict("Anomaly score")
    LOG.info(verdict)

    keep = conditional.sign_test_p < 0.05 and conditional.mean_gap_difference > 0
    lines = [
        "# EXP08: tabular anomaly as a blind-spot signal",
        "",
        f"An autoencoder ({n_numeric} numeric features, latent dim {ae_config.latent_dim}) was "
        f"fitted on the training window and never saw the target. Reconstruction error scores "
        f"the {len(test):,} out-of-time test borrowers.",
        "",
        "## Marginal view (descriptive only)",
        "",
        marginal.round(4).to_markdown(index=False),
        "",
        "Read this table with care. If the anomaly score correlates with the PD level, "
        "calibration will appear to degrade across quintiles simply because a boosted model is "
        "less well calibrated at high PD. That is not a blind spot; it is a known property.",
        "",
        "## Conditional view (the evidence)",
        "",
        "Within each PD decile the population is split at the median anomaly score, and the "
        "calibration gap (observed minus predicted default rate) is compared. A positive "
        "`gap_difference` means the model under-states risk more for anomalous borrowers *at the "
        "same predicted PD*.",
        "",
        conditional.table.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"**Decision: {'KEEP' if keep else 'REJECT'}.** "
        + (
            "The signal carries information about model reliability that the PD does not. Note "
            "that the effect on the outcome itself is modest -- the value is in flagging "
            "*miscalibration*, not in improving discrimination."
            if keep
            else "The apparent effect does not survive conditioning on the PD level, so the "
            "marginal table was showing the model's known behaviour at high PD rather than a "
            "blind spot."
        ),
        "",
        "**Scope limit.** Only numeric features are encoded; the "
        f"{n_dropped} categorical columns are not. A borrower unusual only in loan purpose or "
        "state would not be flagged.",
    ]
    write_report(lines, paths, f"{run_name}", source)
    save_json(
        {
            "n_numeric_features": n_numeric,
            "n_categorical_excluded": n_dropped,
            "marginal": marginal.to_dict(orient="records"),
            "conditional": conditional.table.to_dict(orient="records"),
            "n_strata": conditional.n_strata,
            "n_strata_positive": conditional.n_strata_positive,
            "mean_gap_difference": conditional.mean_gap_difference,
            "sign_test_p": conditional.sign_test_p,
            "odds_ratio_per_sd": conditional.odds_ratio_per_sd,
            "spearman_with_pd": conditional.spearman_with_pd,
            "decision": "KEEP" if keep else "REJECT",
        },
        paths, f"{run_name}",
    )
    write_manifest(run_name, config, seed, paths, data_source=source,
                   device=autoencoder.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
