#!/usr/bin/env python
"""EXP09: does relational uncertainty from a VGAE identify where the PD model fails?

    python scripts/train_vgae.py --config configs/experiments/exp09_vgae.yaml

A variational graph autoencoder reconstructs the time-respecting cohort graph and carries a
per-node posterior variance. The hypothesis is that a borrower whose neighbourhood the model
cannot reconstruct confidently is one whose PD is less trustworthy -- a *relational* blind spot,
distinct from the tabular one EXP08 measures.

This experiment is designed to be able to return a clean negative. Two things make a null result
interpretable rather than ambiguous:

* the VGAE trains to convergence with early stopping on held-out link reconstruction, so
  "no signal" cannot be confused with "undertrained";
* the signal is tested **conditionally on the PD level**, the same way as EXP08 and EXP10, so a
  positive result could not be an artefact of the signal merely tracking PD.
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
from credit_risk.features.build import (  # noqa: E402
    FeatureBuilder,
    FeatureSpec,
    to_numeric_matrix,
)
from credit_risk.graph.construction import TimeRespectingGraph  # noqa: E402
from credit_risk.models.vgae import VGAEConfig, VGAETrainer  # noqa: E402
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

LOG = get_logger("scripts.exp09")

QUINTILE_LABELS = ("Q1_certain", "Q2", "Q3", "Q4", "Q5_uncertain")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp09_vgae.yaml")
    parser.add_argument("--model-run", default="boosting")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    graph_path = paths.data_processed / "graph.npz"
    pd_path = paths.artifacts / f"{args.model_run}_pd_test.npy"
    for path in (graph_path, pd_path):
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found. Run build_graph.py and train_baseline.py first."
            )

    graph = TimeRespectingGraph.load(graph_path)
    if graph.n_nodes != len(df):
        raise SystemExit(
            f"graph has {graph.n_nodes:,} nodes but the prepared table has {len(df):,} rows"
        )

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    idx = {s: np.flatnonzero(m) for s, m in masks.items()}

    # Node features are the same decision-time features the PD model uses, standardised on
    # training statistics. No label information enters the encoder.
    builder = FeatureBuilder(FeatureSpec(feature_set="primary")).fit(df[masks["train"]], df[masks["valid"]])
    X_all = builder.transform(df)
    _, _, means, stds = to_numeric_matrix(
        X_all[masks["train"]], builder.categorical_features, builder.state.categorical_levels
    )
    node_features, _, _, _ = to_numeric_matrix(
        X_all, builder.categorical_features, builder.state.categorical_levels,
        means=means, stds=stds,
    )

    vgae_config = VGAEConfig(
        hidden_dim=int(config.get_path("vgae.hidden_dim", 32)),
        latent_dim=int(config.get_path("vgae.latent_dim", 16)),
        n_layers=int(config.get_path("vgae.n_layers", 2)),
        kl_weight=float(config.get_path("vgae.kl_weight", 1e-4)),
        learning_rate=float(config.get_path("vgae.learning_rate", 5e-3)),
        max_epochs=int(config.get_path("vgae.max_epochs", 40)),
        patience=int(config.get_path("vgae.patience", 5)),
        batch_size=int(config.get_path("vgae.batch_size", 1024)),
        fanout=tuple(config.get_path("vgae.fanout", [10, 5])),
        seed=seed,
    )

    trainer = VGAETrainer(vgae_config)
    trainer.fit(node_features, graph.neighbour_index, graph.neighbour_mask, idx["train"])
    trainer.save(paths.artifacts / f"{run_name}_vgae.pt")
    save_table(pd.DataFrame(trainer.history), paths, f"{run_name}_history", index=False)

    _, sigma = trainer.embed(
        node_features, graph.neighbour_index, graph.neighbour_mask, idx["test"]
    )
    mean_sigma = sigma.mean(axis=1)
    np.save(paths.artifacts / "vgae_sigma_test.npy", mean_sigma)

    test = df[masks["test"]]
    y_test = test["default"].astype(int).to_numpy()
    p_test = np.load(pd_path)
    if len(p_test) != len(y_test):
        raise SystemExit(
            f"{pd_path.name} has {len(p_test):,} rows but the test split has {len(y_test):,}"
        )

    n_bins = int(config.get_path("analysis.n_bins", 5))
    marginal = marginal_signal_table(
        y_test, p_test, mean_sigma, n_bins=n_bins,
        labels=QUINTILE_LABELS if n_bins == 5 else None,
    )
    save_table(marginal, paths, f"{run_name}_sigma_bins", index=False)

    conditional = conditional_signal_table(
        y_test, p_test, mean_sigma,
        n_strata=int(config.get_path("analysis.n_strata", 10)),
        min_per_half=int(config.get_path("analysis.min_per_half", 200)),
    )
    save_table(conditional.table, paths, f"{run_name}_conditional", index=False)

    verdict = conditional.verdict("Graph uncertainty (VGAE sigma)")
    LOG.info(verdict)
    keep = conditional.sign_test_p < 0.05 and conditional.mean_gap_difference > 0

    # Degree is the obvious non-neural competitor: if sigma is just a proxy for "few neighbours",
    # the VGAE adds nothing a one-line count would not.
    degree = graph.degree()[idx["test"]]
    degree_corr = float(pd.Series(mean_sigma).corr(pd.Series(degree), method="spearman"))

    lines = [
        "# EXP09: relational uncertainty as a blind-spot signal",
        "",
        f"A VGAE ({vgae_config.n_layers} GraphSAGE layers, latent dim {vgae_config.latent_dim}) "
        f"was trained to reconstruct the cohort graph, stopping early on held-out link "
        f"reconstruction after {len(trainer.history)} epochs "
        f"(best validation loss {trainer.best_valid_loss:.4f}). Per-node posterior sigma scores "
        f"the {len(test):,} out-of-time test borrowers.",
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
        f"**Decision: {'KEEP' if keep else 'REJECT'}.** "
        + (
            "Relational uncertainty carries information about PD reliability beyond the PD level."
            if keep
            else "Relational uncertainty does **not** predict where the PD model fails. This is a "
            "clean negative: the model was trained to convergence with early stopping, so the "
            "null is a property of the signal rather than of the training budget. Reporting it "
            "is the point -- a complex model that adds nothing should be dropped, and the tabular "
            "anomaly signal in EXP08 is doing different work."
        ),
        "",
        f"Sigma correlates with node degree at Spearman {degree_corr:+.3f}. "
        + (
            "It is largely a restatement of how many neighbours a borrower has, which a simple "
            "count would give for free."
            if abs(degree_corr) > 0.5
            else "It is not simply a proxy for neighbourhood size."
        ),
        "",
        "> **Correction.** The first version of this experiment trained for 10 epochs with no "
        "validation split and no early stopping, which made 'no signal' and 'undertrained' "
        "indistinguishable. See `docs/AUDIT_REPORT.md`.",
    ]
    write_report(lines, paths, f"{run_name}", source)
    save_json(
        {
            "epochs_run": len(trainer.history),
            "best_valid_loss": trainer.best_valid_loss,
            "latent_dim": vgae_config.latent_dim,
            "marginal": marginal.to_dict(orient="records"),
            "conditional": conditional.table.to_dict(orient="records"),
            "n_strata": conditional.n_strata,
            "n_strata_positive": conditional.n_strata_positive,
            "mean_gap_difference": conditional.mean_gap_difference,
            "sign_test_p": conditional.sign_test_p,
            "odds_ratio_per_sd": conditional.odds_ratio_per_sd,
            "spearman_with_pd": conditional.spearman_with_pd,
            "spearman_with_degree": degree_corr,
            "decision": "KEEP" if keep else "REJECT",
        },
        paths, f"{run_name}",
    )
    write_manifest(run_name, config, seed, paths, data_source=source, device=trainer.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
