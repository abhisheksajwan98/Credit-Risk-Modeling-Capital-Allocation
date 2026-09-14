#!/usr/bin/env python
"""EXP04 + EXP05: does relational information help, and does a GNN beat plain aggregates?

    python scripts/build_graph.py --config configs/graph/cohort_graph.yaml
    python scripts/train_gnn.py   --config configs/graph/cohort_graph.yaml

Three arms, all scored on the same out-of-time window:

    A  tabular features only                        (the control)
    B  tabular + time-respecting cohort aggregates  (EXP04)
    C  GraphSAGE over sampled cohort neighbourhoods (EXP05)

**The GNN is not assumed to win.** If arm B captures the relational signal and arm C does not
improve on it, that is the finding, and it is a more useful one than a marginal AUC gain: it says
the value was in the aggregation, not in the learned message passing. The script writes whichever
verdict the numbers support.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.evaluation.metrics import compare_models, compute_metrics  # noqa: E402
from credit_risk.features.build import (  # noqa: E402
    FeatureBuilder,
    FeatureSpec,
    to_numeric_matrix,
)
from credit_risk.features.financial import add_derived_features  # noqa: E402
from credit_risk.graph.construction import TimeRespectingGraph  # noqa: E402
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

LOG = get_logger("scripts.gnn")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/graph/cohort_graph.yaml")
    parser.add_argument("--skip-gnn", action="store_true", help="Run EXP04 only (no torch needed).")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = add_derived_features(load_prepared(paths).reset_index(drop=True))
    cohort_path = paths.data_processed / "cohort_features.parquet"
    graph_path = paths.data_processed / "graph.npz"
    if not cohort_path.exists() or not graph_path.exists():
        raise SystemExit(
            "Graph artefacts missing. Run:\n"
            "  python scripts/build_graph.py --config configs/graph/cohort_graph.yaml"
        )
    cohort = pd.read_parquet(cohort_path)
    df = pd.concat([df, cohort.set_index(df.index)], axis=1)

    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}
    train, valid, test = (df[masks[s]] for s in ("train", "valid", "test"))
    y = df["default"].astype("float32").to_numpy()
    y_train, y_valid, y_test = (y[masks[s]] for s in ("train", "valid", "test"))

    metrics: dict[str, object] = {}
    scores: dict[str, np.ndarray] = {}

    def run_boosting(label: str, extra: tuple[str, ...]) -> None:
        spec = FeatureSpec(feature_set="primary", extra_numeric=extra)
        builder = FeatureBuilder(spec).fit(train, valid)
        X = {k: builder.transform(d) for k, d in (("tr", train), ("va", valid), ("te", test))}
        model = BoostingConfig(n_estimators=2000, early_stopping_rounds=100)
        boosting = BoostingModel(builder.categorical_features, model).fit(
            X["tr"], y_train, X["va"], y_valid, seed=seed
        )
        calibrator = ProbabilityCalibrator("isotonic").fit(y_valid, boosting.predict_proba(X["va"]))
        scores[label] = calibrator.transform(boosting.predict_proba(X["te"]))
        metrics[label] = compute_metrics(y_test, scores[label])
        LOG.info("%-24s | %s", label, metrics[label].summary_line())
        if extra:
            importance = boosting.feature_importance()
            graph_share = importance[importance["feature"].str.startswith("cohort_")][
                "importance"
            ].sum() / max(importance["importance"].sum(), 1)
            LOG.info("  graph features account for %.1f%% of total gain", 100 * graph_share)
            metrics[label].extras["graph_gain_share"] = float(graph_share)
            save_table(importance.head(30), paths, f"{run_name}_{label}_importance", index=False)

    # -- arm A: tabular only ---------------------------------------------------
    run_boosting("A_tabular", ())

    # -- arm B: tabular + cohort aggregates (EXP04) ---------------------------
    cohort_columns = tuple(c for c in cohort.columns if c in df.columns)
    run_boosting("B_tabular_plus_graph", cohort_columns)

    # -- arm C: GraphSAGE (EXP05) ---------------------------------------------
    gnn_summary: dict[str, object] = {}
    device = "cpu"
    if not args.skip_gnn:
        try:
            from credit_risk.models.gnn import GNNConfig, GraphSAGETrainer
        except ImportError as exc:
            LOG.warning("skipping EXP05: %s", exc)
        else:
            gcfg = config.get("gnn", {})
            graph = TimeRespectingGraph.load(graph_path)

            base_spec = FeatureSpec(feature_set="primary")
            node_builder = FeatureBuilder(base_spec).fit(train, valid)
            X_all = node_builder.transform(df)
            _, _, means, stds = to_numeric_matrix(
                X_all[masks["train"]],
                node_builder.categorical_features,
                node_builder.state.categorical_levels,
            )
            node_features, node_names, _, _ = to_numeric_matrix(
                X_all,
                node_builder.categorical_features,
                node_builder.state.categorical_levels,
                means=means,
                stds=stds,
            )
            LOG.info("node feature matrix: %s", node_features.shape)

            trainer = GraphSAGETrainer(
                GNNConfig(
                    hidden_dim=int(gcfg.get("hidden_dim", 64)),
                    embed_dim=int(gcfg.get("embed_dim", 32)),
                    n_layers=int(gcfg.get("n_layers", 2)),
                    dropout=float(gcfg.get("dropout", 0.2)),
                    fanout=tuple(gcfg.get("fanout", [10, 5])),
                    learning_rate=float(gcfg.get("learning_rate", 1e-3)),
                    weight_decay=float(gcfg.get("weight_decay", 1e-5)),
                    batch_size=int(gcfg.get("batch_size", 1024)),
                    max_epochs=int(gcfg.get("max_epochs", 30)),
                    patience=int(gcfg.get("patience", 5)),
                    l2_normalize=bool(gcfg.get("l2_normalize", True)),
                    balance_classes=bool(gcfg.get("balance_classes", False)),
                    device=str(gcfg.get("device", "auto")),
                    seed=seed,
                )
            )
            device = trainer.device
            idx = {s: np.flatnonzero(masks[s]) for s in masks}
            trainer.fit(
                node_features,
                graph.neighbour_index,
                graph.neighbour_mask,
                idx["train"], y[idx["train"]],
                idx["valid"], y[idx["valid"]],
            )
            calibrator = ProbabilityCalibrator("isotonic").fit(
                y[idx["valid"]], trainer.predict(idx["valid"])
            )
            raw = trainer.predict(idx["test"])
            scores["C_graphsage"] = calibrator.transform(raw)
            metrics["C_graphsage"] = compute_metrics(y_test, scores["C_graphsage"])
            LOG.info("%-24s | %s", "C_graphsage", metrics["C_graphsage"].summary_line())
            trainer.save(paths.artifacts / f"{run_name}_graphsage.pt")
            gnn_summary = {
                "device": trainer.device,
                "best_epoch": trainer.result.best_epoch,
                "best_valid_auc": trainer.result.best_valid_auc,
                "n_node_features": int(node_features.shape[1]),
                "history": trainer.result.history,
            }
            save_table(pd.DataFrame(trainer.result.history), paths, f"{run_name}_gnn_history",
                       index=False)

    comparison = compare_models(metrics)
    save_table(comparison, paths, f"{run_name}_exp04_exp05")

    if not args.no_figures and scores:
        from credit_risk.evaluation.plots import reliability_diagram, roc_curves, save
        curves = {k: (y_test, v) for k, v in scores.items()}
        save(roc_curves(curves, title="Layer B arms (out-of-time)"),
             paths.results_figures / f"{run_name}_roc.png")
        save(reliability_diagram(curves), paths.results_figures / f"{run_name}_reliability.png")

    # -- verdicts --------------------------------------------------------------
    auc_a = metrics["A_tabular"].roc_auc
    auc_b = metrics["B_tabular_plus_graph"].roc_auc
    lines = [
        f"# {run_name}: EXP04 and EXP05",
        "",
        f"Out-of-time test window: {len(test):,} loans.",
        "",
        comparison.round(5).to_markdown(),
        "",
        "## EXP04 - do graph-derived cohort features add signal beyond tabular features?",
        "",
        f"Arm A (tabular) ROC-AUC {auc_a:.4f}; arm B (tabular + cohort aggregates) "
        f"{auc_b:.4f}; difference **{auc_b - auc_a:+.4f}**.",
        "",
    ]
    share = metrics["B_tabular_plus_graph"].extras.get("graph_gain_share")
    if share is not None:
        lines.append(
            f"The {len(cohort_columns)} cohort features account for {100 * share:.1f}% of total "
            f"LightGBM gain in arm B."
        )
    lines.append(
        "**Verdict.** "
        + (
            "Relational information adds measurable signal beyond the borrower's own file."
            if auc_b - auc_a > 0.002
            else "Relational information adds little or nothing beyond the borrower's own file "
            "on this dataset. That is a legitimate result, and the more likely one for a "
            "shared-attribute cohort graph: geography and employment are already partly "
            "encoded in the tabular features (`addr_state`, `emp_length`), so the cohort "
            "aggregates are largely redundant with them."
        )
    )

    if "C_graphsage" in metrics:
        auc_c = metrics["C_graphsage"].roc_auc
        lines += [
            "",
            "## EXP05 - does GraphSAGE add value beyond the aggregates?",
            "",
            f"Arm C (GraphSAGE) ROC-AUC {auc_c:.4f}, against arm B at {auc_b:.4f} "
            f"(**{auc_c - auc_b:+.4f}**) and arm A at {auc_a:.4f} (**{auc_c - auc_a:+.4f}**).",
            f"Trained on `{gnn_summary.get('device', device)}`, best epoch "
            f"{gnn_summary.get('best_epoch')}, {gnn_summary.get('n_node_features')} node features.",
            "",
            "**Verdict.** "
            + (
                "The GNN improves on the hand-built aggregates, so the learned message passing "
                "is capturing structure the fixed aggregation does not."
                if auc_c - auc_b > 0.002
                else "The GNN does **not** beat the hand-built cohort aggregates. This is the "
                "expected outcome and worth stating plainly: mean-aggregation over a "
                "one-hop cohort is close to what the aggregate features already compute in "
                "closed form, so the GNN is re-learning a known function from far less "
                "signal, with more parameters. On a genuine transaction network -- where "
                "multi-hop structure carries information no aggregate captures -- the "
                "comparison could well go the other way. This dataset has no such network."
            ),
            "",
            "Practical note: arm B is also the more deployable of the two. Its contribution is "
            "legible in a SHAP plot and can be given to an applicant as a reason; the GNN's "
            "neighbourhood aggregation cannot, which matters under adverse-action requirements.",
        ]

    write_report(lines, paths, f"{run_name}_exp04_exp05", source)
    save_json(
        {
            "arms": {k: v.to_dict() for k, v in metrics.items()},
            "cohort_features": list(cohort_columns),
            "gnn": gnn_summary,
        },
        paths, f"{run_name}_exp04_exp05",
    )
    write_manifest(run_name, config, seed, paths, data_source=source, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
