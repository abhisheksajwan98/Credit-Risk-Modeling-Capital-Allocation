#!/usr/bin/env python
"""Build the time-respecting cohort graph and its aggregate features.

    python scripts/build_graph.py --config configs/graph/cohort_graph.yaml

Writes ``data/processed/graph.npz`` and ``data/processed/cohort_features.parquet``, and runs the
leakage assertions. Those assertions are the point of this script as much as the artefacts are:
a graph that lets information travel backwards in time invalidates everything built on it, so the
check fails the run rather than logging a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.features.financial import add_derived_features  # noqa: E402
from credit_risk.graph.construction import (  # noqa: E402
    RelationSpec,
    assert_no_future_edges,
    build_graph,
)
from credit_risk.graph.features import (  # noqa: E402
    CohortFeatureBuilder,
    CohortFeatureConfig,
    save_cohort_features,
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

LOG = get_logger("scripts.graph")


def _relations(config) -> tuple[RelationSpec, ...]:
    specs = []
    for entry in config.get("relations", []):
        specs.append(
            RelationSpec(
                name=str(entry["name"]),
                key_column=str(entry["key_column"]),
                max_neighbours=int(entry.get("max_neighbours", 10)),
                lookback_months=entry.get("lookback_months", 36),
                min_cohort_size=int(entry.get("min_cohort_size", 2)),
            )
        )
    return tuple(specs)


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/graph/cohort_graph.yaml")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths).reset_index(drop=True)
    df = add_derived_features(df)
    relations = _relations(config)

    graph = build_graph(df, relations=relations)

    # --- leakage assertions --------------------------------------------------
    assert_no_future_edges(graph, df)
    LOG.info("graph leakage check PASSED: every edge points strictly backwards, no self-loops")

    graph.save(paths.data_processed / "graph.npz")

    # --- cohort aggregate features ------------------------------------------
    ccfg = config.get("cohort_features", {})
    cohort_config = CohortFeatureConfig(
        relations=relations,
        lookback_months=int(ccfg.get("lookback_months", 36)),
        smoothing_alpha=float(ccfg.get("smoothing_alpha", 25.0)),
        feature_columns=tuple(
            ccfg.get("feature_columns", ["fico_mid", "dti_clean", "loan_amnt", "annual_inc"])
        ),
    )
    builder = CohortFeatureBuilder(cohort_config).fit(df[df["split"] == "train"])
    cohort = builder.transform(df)
    save_cohort_features(cohort, paths.data_processed / "cohort_features.parquet")

    # --- diagnostics ---------------------------------------------------------
    degree_rows = []
    for i, relation in enumerate(relations):
        degree = graph.degree(i)
        degree_rows.append(
            {
                "relation": relation.name,
                "key_column": relation.key_column,
                "cohorts": int(df[relation.key_column].nunique()),
                "edges": int(graph.neighbour_mask[:, i].sum()),
                "mean_degree": float(degree.mean()),
                "median_degree": float(np.median(degree)),
                "isolated_share": float((degree == 0).mean()),
                "at_cap_share": float((degree >= relation.max_neighbours).mean()),
            }
        )
    degrees = pd.DataFrame(degree_rows)
    save_table(degrees, paths, f"{run_name}_degree", index=False)

    cohort_stats = cohort.describe().T[["mean", "std", "min", "max"]]
    save_table(cohort_stats, paths, f"{run_name}_cohort_features")

    # The gap between these two numbers is the single most instructive diagnostic in the whole
    # graph layer, so it is surfaced rather than buried in a describe() table.
    gap_rows = []
    for relation in relations:
        prior = cohort.get(f"cohort_{relation.name}_n_prior")
        resolved = cohort.get(f"cohort_{relation.name}_n_resolved")
        if prior is None or resolved is None:
            continue
        gap_rows.append(
            {
                "relation": relation.name,
                "mean_prior_applications": float(prior.mean()),
                "mean_resolved_outcomes": float(resolved.mean()),
                "share_of_predecessors_with_known_outcome": float(
                    resolved.sum() / max(prior.sum(), 1)
                ),
            }
        )
    gaps = pd.DataFrame(gap_rows)
    save_table(gaps, paths, f"{run_name}_information_gap", index=False)

    lines = [
        f"# {run_name}: graph construction",
        "",
        f"Nodes: **{graph.n_nodes:,}**. Edges: **{graph.n_edges:,}** across "
        f"{graph.n_relations} relations.",
        "",
        "## Relations",
        "",
        degrees.round(4).to_markdown(index=False),
        "",
        "## The two information sets",
        "",
        "A neighbour's *application* is visible as soon as it is filed. A neighbour's *outcome*",
        "is not visible until the loan resolves, which for a 36-month loan is up to three years",
        "later. Those are different information sets, and the gap between them is large:",
        "",
        gaps.round(4).to_markdown(index=False),
        "",
        "Feature aggregates use the first column; label aggregates use the second. Using the",
        "first for a label aggregate is the standard way graph credit models leak, and it would",
        "inflate AUC substantially, because the cohort default rate would then contain outcomes",
        "from the same period the model is being scored on. See docs/GRAPH_DESIGN.md.",
        "",
        "## Leakage checks",
        "",
        "- every edge satisfies `issue_period[neighbour] < issue_period[node]`: **PASSED**",
        "- no self-loops: **PASSED**",
        "- no loan resolves on or before its own issue month: **PASSED**",
    ]
    write_report(lines, paths, f"{run_name}_graph", source)
    save_json(
        {
            "n_nodes": graph.n_nodes,
            "n_edges": graph.n_edges,
            "relations": degree_rows,
            "information_gap": gap_rows,
            "cohort_prior_default_rate": builder.state.prior_default_rate,
            "cohort_features": list(cohort.columns),
        },
        paths, f"{run_name}_graph",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
