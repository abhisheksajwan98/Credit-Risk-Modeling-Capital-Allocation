"""Time-respecting cohort graph construction.

What the graph is
-----------------
LendingClub scrubbed ``member_id`` and publishes no counterparty links, so there is no observed
transaction network here. What there *is* is shared-attribute structure: borrowers who live in the
same 3-digit ZIP prefix share a local labour and housing market, and borrowers with the same
employment string share an occupational or employer exposure. Credit risk is not i.i.d. across
such groups -- that is the entire reason lenders carry geographic and industry concentration
limits -- so the question "does cohort membership carry information beyond the applicant's own
file?" is a real one.

This is stated plainly as a **cohort graph**, not dressed up as an interbank network. What it can
test is whether correlated-risk structure helps. What it cannot test is whether a true counterparty
graph would help.

The direction of every edge
---------------------------
An edge ``j -> i`` exists only when ``issue_period[j] < issue_period[i]``: information flows
forward in time and never backward. Same-month loans are **not** connected, because within a month
we cannot establish ordering, and a mutual edge between two simultaneous applications would let
each see the other.

That constraint governs neighbour *features* only. Neighbour *labels* carry a second, stricter
condition handled in :mod:`credit_risk.graph.features` -- a neighbour's outcome is usable only once
it has actually resolved. Conflating those two information sets is the standard way graph credit
models leak, and keeping them separate is why this module and that one are separate files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.utils.runtime import get_logger

LOG = get_logger("graph.construction")

#: Sentinel written into the neighbour index where a node has fewer than ``max_neighbours``
#: eligible predecessors. Paired with a boolean mask so aggregation can ignore the padding.
PAD_INDEX = -1

#: Stand-in for a missing cohort key. Sorts after every real key and is skipped, so that
#: "no employer stated" never becomes a cohort of its own.
_NA_SENTINEL = "￿__missing__"


@dataclass(frozen=True)
class RelationSpec:
    """One relation: a column whose shared value defines a cohort."""

    name: str
    key_column: str
    max_neighbours: int = 10
    #: Cohorts larger than this are still used, but a node only ever sees the most recent
    #: ``max_neighbours`` predecessors, so a 20,000-loan ZIP does not swamp a 30-loan one.
    min_cohort_size: int = 2
    #: Optional recency window in months. ``None`` means "any predecessor".
    lookback_months: int | None = 36

    def __post_init__(self) -> None:
        if self.max_neighbours < 1:
            raise ValueError("max_neighbours must be at least 1")


DEFAULT_RELATIONS: tuple[RelationSpec, ...] = (
    # Primary relation. ~900 cohorts, a defensible economic story, and stable semantics
    # across the whole history.
    RelationSpec(name="zip3", key_column="zip3", max_neighbours=10, lookback_months=36),
    # Secondary relation, ablated separately. LendingClub's `emp_title` holds employer name
    # before 2013-09-23 and job title after, so this column changes meaning mid-history.
    RelationSpec(name="emp", key_column="emp_norm", max_neighbours=10, lookback_months=36),
)


@dataclass
class TimeRespectingGraph:
    """Sampled neighbourhoods stored densely.

    ``neighbour_index`` has shape ``(n_nodes, n_relations, max_neighbours)`` and ``neighbour_mask``
    matches it. A dense layout is used rather than a sparse edge list because every node has a
    fixed neighbour budget, which makes the whole structure a couple of int32 arrays that index
    directly in a GNN forward pass -- no sampling library, no scatter kernels, no CSR bookkeeping.
    At 620k nodes, 2 relations and K=10 this is about 50 MB.
    """

    neighbour_index: np.ndarray  # int32 (n_nodes, n_relations, K)
    neighbour_mask: np.ndarray  # bool  (n_nodes, n_relations, K)
    relations: tuple[RelationSpec, ...]
    node_order: np.ndarray  # int64 positional ids matching the frame the graph was built from
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.neighbour_index.shape[0])

    @property
    def n_relations(self) -> int:
        return int(self.neighbour_index.shape[1])

    @property
    def max_neighbours(self) -> int:
        return int(self.neighbour_index.shape[2])

    @property
    def n_edges(self) -> int:
        return int(self.neighbour_mask.sum())

    def degree(self, relation: int | None = None) -> np.ndarray:
        mask = self.neighbour_mask if relation is None else self.neighbour_mask[:, relation]
        return mask.reshape(self.n_nodes, -1).sum(axis=1)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            neighbour_index=self.neighbour_index,
            neighbour_mask=self.neighbour_mask,
            node_order=self.node_order,
            relation_names=np.array([r.name for r in self.relations], dtype=object),
            relation_keys=np.array([r.key_column for r in self.relations], dtype=object),
            relation_k=np.array([r.max_neighbours for r in self.relations]),
            relation_lookback=np.array(
                [-1 if r.lookback_months is None else r.lookback_months for r in self.relations]
            ),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> TimeRespectingGraph:
        with np.load(Path(path), allow_pickle=True) as data:
            relations = tuple(
                RelationSpec(
                    name=str(n),
                    key_column=str(k),
                    max_neighbours=int(kk),
                    lookback_months=None if int(lb) < 0 else int(lb),
                )
                for n, k, kk, lb in zip(
                    data["relation_names"],
                    data["relation_keys"],
                    data["relation_k"],
                    data["relation_lookback"],
                    strict=True,
                )
            )
            return TimeRespectingGraph(
                neighbour_index=data["neighbour_index"],
                neighbour_mask=data["neighbour_mask"],
                relations=relations,
                node_order=data["node_order"],
            )


def _period_to_int(series: pd.Series) -> np.ndarray:
    """Months since epoch as a plain int array, so ordering is a simple integer comparison."""
    periods = pd.PeriodIndex(series, freq="M")
    return (periods.year.to_numpy() * 12 + periods.month.to_numpy()).astype(np.int64)


def build_graph(
    df: pd.DataFrame,
    relations: tuple[RelationSpec, ...] = DEFAULT_RELATIONS,
    period_column: str = "issue_period",
) -> TimeRespectingGraph:
    """Build sampled, strictly backward-looking neighbourhoods.

    For each relation and each node, the neighbours are the ``max_neighbours`` most recent loans
    that share the cohort key and were issued in a **strictly earlier month** (optionally within
    ``lookback_months``).

    The implementation is a per-cohort ``searchsorted`` rather than a join. Within a cohort sorted
    by issue month, the number of strictly-earlier members is exactly
    ``searchsorted(months, month_i, side="left")``, so the eligible window for node ``i`` is the
    slice ending there. That is O(m log m) per cohort and needs no edge table -- which matters,
    because the naive self-join on ZIP would generate billions of pairs before any cap is applied.
    """
    n = len(df)
    k_max = max(r.max_neighbours for r in relations)
    n_rel = len(relations)

    neighbour_index = np.full((n, n_rel, k_max), PAD_INDEX, dtype=np.int32)
    neighbour_mask = np.zeros((n, n_rel, k_max), dtype=bool)

    months = _period_to_int(df[period_column])
    stats: dict[str, float] = {}

    for r_i, relation in enumerate(relations):
        if relation.key_column not in df.columns:
            LOG.warning(
                "relation %r skipped: column %r not present", relation.name, relation.key_column
            )
            continue

        # Missing keys must not form a cohort: two borrowers who both declined to state an
        # employer have nothing in common. They are replaced by a sentinel that sorts last and
        # then skipped, rather than left as NA (which makes argsort raise).
        key_series = df[relation.key_column].astype("string")
        valid_series = key_series.notna().to_numpy()
        keys = key_series.fillna(_NA_SENTINEL).to_numpy(dtype=object)
        n_linked = 0

        # Group positions by cohort key without materialising a DataFrame groupby, which is
        # measurably faster and keeps positional indices intact.
        order = np.argsort(keys, kind="stable")
        ordered_keys = keys[order]
        boundaries = np.flatnonzero(
            np.r_[True, ordered_keys[1:] != ordered_keys[:-1], True]
        )

        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            positions = order[start:stop]
            if len(positions) < relation.min_cohort_size or not valid_series[positions[0]]:
                continue

            # Sort cohort members by issue month; ties keep a stable, reproducible order.
            local_months = months[positions]
            time_order = np.argsort(local_months, kind="stable")
            positions = positions[time_order]
            local_months = local_months[time_order]

            # Number of strictly earlier members for each node -- the exclusive upper bound of
            # the eligible window. `side="left"` is what makes same-month loans ineligible.
            upper = np.searchsorted(local_months, local_months, side="left")

            if relation.lookback_months is not None:
                lower_bound_month = local_months - relation.lookback_months
                lower = np.searchsorted(local_months, lower_bound_month, side="left")
            else:
                lower = np.zeros_like(upper)

            take_from = np.maximum(lower, upper - relation.max_neighbours)
            counts = np.maximum(upper - take_from, 0)

            for local_i in np.flatnonzero(counts > 0):
                c = int(counts[local_i])
                window = positions[take_from[local_i] : upper[local_i]]
                # Most recent first, so a truncated neighbourhood keeps the freshest information.
                window = window[::-1][:c]
                node = positions[local_i]
                neighbour_index[node, r_i, :c] = window
                neighbour_mask[node, r_i, :c] = True
                n_linked += c

        degree = neighbour_mask[:, r_i].sum(axis=1)
        stats[f"{relation.name}_edges"] = float(n_linked)
        stats[f"{relation.name}_mean_degree"] = float(degree.mean())
        stats[f"{relation.name}_isolated_share"] = float((degree == 0).mean())
        LOG.info(
            "relation %-6s | %s edges | mean degree %.2f | isolated %.1f%%",
            relation.name,
            f"{n_linked:,}",
            degree.mean(),
            100.0 * (degree == 0).mean(),
        )

    graph = TimeRespectingGraph(
        neighbour_index=neighbour_index,
        neighbour_mask=neighbour_mask,
        relations=relations,
        node_order=np.arange(n, dtype=np.int64),
        stats=stats,
    )
    LOG.info(
        "graph built: %s nodes, %s edges, %d relations",
        f"{graph.n_nodes:,}",
        f"{graph.n_edges:,}",
        graph.n_relations,
    )
    return graph


def assert_no_future_edges(
    graph: TimeRespectingGraph, df: pd.DataFrame, period_column: str = "issue_period"
) -> None:
    """Fail if any edge points from the present or future into the past.

    This is the graph equivalent of the post-origination deny-list, and it is called from
    ``tests/test_graph_leakage.py``. It checks the property directly on the built structure
    rather than trusting the construction code, because the construction code is exactly what
    a future edit might break.
    """
    months = _period_to_int(df[period_column])
    idx = graph.neighbour_index
    mask = graph.neighbour_mask

    target_months = np.broadcast_to(
        months[:, None, None], idx.shape
    )
    safe_idx = np.where(mask, idx, 0)
    neighbour_months = months[safe_idx]

    violations = mask & (neighbour_months >= target_months)
    if violations.any():
        count = int(violations.sum())
        first = np.argwhere(violations)[0]
        node, rel, slot = int(first[0]), int(first[1]), int(first[2])
        raise AssertionError(
            f"{count} edges violate the time-respecting constraint. Example: node {node} "
            f"(month {months[node]}) links to node {idx[node, rel, slot]} "
            f"(month {neighbour_months[node, rel, slot]}) via relation "
            f"{graph.relations[rel].name}. Neighbours must be strictly earlier."
        )

    self_loops = mask & (idx == np.arange(len(df))[:, None, None])
    if self_loops.any():
        raise AssertionError(f"{int(self_loops.sum())} self-loops found; a node cannot be its own neighbour")
