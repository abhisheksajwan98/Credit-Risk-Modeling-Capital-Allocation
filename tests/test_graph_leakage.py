"""Graph leakage controls.

The graph is where a credit model leaks most easily and least visibly, because the leak travels
along an edge rather than sitting in a column. Two rules, tested independently:

1. **Every edge points strictly backwards in time.** Neighbour features may only come from
   applications that already existed.
2. **Neighbour labels require the neighbour's outcome to have resolved.** This is much stricter
   than rule 1 -- a 36-month loan issued in 2013 does not resolve until 2016 -- and conflating the
   two is the standard failure.

The final test in this file demonstrates the failure quantitatively rather than asserting it:
build the same cohort default rate using rule 1 instead of rule 2 and watch its correlation with
the target jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.graph.construction import (
    PAD_INDEX,
    RelationSpec,
    assert_no_future_edges,
    build_graph,
)
from credit_risk.graph.features import CohortFeatureBuilder, CohortFeatureConfig


@pytest.fixture(scope="module")
def graph(enriched):
    return build_graph(enriched)


# ---------------------------------------------------------------------------
# Rule 1: edges point backwards
# ---------------------------------------------------------------------------

def test_no_future_edges(graph, enriched):
    assert_no_future_edges(graph, enriched)


def test_edges_are_strictly_earlier_not_merely_not_later(graph, enriched):
    """Same-month loans must not be connected.

    Within a month there is no ordering to appeal to, so a mutual edge would let two simultaneous
    applications each see the other.
    """
    months = (
        pd.PeriodIndex(enriched["issue_period"]).year * 12
        + pd.PeriodIndex(enriched["issue_period"]).month
    ).to_numpy()
    idx, mask = graph.neighbour_index, graph.neighbour_mask
    neighbour_months = months[np.where(mask, idx, 0)]
    target_months = np.broadcast_to(months[:, None, None], idx.shape)
    assert not (mask & (neighbour_months == target_months)).any()


def test_no_self_loops(graph):
    positions = np.arange(graph.n_nodes)[:, None, None]
    assert not (graph.neighbour_mask & (graph.neighbour_index == positions)).any()


def test_padding_is_masked_out(graph):
    """Every padded slot must be masked; every masked-in slot must hold a real index."""
    padded = graph.neighbour_index == PAD_INDEX
    assert not (padded & graph.neighbour_mask).any()
    real = graph.neighbour_mask
    assert (graph.neighbour_index[real] >= 0).all()
    assert (graph.neighbour_index[real] < graph.n_nodes).all()


def test_neighbours_share_the_cohort_key(graph, enriched):
    for r, relation in enumerate(graph.relations):
        keys = enriched[relation.key_column].astype("string").to_numpy()
        nodes, slots = np.nonzero(graph.neighbour_mask[:, r, :])
        if len(nodes) == 0:
            continue
        sample = np.random.default_rng(0).choice(len(nodes), size=min(500, len(nodes)),
                                                 replace=False)
        for i in sample:
            node = nodes[i]
            neighbour = graph.neighbour_index[node, r, slots[i]]
            assert keys[node] == keys[neighbour]


def test_degree_respects_the_cap():
    spec = RelationSpec(name="zip3", key_column="zip3", max_neighbours=3, lookback_months=None)
    df = pd.DataFrame(
        {
            "zip3": ["100"] * 10,
            "issue_period": pd.period_range("2012-01", periods=10, freq="M"),
        }
    )
    graph = build_graph(df, relations=(spec,))
    assert graph.degree().max() <= 3


def test_missing_cohort_key_forms_no_edges():
    """Two borrowers who both declined to state an employer have nothing in common."""
    spec = RelationSpec(name="emp", key_column="emp_norm", max_neighbours=5, lookback_months=None)
    df = pd.DataFrame(
        {
            "emp_norm": [None] * 6,
            "issue_period": pd.period_range("2012-01", periods=6, freq="M"),
        }
    )
    graph = build_graph(df, relations=(spec,))
    assert graph.n_edges == 0


def test_lookback_window_is_enforced():
    spec = RelationSpec(name="zip3", key_column="zip3", max_neighbours=50, lookback_months=6)
    df = pd.DataFrame(
        {
            "zip3": ["100"] * 24,
            "issue_period": pd.period_range("2012-01", periods=24, freq="M"),
        }
    )
    graph = build_graph(df, relations=(spec,))
    # The last node may look back at most six months, so at most six predecessors.
    assert graph.degree()[-1] <= 6


def test_assert_no_future_edges_catches_a_planted_violation(graph, enriched):
    """The check must fail on a corrupted graph, not merely pass on a correct one.

    A guard that has only ever been shown passing inputs is not known to be a guard.
    """
    import copy

    periods = pd.PeriodIndex(enriched["issue_period"])
    earliest = int(np.argmin(periods.asi8))
    latest = int(np.argmax(periods.asi8))
    assert periods[latest] > periods[earliest]

    broken = copy.deepcopy(graph)
    broken.neighbour_index[earliest, 0, 0] = latest
    broken.neighbour_mask[earliest, 0, 0] = True
    with pytest.raises(AssertionError, match="time-respecting"):
        assert_no_future_edges(broken, enriched)


# ---------------------------------------------------------------------------
# Rule 2: labels require resolution
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cohort(enriched):
    builder = CohortFeatureBuilder(CohortFeatureConfig()).fit(enriched[enriched["split"] == "train"])
    return builder.transform(enriched)


def test_counts_are_well_formed(cohort):
    """Both counts are finite and non-negative.

    Note what is deliberately *not* asserted here: ``n_resolved <= n_prior``. That looks like it
    ought to hold and does not, because the two window on different time axes -- ``n_prior`` on
    the issue date (cohort activity) and ``n_resolved`` on the resolution date (cohort
    experience). A neighbour issued forty months ago falls outside the first window while its
    outcome, resolved ten months ago, falls inside the second. See graph/features.py.
    """
    for relation in ("zip3", "emp"):
        for suffix in ("n_prior", "n_resolved"):
            values = cohort[f"cohort_{relation}_{suffix}"].to_numpy()
            assert np.isfinite(values).all()
            assert (values >= 0).all()


def test_label_information_set_is_strictly_smaller(cohort):
    """The gap is the whole point: most predecessors have not resolved yet at decision time."""
    prior = cohort["cohort_zip3_n_prior"].sum()
    resolved = cohort["cohort_zip3_n_resolved"].sum()
    assert resolved < prior, "label gating is not restricting anything -- check the as-of logic"


def test_cohort_rate_is_bounded_and_shrunk(cohort):
    rate = cohort["cohort_zip3_default_rate"]
    assert rate.between(0.0, 1.0).all()
    # With no resolved neighbours the rate must equal the prior exactly, not 0 or NaN.
    isolated = cohort["cohort_zip3_n_resolved"] == 0
    if isolated.any():
        assert rate[isolated].std() < 1e-6


def test_evidence_weight_matches_resolved_count(cohort):
    alpha = CohortFeatureConfig().smoothing_alpha
    resolved = cohort["cohort_zip3_n_resolved"].to_numpy()
    expected = resolved / (resolved + alpha)
    assert np.allclose(cohort["cohort_zip3_evidence_weight"].to_numpy(), expected, atol=1e-5)


def test_cohort_builder_rejects_resolution_before_issue(enriched):
    """Corrupt the observation date and the builder must refuse to compute anything."""
    broken = enriched.copy()
    broken["outcome_observed_period"] = broken["issue_period"]
    builder = CohortFeatureBuilder(CohortFeatureConfig()).fit(broken[broken["split"] == "train"])
    with pytest.raises(AssertionError, match="resolve on or before"):
        builder.transform(broken)


def test_ungated_cohort_rate_leaks_measurably():
    """Quantify the failure this design prevents, on a controlled example.

    A purpose-built frame is used rather than the shared fixture, because the demonstration needs
    dense cohorts and a cohort risk level that actually moves.

    The construction matters. Each cohort's default rate is redrawn independently every 24 months,
    so what a cohort's risk was three years ago says nothing about what it is now. The aggregation
    window is short (6 months) so that the average tracks the current level rather than smoothing
    across several eras -- with a long window both versions average out to the portfolio mean and
    the comparison shows nothing, which is a real trap in designing this kind of check.

    Gated correctly, a borrower sees outcomes that resolved before they applied: for a 36-month
    loan, that is the era-before-last. Gated naively on the application date, they effectively see
    *this* era's outcomes -- the same period the model is scored on. The naive version should be
    conspicuously more correlated with the target, and all of that extra correlation is leakage.
    """
    rng = np.random.default_rng(7)
    n_cohorts, n_months, per_month, era_length = 10, 120, 14, 24
    n_eras = n_months // era_length + 2
    # Independent risk level per (cohort, era): the past predicts nothing about the present.
    era_rate = rng.uniform(0.05, 0.45, size=(n_cohorts, n_eras))

    rows = []
    start_period = pd.Period("2010-01", freq="M")
    for m in range(n_months):
        period = start_period + m
        era = m // era_length
        for c in range(n_cohorts):
            for _ in range(per_month):
                default = float(rng.random() < era_rate[c, era])
                rows.append(
                    {
                        "zip3": f"{c:03d}",
                        "issue_period": period,
                        "default": default,
                        "default_observed": default,
                        "outcome_observed_period": period + 36,
                        "split": "train" if m < n_months // 2 else "test",
                    }
                )

    frame = pd.DataFrame(rows)
    frame["issue_period"] = pd.PeriodIndex(frame["issue_period"], freq="M")
    frame["outcome_observed_period"] = pd.PeriodIndex(frame["outcome_observed_period"], freq="M")
    y = frame["default"].to_numpy()

    config = CohortFeatureConfig(
        relations=(RelationSpec(name="zip3", key_column="zip3", lookback_months=6),),
        lookback_months=6,
        smoothing_alpha=5.0,
        feature_columns=(),
    )
    builder = CohortFeatureBuilder(config).fit(frame[frame["split"] == "train"])
    gated = builder.transform(frame)["cohort_zip3_default_rate"].to_numpy()

    # The naive version: pretend a loan's outcome is known one month after origination.
    naive_frame = frame.copy()
    naive_frame["outcome_observed_period"] = frame["issue_period"] + 1
    naive = builder.transform(naive_frame)["cohort_zip3_default_rate"].to_numpy()

    gated_corr = abs(np.corrcoef(gated, y)[0, 1])
    naive_corr = abs(np.corrcoef(naive, y)[0, 1])
    assert naive_corr > gated_corr + 0.10, (
        f"expected the ungated cohort rate to be substantially more correlated with the target "
        f"(naive {naive_corr:.4f} vs gated {gated_corr:.4f}). If it is not, the resolution "
        f"gating is not restricting anything."
    )
