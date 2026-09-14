"""Credit economics and the static policy ladder."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    breakeven_pd,
    expected_loss,
    expected_profit,
    fit_cashflow_model,
    realised_profit,
    realised_return,
    weighted_average_life,
)
from credit_risk.decision.policies import (
    ActionSpace,
    ApproveAllPolicy,
    DecisionContext,
    ExpectedProfitPolicy,
    FixedThresholdPolicy,
    RiskBandPolicy,
)


# ---------------------------------------------------------------------------
# Amortisation
# ---------------------------------------------------------------------------

def test_weighted_average_life_is_well_under_the_term():
    """The average dollar is outstanding for far less than the full term.

    Using the full 3 years would roughly double the funding cost, which would flow straight into
    every break-even calculation.
    """
    wal = weighted_average_life(0.13, 36)
    assert 1.3 < wal < 1.8
    assert wal < 3.0


def test_weighted_average_life_rises_with_term():
    assert weighted_average_life(0.13, 60) > weighted_average_life(0.13, 36)


# ---------------------------------------------------------------------------
# Realised cashflows: arithmetic, not modelling
# ---------------------------------------------------------------------------

def test_realised_profit_is_arithmetic(loans):
    """Recoveries are reported separately from `total_pymnt` and must be added, not assumed in."""
    import pandas as pd

    frame = loans.head(200)
    expected = (
        pd.to_numeric(frame["total_pymnt"], errors="coerce").fillna(0)
        + pd.to_numeric(frame["recoveries"], errors="coerce").fillna(0)
        - pd.to_numeric(frame["collection_recovery_fee"], errors="coerce").fillna(0)
        - pd.to_numeric(frame["funded_amnt"], errors="coerce")
    )
    assert np.allclose(realised_profit(frame).to_numpy(), expected.to_numpy(), equal_nan=True)


def test_fully_paid_loans_are_profitable_on_average(split_frames):
    train = split_frames["train"]
    good = train[train["default"] == 0]
    assert realised_return(good).mean() > 0


def test_defaulted_loans_lose_money_on_average(split_frames):
    train = split_frames["train"]
    bad = train[train["default"] == 1]
    assert realised_return(bad).mean() < 0


def test_empirical_lgd_is_estimated_not_assumed(split_frames):
    """LGD comes out of observed cashflows rather than a round number off a slide."""
    model = fit_cashflow_model(split_frames["train"])
    assert model.n_bad > 100
    assert 0.2 < model.observed_lgd_mean < 1.0
    # The fitted intercept must have moved off the configured fallback.
    assert model.bad_intercept != pytest.approx(EconomicsConfig().fallback_bad_loss_rate)


def test_higher_rate_loans_lose_less_when_they_default(split_frames):
    """A higher-priced loan collects more interest before defaulting, so its net loss is smaller.

    A positive `bad_slope` would mean the opposite and would signal the fit is picking up
    something other than the intended relationship.
    """
    model = fit_cashflow_model(split_frames["train"])
    assert model.bad_slope < 0


# ---------------------------------------------------------------------------
# Expected profit and break-even
# ---------------------------------------------------------------------------

@pytest.fixture
def cashflow() -> CashflowModel:
    return CashflowModel(
        good_slope=0.3, good_intercept=0.17, bad_slope=-0.5, bad_intercept=0.60,
        n_good=1000, n_bad=200,
    )


def test_expected_loss_matches_the_textbook_identity():
    assert expected_loss(0.1, 10_000, 0.6) == pytest.approx(600.0)


def test_expected_profit_falls_as_pd_rises(cashflow):
    profits = expected_profit(
        np.array([0.02, 0.10, 0.30, 0.60]), 10_000.0, 0.15, cashflow, EconomicsConfig()
    )
    assert np.all(np.diff(profits) < 0)


def test_expected_profit_rises_with_price(cashflow):
    profits = expected_profit(
        0.12, 10_000.0, np.array([0.07, 0.14, 0.24]), cashflow, EconomicsConfig()
    )
    assert np.all(np.diff(profits) > 0)


def test_breakeven_pd_zeroes_expected_profit(cashflow):
    """The break-even PD must actually be where expected profit crosses zero."""
    economics = EconomicsConfig()
    for apr in (0.08, 0.15, 0.25):
        exposure = 12_000.0
        threshold = float(breakeven_pd(apr, exposure, cashflow, economics))
        at = expected_profit(threshold, exposure, apr, cashflow, economics)
        assert abs(float(at)) < 1.0


def test_breakeven_pd_rises_with_price(cashflow):
    """This is the whole argument against a fixed PD cut-off.

    A 26% APR loan can carry far more risk than a 7% one and still be profitable, so any single
    threshold is simultaneously too tight at the top of the book and too loose at the bottom.
    """
    thresholds = breakeven_pd(
        np.array([0.07, 0.13, 0.20, 0.26]), 12_000.0, cashflow, EconomicsConfig()
    )
    assert np.all(np.diff(thresholds) > 0)
    assert thresholds[-1] > thresholds[0] + 0.05


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

def test_action_space_exposures_respect_the_cap():
    space = ActionSpace(max_exposure=20_000.0)
    exposures = space.exposures(np.array([40_000.0]))
    assert exposures.max() <= 20_000.0


def test_reject_action_has_zero_exposure():
    space = ActionSpace()
    exposures = space.exposures(np.array([10_000.0]))
    assert exposures[0, space.reject_index] == 0.0


def test_without_review_removes_only_the_review_action():
    full = ActionSpace()
    trimmed = full.without_review()
    assert trimmed.review_index is None
    assert set(trimmed.names) == set(full.names) - {"review"}
    assert trimmed.reject_index == trimmed.names.index("reject")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

@pytest.fixture
def context() -> DecisionContext:
    return DecisionContext(
        pd_estimate=np.array([0.01, 0.08, 0.14, 0.22, 0.45]),
        requested_amount=np.full(5, 12_000.0),
        apr=np.array([0.07, 0.11, 0.15, 0.20, 0.28]),
    )


def test_fixed_threshold_approves_below_and_declines_above(context):
    space = ActionSpace().without_review()
    policy = FixedThresholdPolicy(0.15, space)
    actions = policy.act(context)
    approved = actions != space.reject_index
    assert approved.tolist() == [True, True, True, False, False]


def test_risk_band_lends_less_as_risk_rises(context):
    space = ActionSpace().without_review()
    policy = RiskBandPolicy(action_space=space)
    exposures = space.exposures(context.requested_amount)[
        np.arange(len(context)), policy.act(context)
    ]
    assert np.all(np.diff(exposures) <= 0)


def test_expected_profit_declines_negative_value_applicants(cashflow, context):
    space = ActionSpace().without_review()
    policy = ExpectedProfitPolicy(cashflow, EconomicsConfig(), space)
    values = policy.action_values(context)
    actions = policy.act(context)
    for i, action in enumerate(actions):
        if action == space.reject_index:
            assert values[i].max() <= policy.min_profit + 1e-9


def test_expected_profit_picks_the_argmax(cashflow, context):
    space = ActionSpace().without_review()
    policy = ExpectedProfitPolicy(cashflow, EconomicsConfig(), space)
    values = policy.action_values(context)
    actions = policy.act(context)
    for i, action in enumerate(actions):
        if values[i].max() > 0:
            assert values[i][action] == pytest.approx(values[i].max())


def test_shadow_price_makes_the_policy_more_selective(cashflow, context):
    """Charging for capital must never approve *more* than charging nothing for it."""
    space = ActionSpace().without_review()
    free = ExpectedProfitPolicy(cashflow, EconomicsConfig(), space, capital_shadow_price=0.0)
    priced = ExpectedProfitPolicy(cashflow, EconomicsConfig(), space, capital_shadow_price=0.06)
    exposures = space.exposures(context.requested_amount)
    free_capital = exposures[np.arange(len(context)), free.act(context)].sum()
    priced_capital = exposures[np.arange(len(context)), priced.act(context)].sum()
    assert priced_capital <= free_capital


def test_approve_all_approves_everything(context):
    space = ActionSpace().without_review()
    policy = ApproveAllPolicy(space)
    assert (policy.act(context) != space.reject_index).all()
