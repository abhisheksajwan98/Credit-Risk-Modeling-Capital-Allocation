"""Blind-spot signal analysis and the capital-constrained portfolio comparison.

These tests exist because of specific defects found in the audit, and each one would have caught
the defect it names:

* EXP11 computed "realised profit" from an assumed ``LGD=0.5`` formula rather than from observed
  cashflows, understating book profit by roughly 2.4x and inverting the headline result.
* It compared the two *swap sets* rather than the two *portfolios*, so the difference reflected
  unequal capital as much as better selection.
* Its report asserted an improvement regardless of the sign of the number it had just printed.
* EXP08/EXP10 supported a "blind spot" claim from a marginal quantile table alone, which cannot
  distinguish a real signal from one that merely tracks the PD level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.decision.blindspot import (
    build_portfolio,
    compare_capital_constrained,
    conditional_signal_table,
    greedy_allocate,
    marginal_signal_table,
)
from credit_risk.decision.expected_loss import CashflowModel, EconomicsConfig, realised_profit


@pytest.fixture
def cashflow() -> CashflowModel:
    return CashflowModel(
        good_slope=0.3, good_intercept=0.17, bad_slope=-0.4, bad_intercept=0.55,
        n_good=1000, n_bad=200,
    )


def _book(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """A small loan book with realised cashflows consistent with its default flags."""
    rng = np.random.default_rng(seed)
    amount = rng.choice([5_000.0, 10_000.0, 20_000.0], size=n)
    apr = rng.uniform(0.06, 0.28, size=n)
    default = (rng.random(n) < 0.15).astype(float)

    # Good loans return roughly apr * WAL; bad loans lose roughly half the principal.
    gross = np.where(default > 0.5, -0.5 * amount, apr * 1.55 * amount)
    return pd.DataFrame(
        {
            "loan_amnt": amount,
            "funded_amnt": amount,
            "int_rate": apr * 100.0,
            "default": default,
            "total_pymnt": amount + gross,
            "recoveries": np.zeros(n),
            "collection_recovery_fee": np.zeros(n),
        }
    )


# ---------------------------------------------------------------------------
# Profit must come from observed cashflows
# ---------------------------------------------------------------------------

def test_portfolio_profit_uses_observed_cashflow_not_an_assumed_lgd():
    """The defect that inverted EXP11: profit modelled from an assumed LGD instead of measured."""
    df = _book()
    approve = pd.Series(True, index=df.index)
    portfolio = build_portfolio(df, approve, "all", budget=df["funded_amnt"].sum())

    expected = float(realised_profit(df).sum())
    assert portfolio.observed_profit == pytest.approx(expected)

    # The old formula, for contrast. It must NOT match, which is the whole point.
    formula = (
        df["loan_amnt"] * df["int_rate"] / 100.0 * 1.55
        - df["loan_amnt"] * 0.03 * 1.55
        - (150.0 + df["loan_amnt"] * 0.01)
        - df["loan_amnt"] * 0.5 * df["default"]
    ).sum()
    assert abs(formula - expected) > 0.05 * abs(expected)


def test_return_on_capital_is_profit_over_capital():
    df = _book()
    approve = pd.Series(True, index=df.index)
    portfolio = build_portfolio(df, approve, "all", budget=1.0)
    assert portfolio.return_on_capital == pytest.approx(
        portfolio.observed_profit / portfolio.capital_deployed
    )
    assert portfolio.capital_deployed == pytest.approx(df["funded_amnt"].sum())


def test_empty_book_does_not_divide_by_zero():
    df = _book(200)
    portfolio = build_portfolio(df, pd.Series(False, index=df.index), "none", budget=1000.0)
    assert portfolio.n_loans == 0
    assert portfolio.capital_deployed == 0.0
    assert np.isnan(portfolio.return_on_capital)


# ---------------------------------------------------------------------------
# Capital constraint
# ---------------------------------------------------------------------------

def test_greedy_allocation_respects_the_budget():
    df = _book()
    density = pd.Series(np.random.default_rng(1).uniform(-0.05, 0.15, len(df)), index=df.index)
    budget = 5_000_000.0
    approve = greedy_allocate(density, df["funded_amnt"], budget)
    assert df.loc[approve, "funded_amnt"].sum() <= budget


def test_greedy_allocation_never_funds_negative_density():
    df = _book()
    density = pd.Series(np.linspace(-0.2, 0.2, len(df)), index=df.index)
    approve = greedy_allocate(density, df["funded_amnt"], budget=1e12)
    assert (density[approve] > 0).all()


def test_greedy_allocation_prefers_higher_density():
    """With a binding budget the funded set must be the top of the density ranking."""
    df = _book(500)
    density = pd.Series(np.random.default_rng(2).uniform(0.01, 0.2, len(df)), index=df.index)
    budget = df["funded_amnt"].sum() * 0.3
    approve = greedy_allocate(density, df["funded_amnt"], budget)
    assert density[approve].min() >= density[~approve].max() - 1e-9


def test_both_policies_are_compared_at_the_same_budget(cashflow):
    """Unequal budgets would make the profit difference meaningless."""
    df = _book(5000, seed=3)
    rng = np.random.default_rng(4)
    pd_hat = np.clip(0.15 + 0.05 * rng.standard_normal(len(df)), 0.01, 0.9)
    flagged = rng.random(len(df)) < 0.1

    analysis = compare_capital_constrained(
        df, pd_hat, flagged, cashflow, EconomicsConfig(), budget_fraction=0.7
    )
    for portfolio in analysis.portfolios:
        assert portfolio.capital_deployed <= analysis.budget + 1e-6
    assert len(analysis.portfolios) == 2


def test_verdict_reports_no_material_difference_when_return_is_unchanged(cashflow):
    """The defect: asserting an improvement regardless of the sign of the number printed."""
    df = _book(5000, seed=5)
    rng = np.random.default_rng(6)
    pd_hat = np.clip(0.15 + 0.05 * rng.standard_normal(len(df)), 0.01, 0.9)
    # Flag at random, so the two policies should be economically indistinguishable.
    flagged = rng.random(len(df)) < 0.1

    analysis = compare_capital_constrained(
        df, pd_hat, flagged, cashflow, EconomicsConfig(), budget_fraction=0.7
    )
    verdict = analysis.verdict("B_blindspot_aware", "A_pd_ranked")
    a, b = analysis.portfolios
    if abs(b.return_on_capital - a.return_on_capital) < 0.005:
        assert "no material difference" in verdict
    # And the verdict must never claim an increase while reporting a decrease.
    if b.observed_profit < a.observed_profit:
        assert "net increase in profit" not in verdict


# ---------------------------------------------------------------------------
# Signal analysis: marginal vs conditional
# ---------------------------------------------------------------------------

def test_a_signal_that_is_pure_noise_fails_the_conditional_test():
    """Random noise must not be reported as a blind-spot signal."""
    rng = np.random.default_rng(7)
    n = 60_000
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    noise = rng.standard_normal(n)

    result = conditional_signal_table(y, p, noise, n_strata=10)
    assert result.sign_test_p > 0.05
    assert abs(result.odds_ratio_per_sd - 1.0) < 0.05
    assert "does not survive" in result.verdict("noise")


def test_a_signal_that_only_restates_pd_fails_the_conditional_test():
    """The confound the marginal table cannot detect.

    A signal that is a monotone function of the PD will look impressive in a quantile table -- the
    high bins have higher default rates and worse calibration -- while carrying no information the
    PD does not already have. Conditioning on PD must reveal that.
    """
    rng = np.random.default_rng(8)
    n = 60_000
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    signal = p * 3.0  # a pure restatement of the PD

    marginal = marginal_signal_table(y, p, signal, n_bins=5)
    assert marginal["default_rate"].is_monotonic_increasing  # looks like a finding

    result = conditional_signal_table(y, p, signal, n_strata=10)
    assert result.spearman_with_pd > 0.95
    assert result.sign_test_p > 0.05  # but it is not one
    assert "does not survive" in result.verdict("pd restatement")


def test_a_genuine_signal_passes_the_conditional_test():
    """A signal that really does mark where the model under-states risk must be detected."""
    rng = np.random.default_rng(9)
    n = 80_000
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    hidden = rng.random(n) < 0.3          # the model does not see this
    true_p = np.clip(p + 0.06 * hidden, 1e-4, 1 - 1e-4)
    y = (rng.random(n) < true_p).astype(float)
    signal = hidden + 0.35 * rng.standard_normal(n)   # a noisy read on the hidden factor

    result = conditional_signal_table(y, p, signal, n_strata=10)
    assert result.sign_test_p < 0.05
    assert result.mean_gap_difference > 0
    assert result.odds_ratio_per_sd > 1.0
    assert "survives" in result.verdict("genuine")


def test_conditional_table_is_well_formed():
    rng = np.random.default_rng(10)
    n = 40_000
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    result = conditional_signal_table(y, p, rng.standard_normal(n), n_strata=8)

    assert result.n_strata <= 8
    assert result.n_strata_positive <= result.n_strata
    assert 0.0 <= result.sign_test_p <= 1.0
    expected = result.table["gap_high"] - result.table["gap_low"]
    assert np.allclose(result.table["gap_difference"], expected)


def test_conditional_test_requires_enough_rows_per_stratum():
    rng = np.random.default_rng(11)
    n = 300
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    with pytest.raises(ValueError, match="enough rows"):
        conditional_signal_table(y, p, rng.standard_normal(n), n_strata=10, min_per_half=200)


def test_marginal_table_reports_the_calibration_gap():
    rng = np.random.default_rng(12)
    n = 20_000
    p = np.clip(rng.beta(2, 10, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < p).astype(float)
    table = marginal_signal_table(y, p, rng.standard_normal(n), n_bins=5)

    assert len(table) == 5
    assert table["n"].sum() == n
    assert np.allclose(
        table["calibration_gap"], table["default_rate"] - table["mean_pd"], atol=1e-9
    )
