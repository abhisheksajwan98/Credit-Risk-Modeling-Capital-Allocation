"""Credit economics: from a probability of default to an expected profit.

The distinction this module exists to enforce is between **observed**, **estimated** and
**assumed** quantities. Portfolio simulations go wrong when an invented number is quietly given the
same status as a measured one, so each is labelled here and in ``docs/ASSUMPTIONS.md``.

**Observed** (read directly from the data)
    ``funded_amnt``, ``int_rate``, ``term``, ``total_pymnt``, ``total_rec_prncp``,
    ``total_rec_int``, ``recoveries``, ``collection_recovery_fee``. From these the realised profit
    of every matured loan is an arithmetic fact, not a model.

**Estimated** (fitted on the training window from observed cashflows)
    The good-loan yield ``y_good(apr)`` and the loss rate on defaulted loans ``l_bad(apr)``.
    Notably this gives an **empirical LGD** rather than the round number -- usually "45%" copied
    from a Basel slide -- that most portfolio simulations assume.

**Assumed** (chosen, documented, and swept in sensitivity analysis)
    Cost of funds, operating cost per loan, and the cost of a manual review. These are policy
    parameters of the hypothetical lender, not properties of LendingClub, and no attempt is made
    to pass them off as measured.

The core identity
-----------------
For an exposure ``E`` priced at APR ``r``::

    E[profit] = E * [ (1 - PD) * y_good(r)  -  PD * l_bad(r) ]  -  funding_cost(E)  -  opex(E)

Note that ``l_bad`` is expressed as a loss rate **on the original exposure**, not on the balance
outstanding at default. That is deliberate: it folds LGD and the exposure-at-default profile into a
single quantity that is directly measurable from the data, whereas splitting them would require
assuming an amortisation path for defaulters that the public file does not record.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.utils.runtime import get_logger

LOG = get_logger("decision.economics")


@dataclass
class EconomicsConfig:
    """Assumed cost parameters of the hypothetical lender. Every value here is a choice."""

    #: Annual cost of the capital deployed. A marketplace lender funds from investors; a bank
    #: funds from deposits. 3% is a mid-cycle unsecured-funding placeholder.
    annual_cost_of_funds: float = 0.03
    #: Fixed origination and servicing cost per booked loan, in dollars.
    opex_fixed: float = 150.0
    #: Variable servicing cost as a fraction of exposure over the life of the loan.
    opex_variable_rate: float = 0.010
    #: Cost of routing an application to a human underwriter.
    manual_review_cost: float = 75.0
    #: Weighted-average life of a 36-month fully amortising loan, in years. Derived from the
    #: amortisation schedule (see :func:`weighted_average_life`), not guessed.
    weighted_average_life_years: float = 1.55
    #: Fallback values used only when the APR-conditional fit is unavailable.
    fallback_good_yield: float = 0.16
    fallback_bad_loss_rate: float = 0.63

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weighted_average_life(apr: float = 0.13, term_months: int = 36) -> float:
    """Weighted-average life of a fully amortising loan, in years.

    Principal is repaid gradually, so the average dollar is outstanding for far less than the full
    term -- about 1.55 years on a 36-month loan. Using the full 3 years would overstate funding
    cost by roughly a factor of two.
    """
    r = apr / 12.0
    balance = 1.0
    payment = r / (1.0 - (1.0 + r) ** (-term_months)) if r > 0 else 1.0 / term_months
    weighted = 0.0
    for month in range(1, term_months + 1):
        interest = balance * r
        principal = payment - interest
        weighted += month * principal
        balance -= principal
    return float(weighted / 12.0)


def realised_profit(df: pd.DataFrame) -> pd.Series:
    """Net cash the lender actually received, minus what it lent. An arithmetic fact.

    ``total_pymnt`` covers everything received while the loan was performing; ``recoveries`` is
    collected after charge-off and is reported separately, so it must be added rather than assumed
    to be included. ``collection_recovery_fee`` is what the collection agency kept.
    """
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce")
    received = pd.to_numeric(df["total_pymnt"], errors="coerce").fillna(0.0)
    recovered = pd.to_numeric(df.get("recoveries", 0.0), errors="coerce").fillna(0.0)
    fee = pd.to_numeric(df.get("collection_recovery_fee", 0.0), errors="coerce").fillna(0.0)
    return received + recovered - fee - funded


def realised_return(df: pd.DataFrame) -> pd.Series:
    """Realised profit as a fraction of the amount lent."""
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce")
    return realised_profit(df) / funded.where(funded > 0)


@dataclass
class CashflowModel:
    """APR-conditional good-loan yield and defaulted-loan loss rate, fitted on observed cashflows.

    Both are fitted as simple linear functions of APR. Linear rather than anything cleverer for a
    reason: the relationships genuinely are close to linear over LendingClub's 5-31% range, a
    two-parameter fit cannot overfit, and a coefficient is something that can be read aloud.

    ``l_bad`` decreasing in APR is not an error -- a higher-rate loan collects more interest before
    it defaults, so its net loss on original exposure is slightly smaller.
    """

    good_slope: float = 0.0
    good_intercept: float = 0.16
    bad_slope: float = 0.0
    bad_intercept: float = 0.63
    n_good: int = 0
    n_bad: int = 0
    observed_lgd_mean: float = float("nan")
    observed_lgd_median: float = float("nan")
    notes: dict[str, Any] = field(default_factory=dict)

    def good_yield(self, apr: np.ndarray | float) -> np.ndarray:
        apr = np.asarray(apr, dtype=float)
        return np.clip(self.good_intercept + self.good_slope * apr, -0.5, 1.5)

    def bad_loss_rate(self, apr: np.ndarray | float) -> np.ndarray:
        apr = np.asarray(apr, dtype=float)
        return np.clip(self.bad_intercept + self.bad_slope * apr, 0.0, 1.2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_cashflow_model(
    train: pd.DataFrame,
    config: EconomicsConfig | None = None,
    target_column: str = "default",
) -> CashflowModel:
    """Estimate ``y_good(apr)`` and ``l_bad(apr)`` from matured training-window loans.

    Fitted on ``train`` only, like everything else that learns. The resulting LGD is a measured
    property of this portfolio rather than a regulatory default.
    """
    config = config or EconomicsConfig()
    frame = train.loc[pd.to_numeric(train[target_column], errors="coerce").notna()].copy()
    frame["_return"] = realised_return(frame)
    frame["_apr"] = pd.to_numeric(frame["int_rate"], errors="coerce") / 100.0
    frame = frame.loc[frame["_return"].notna() & frame["_apr"].notna()]
    # Realised returns outside this band are data errors, not loans.
    frame = frame.loc[frame["_return"].between(-1.2, 1.2)]

    is_bad = pd.to_numeric(frame[target_column], errors="coerce") > 0.5
    good, bad = frame.loc[~is_bad], frame.loc[is_bad]

    model = CashflowModel(
        good_intercept=config.fallback_good_yield,
        bad_intercept=config.fallback_bad_loss_rate,
        n_good=int(len(good)),
        n_bad=int(len(bad)),
    )

    if len(good) >= 100:
        slope, intercept = np.polyfit(good["_apr"].to_numpy(), good["_return"].to_numpy(), 1)
        model.good_slope, model.good_intercept = float(slope), float(intercept)
    if len(bad) >= 100:
        # Loss rate is the negative of the (negative) realised return.
        slope, intercept = np.polyfit(bad["_apr"].to_numpy(), -bad["_return"].to_numpy(), 1)
        model.bad_slope, model.bad_intercept = float(slope), float(intercept)

        # Empirical LGD as conventionally defined: principal not recovered, over amount lent.
        funded = pd.to_numeric(bad["funded_amnt"], errors="coerce")
        principal_back = (
            pd.to_numeric(bad["total_rec_prncp"], errors="coerce").fillna(0.0)
            + pd.to_numeric(bad.get("recoveries", 0.0), errors="coerce").fillna(0.0)
            - pd.to_numeric(bad.get("collection_recovery_fee", 0.0), errors="coerce").fillna(0.0)
        )
        lgd = (1.0 - principal_back / funded.where(funded > 0)).clip(0.0, 1.0)
        model.observed_lgd_mean = float(lgd.mean())
        model.observed_lgd_median = float(lgd.median())

    LOG.info(
        "cashflow model | good n=%s yield=%.3f%+.3f*apr | bad n=%s loss=%.3f%+.3f*apr | "
        "empirical LGD mean=%.3f median=%.3f",
        f"{model.n_good:,}",
        model.good_intercept,
        model.good_slope,
        f"{model.n_bad:,}",
        model.bad_intercept,
        model.bad_slope,
        model.observed_lgd_mean,
        model.observed_lgd_median,
    )
    return model


def expected_loss(
    pd_estimate: np.ndarray | float,
    exposure: np.ndarray | float,
    lgd: np.ndarray | float,
) -> np.ndarray:
    """The textbook identity ``EL = PD x LGD x EAD``.

    Reported because it is the number a credit committee asks for, but note it is *not* the
    quantity a decision should maximise against -- it ignores revenue entirely, so minimising it
    is achieved perfectly by lending to nobody. That gap is the point of EXP03.
    """
    return np.asarray(pd_estimate, dtype=float) * np.asarray(lgd, dtype=float) * np.asarray(
        exposure, dtype=float
    )


def funding_cost(
    exposure: np.ndarray | float, config: EconomicsConfig, apr: np.ndarray | float | None = None
) -> np.ndarray:
    """Cost of the capital tied up, over the weighted-average life of the loan."""
    exposure = np.asarray(exposure, dtype=float)
    return exposure * config.annual_cost_of_funds * config.weighted_average_life_years


def operating_cost(exposure: np.ndarray | float, config: EconomicsConfig) -> np.ndarray:
    """Fixed plus variable servicing cost."""
    exposure = np.asarray(exposure, dtype=float)
    return config.opex_fixed + config.opex_variable_rate * exposure


def expected_profit(
    pd_estimate: np.ndarray | float,
    exposure: np.ndarray | float,
    apr: np.ndarray | float,
    cashflow_model: CashflowModel,
    config: EconomicsConfig | None = None,
) -> np.ndarray:
    """Expected profit in dollars for lending ``exposure`` at ``apr`` to a borrower with ``PD``.

    This is the quantity the decision layer maximises. It differs from ``-EL`` in the way that
    matters: it contains revenue, so it has an interior optimum. Minimising expected loss says
    "approve nobody"; maximising expected profit says "approve everybody whose price covers their
    risk", which is what lending actually is.
    """
    config = config or EconomicsConfig()
    pd_estimate = np.asarray(pd_estimate, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    apr = np.asarray(apr, dtype=float)

    revenue = (1.0 - pd_estimate) * cashflow_model.good_yield(apr) * exposure
    loss = pd_estimate * cashflow_model.bad_loss_rate(apr) * exposure
    return revenue - loss - funding_cost(exposure, config, apr) - operating_cost(exposure, config)


def breakeven_pd(
    apr: np.ndarray | float,
    exposure: np.ndarray | float,
    cashflow_model: CashflowModel,
    config: EconomicsConfig | None = None,
) -> np.ndarray:
    """The PD at which expected profit is exactly zero.

    Solving ``E[profit] = 0`` for PD gives the *economically correct* approval cut-off for a given
    price, which is what a fixed 15%-PD threshold is silently approximating -- badly, because the
    correct cut-off moves with the interest rate. A 26% APR loan can carry far more risk than a 7%
    one and still be profitable. This function is the analytical core of EXP03.
    """
    config = config or EconomicsConfig()
    exposure = np.asarray(exposure, dtype=float)
    apr = np.asarray(apr, dtype=float)

    y = cashflow_model.good_yield(apr)
    l = cashflow_model.bad_loss_rate(apr)
    fixed = funding_cost(exposure, config, apr) + operating_cost(exposure, config)

    numerator = y * exposure - fixed
    denominator = (y + l) * exposure
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator > 0, numerator / denominator, 0.0)
    return np.clip(result, 0.0, 1.0)
