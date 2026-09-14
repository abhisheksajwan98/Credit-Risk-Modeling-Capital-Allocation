"""Blind-spot signals and the portfolio comparison that uses them.

This module exists because the analysis scripts for EXP08-EXP11 had each grown their own copy of
the credit economics, and one of those copies was wrong in a way that inverted a headline result.
Everything economic now routes through :mod:`credit_risk.decision.expected_loss`.

The central research question these signals serve:

    *Can representation-learning signals identify situations where a conventional credit-risk
    model is unreliable, and can that information improve lending decisions?*

Two distinct claims are involved, and they need separate evidence:

1. **Detection.** Does the signal identify borrowers whose PD is untrustworthy? The honest test is
   *conditional*: within a narrow PD stratum, does a high signal still predict that the model
   under-states risk? A marginal quintile table cannot answer this, because both signals correlate
   with the PD level itself, and a boosted model is already less well calibrated at high PD.
2. **Decision value.** Does acting on the signal improve the book? This must be measured on the
   **whole portfolio at equal capital**, using **observed cashflows** -- not by comparing the two
   swap sets, and not with an assumed LGD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    expected_profit,
    realised_profit,
)
from credit_risk.utils.runtime import get_logger

LOG = get_logger("decision.blindspot")


# ---------------------------------------------------------------------------
# 1. Detection: is the signal informative beyond the PD level?
# ---------------------------------------------------------------------------

def marginal_signal_table(
    y_true: np.ndarray,
    pd_estimate: np.ndarray,
    signal: np.ndarray,
    n_bins: int = 5,
    labels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Model performance by signal quantile -- the *marginal* view.

    Useful for description, but **not** sufficient evidence that a signal finds blind spots: if the
    signal correlates with PD, this table will show calibration degrading simply because the model
    is less well calibrated at higher PD. Always read it beside
    :func:`conditional_signal_table`.
    """
    from credit_risk.evaluation.metrics import compute_metrics

    frame = pd.DataFrame({"y": y_true, "pd": pd_estimate, "signal": signal})
    frame["bin"] = pd.qcut(frame["signal"], q=n_bins, labels=labels, duplicates="drop")

    rows = []
    for name, group in frame.groupby("bin", observed=True):
        metrics = compute_metrics(group["y"].to_numpy(), group["pd"].to_numpy())
        rows.append(
            {
                "quantile": str(name),
                "n": len(group),
                "mean_signal": float(group["signal"].mean()),
                "default_rate": metrics.positive_rate,
                "mean_pd": metrics.mean_prediction,
                "calibration_gap": metrics.positive_rate - metrics.mean_prediction,
                "roc_auc": metrics.roc_auc,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "predicted_observed_ratio": metrics.predicted_observed_ratio,
                "observed_expected_ratio": metrics.observed_expected_ratio,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ConditionalResult:
    """Does the signal predict miscalibration *within* PD strata?"""

    table: pd.DataFrame
    n_strata: int
    n_strata_positive: int
    mean_gap_difference: float
    sign_test_p: float
    odds_ratio_per_sd: float
    spearman_with_pd: float

    def verdict(self, name: str) -> str:
        direction = "under-states" if self.mean_gap_difference > 0 else "over-states"
        strength = (
            "survives" if self.sign_test_p < 0.05 and self.mean_gap_difference > 0
            else "does not survive"
        )
        return (
            f"{name}: correlation with PD itself is {self.spearman_with_pd:+.3f}. "
            f"Within PD strata the high-signal half shows a larger gap in "
            f"{self.n_strata_positive}/{self.n_strata} strata "
            f"(mean {self.mean_gap_difference:+.4f}, sign-test p={self.sign_test_p:.4f}), so the "
            f"effect **{strength}** the PD control -- the model {direction} risk for high-signal "
            f"borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal "
            f"multiplies default odds by {self.odds_ratio_per_sd:.3f}."
        )


def conditional_signal_table(
    y_true: np.ndarray,
    pd_estimate: np.ndarray,
    signal: np.ndarray,
    n_strata: int = 10,
    min_per_half: int = 200,
) -> ConditionalResult:
    """Test the signal *holding the PD level fixed*.

    Within each PD decile the population is split at the median of the signal, and the calibration
    gap (observed minus predicted default rate) is compared between halves. If the signal only
    restates the PD level, the two halves inside a stratum will show the same gap and the effect
    vanishes here even when the marginal table looks dramatic.

    Two summaries are returned: an exact binomial sign test across strata, and the coefficient on
    the standardised signal in a logistic regression of the outcome on ``logit(PD)`` plus the
    signal -- the incremental information the signal carries about the outcome itself.
    """
    from scipy.stats import binomtest
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(pd_estimate, dtype=float), 1e-6, 1 - 1e-6)
    s = np.asarray(signal, dtype=float)

    frame = pd.DataFrame({"y": y, "p": p, "s": s})
    frame["stratum"] = pd.qcut(frame["p"], n_strata, labels=False, duplicates="drop")

    rows = []
    for stratum, group in frame.groupby("stratum"):
        median = group["s"].median()
        high, low = group[group["s"] > median], group[group["s"] <= median]
        if len(high) < min_per_half or len(low) < min_per_half:
            continue
        rows.append(
            {
                "pd_stratum": int(stratum),
                "mean_pd": float(group["p"].mean()),
                "n_high": len(high),
                "n_low": len(low),
                "predicted_high": float(high["p"].mean()),
                "observed_high": float(high["y"].mean()),
                "gap_high": float(high["y"].mean() - high["p"].mean()),
                "predicted_low": float(low["p"].mean()),
                "observed_low": float(low["y"].mean()),
                "gap_low": float(low["y"].mean() - low["p"].mean()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no PD stratum had enough rows on both sides of the signal median")
    table["gap_difference"] = table["gap_high"] - table["gap_low"]

    positive = int((table["gap_difference"] > 0).sum())
    total = int(len(table))
    sign_p = float(binomtest(positive, total, 0.5, alternative="greater").pvalue)

    logit_p = np.log(p / (1 - p))
    z = (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([logit_p, z]), y)

    return ConditionalResult(
        table=table,
        n_strata=total,
        n_strata_positive=positive,
        mean_gap_difference=float(table["gap_difference"].mean()),
        sign_test_p=sign_p,
        odds_ratio_per_sd=float(np.exp(model.coef_[0][1])),
        spearman_with_pd=float(pd.Series(s).corr(pd.Series(p), method="spearman")),
    )


# ---------------------------------------------------------------------------
# 2. Decision value: does acting on the signal improve the book?
# ---------------------------------------------------------------------------

@dataclass
class PortfolioResult:
    """One policy's book, measured on realised cashflows."""

    policy: str
    n_loans: int
    capital_deployed: float
    observed_profit: float
    return_on_capital: float
    default_rate: float
    mean_apr: float
    budget_utilisation: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class SwapAnalysis:
    """Comparison of two capital-constrained books built from the same applicant pool."""

    portfolios: list[PortfolioResult]
    n_swapped_out: int
    n_swapped_in: int
    swapped_out_default_rate: float
    swapped_in_default_rate: float
    swapped_out_return: float
    swapped_in_return: float
    budget: float
    notes: list[str] = field(default_factory=list)

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([p.to_dict() for p in self.portfolios])

    def verdict(self, treatment: str, baseline: str) -> str:
        a = next(p for p in self.portfolios if p.policy == baseline)
        b = next(p for p in self.portfolios if p.policy == treatment)
        d_profit = b.observed_profit - a.observed_profit
        d_roc = b.return_on_capital - a.return_on_capital
        d_capital = b.capital_deployed - a.capital_deployed
        material = abs(d_roc) >= 0.005  # 50bp of return on capital

        headline = (
            f"{treatment} changed realised profit by **${d_profit:+,.0f}** on a "
            f"${a.capital_deployed:,.0f} book, deploying ${d_capital:+,.0f} of capital "
            f"(a {d_roc:+.2%} change in return on capital)."
        )
        if material:
            reading = (
                "That is a material change in return on capital and the direction is the finding."
            )
        else:
            reading = (
                "That is **below 50bp of return on capital and should be read as no material "
                "difference in profitability**. The book default rate moved from "
                f"{a.default_rate:.2%} to {b.default_rate:.2%} at essentially unchanged return, "
                "so the defensible claim is risk reduction at constant profitability -- not a "
                "profit improvement."
            )
        return headline + " " + reading


def build_portfolio(
    df: pd.DataFrame,
    approve: pd.Series | np.ndarray,
    policy: str,
    budget: float,
    exposure_column: str = "funded_amnt",
) -> PortfolioResult:
    """Summarise an approved book using **observed** cashflows.

    Realised profit is read from the data, never modelled: it is
    ``total_pymnt + recoveries - collection_recovery_fee - funded_amnt``. Using an assumed LGD
    here is what inverted the original EXP11 conclusion.
    """
    mask = np.asarray(approve, dtype=bool)
    book = df.loc[mask]
    exposure = pd.to_numeric(book[exposure_column], errors="coerce")
    capital = float(exposure.sum())
    profit = float(realised_profit(book).sum())

    return PortfolioResult(
        policy=policy,
        n_loans=int(mask.sum()),
        capital_deployed=capital,
        observed_profit=profit,
        return_on_capital=profit / capital if capital > 0 else float("nan"),
        default_rate=float(pd.to_numeric(book["default"], errors="coerce").mean()),
        mean_apr=float(pd.to_numeric(book["int_rate"], errors="coerce").mean() / 100.0),
        budget_utilisation=capital / budget if budget > 0 else float("nan"),
    )


def greedy_allocate(
    density: pd.Series,
    exposure: pd.Series,
    budget: float,
    min_density: float = 0.0,
) -> pd.Series:
    """Fill a capital budget in descending profit-per-dollar order.

    The continuous-relaxation solution to the knapsack: with divisible capital, ranking by profit
    density and filling until the budget binds is optimal. Loans below ``min_density`` are never
    funded regardless of remaining budget, since they lose money in expectation.
    """
    order = density.sort_values(ascending=False, kind="mergesort").index
    cumulative = exposure.loc[order].cumsum()
    approve = pd.Series(False, index=density.index)
    approve.loc[order] = (cumulative.to_numpy() <= budget) & (
        density.loc[order].to_numpy() > min_density
    )
    return approve


def compare_capital_constrained(
    df: pd.DataFrame,
    pd_estimate: np.ndarray,
    flagged: np.ndarray,
    cashflow_model: CashflowModel,
    economics: EconomicsConfig | None = None,
    budget_fraction: float = 0.80,
    exposure_column: str = "funded_amnt",
) -> SwapAnalysis:
    """Compare a PD-ranked book against a blind-spot-aware book at **identical budget**.

    Both policies rank by expected profit per dollar and fill the same capital budget. The only
    difference is that the aware policy refuses to fund flagged borrowers.

    Three corrections relative to the original EXP11 implementation, each of which mattered:

    * profit is **observed**, not computed from an assumed LGD;
    * the comparison is between the two **portfolios**, not between the two swap sets, since the
      swap sets need not deploy the same capital;
    * **return on capital** is reported alongside profit, because a book that deploys more capital
      will usually earn more profit without being better.
    """
    economics = economics or EconomicsConfig()
    frame = df.reset_index(drop=True).copy()
    pd_hat = np.asarray(pd_estimate, dtype=float)
    flag = np.asarray(flagged, dtype=bool)

    exposure = pd.to_numeric(frame[exposure_column], errors="coerce").fillna(0.0)
    apr = pd.to_numeric(frame["int_rate"], errors="coerce").fillna(0.13) / 100.0

    profit = expected_profit(pd_hat, exposure.to_numpy(), apr.to_numpy(), cashflow_model, economics)
    density = pd.Series(
        np.divide(profit, exposure.to_numpy(), out=np.zeros_like(profit),
                  where=exposure.to_numpy() > 0),
        index=frame.index,
    )

    fundable = exposure[density > 0].sum()
    budget = float(fundable * budget_fraction)

    aware_density = density.copy()
    aware_density[flag] = -np.inf

    approve_a = greedy_allocate(density, exposure, budget)
    approve_b = greedy_allocate(aware_density, exposure, budget)

    swapped_out = approve_a & ~approve_b
    swapped_in = ~approve_a & approve_b

    def _return(mask: pd.Series) -> float:
        capital = exposure[mask].sum()
        return float(realised_profit(frame.loc[mask]).sum() / capital) if capital > 0 else float("nan")

    analysis = SwapAnalysis(
        portfolios=[
            build_portfolio(frame, approve_a, "A_pd_ranked", budget, exposure_column),
            build_portfolio(frame, approve_b, "B_blindspot_aware", budget, exposure_column),
        ],
        n_swapped_out=int(swapped_out.sum()),
        n_swapped_in=int(swapped_in.sum()),
        swapped_out_default_rate=float(
            pd.to_numeric(frame.loc[swapped_out, "default"], errors="coerce").mean()
        ),
        swapped_in_default_rate=float(
            pd.to_numeric(frame.loc[swapped_in, "default"], errors="coerce").mean()
        ),
        swapped_out_return=_return(swapped_out),
        swapped_in_return=_return(swapped_in),
        budget=budget,
    )

    for portfolio in analysis.portfolios:
        LOG.info(
            "%-20s n=%s capital=$%.0f profit=$%.0f RoC=%.4f default=%.4f",
            portfolio.policy,
            f"{portfolio.n_loans:,}",
            portfolio.capital_deployed,
            portfolio.observed_profit,
            portfolio.return_on_capital,
            portfolio.default_rate,
        )
    return analysis


def exposure_capped_policy(
    density: pd.Series,
    exposure: pd.Series,
    flagged: np.ndarray,
    budget: float,
    cap_multiplier: float = 0.5,
) -> tuple[pd.Series, pd.Series]:
    """Fund flagged borrowers at reduced exposure instead of refusing them.

    The alternative the original EXP11 conclusion pointed at without testing: if a flagged borrower
    is *mispriced* rather than simply bad, the response is to keep the yield and cut the exposure.
    Returns ``(approve, exposure_used)``.
    """
    adjusted = exposure.copy()
    adjusted[flagged] = adjusted[flagged] * cap_multiplier
    approve = greedy_allocate(density, adjusted, budget)
    return approve, adjusted.where(approve, 0.0)
