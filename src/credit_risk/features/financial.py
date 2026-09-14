"""Derived credit features.

Every function here is a pure, row-wise transform of decision-time columns, which is what makes
them safe to compute before splitting. Each one is documented with the three questions the project
requires of any feature:

  1. What does it represent?
  2. Why might it predict default?
  3. Was it knowable at decision time?

The list is deliberately short. A hundred auto-generated ratios would raise validation AUC a little
and make the model impossible to defend; a dozen features a credit officer would recognise is worth
more, and each one below is a quantity that appears on a real underwriting sheet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Reference APR used *only* to estimate the monthly payment of a hypothetical loan when
#: re-scoring counterfactual exposures in the decision simulator. It is deliberately a fixed
#: constant rather than the model's own price, which would make the feature circular.
#: See docs/ASSUMPTIONS.md.
REFERENCE_APR = 0.13

#: Values LendingClub uses as sentinels rather than real measurements.
_DTI_SENTINEL = -1.0


def monthly_payment(principal: np.ndarray | pd.Series, apr: float, term_months: int) -> np.ndarray:
    """Standard amortising payment. Used for counterfactual exposures, never fitted."""
    principal = np.asarray(principal, dtype=float)
    r = apr / 12.0
    if r <= 0:
        return principal / term_months
    return principal * r / (1.0 - (1.0 + r) ** (-term_months))


def add_derived_features(
    df: pd.DataFrame,
    reference_apr: float = REFERENCE_APR,
    term_months: int = 36,
) -> pd.DataFrame:
    """Return ``df`` with derived credit features appended.

    Non-mutating: the caller keeps the original frame.
    """
    out = df.copy()
    inc = pd.to_numeric(out.get("annual_inc"), errors="coerce")
    amt = pd.to_numeric(out.get("loan_amnt"), errors="coerce")
    # An income of zero is not a real observation; treat it as missing rather than dividing by it.
    inc_safe = inc.where(inc > 0)

    # -- affordability ------------------------------------------------------
    # Loan size relative to annual income. The single most intuitive affordability measure:
    # a $30k loan against a $30k income is a different proposition from the same loan
    # against $200k. Knowable at decision time (both are on the application).
    out["loan_to_income"] = (amt / inc_safe).astype(float)

    # Log income. Income is strongly right-skewed, and a linear model in raw dollars is
    # dominated by a handful of high earners. The log is what a scorecard would band.
    out["log_annual_inc"] = np.log1p(inc.clip(lower=0))

    # Estimated payment burden of the *new* loan, as a share of income, priced at a fixed
    # reference APR rather than the lender's own price. This is what makes counterfactual
    # exposures scoreable in the simulator without circularity.
    est_payment = monthly_payment(amt.fillna(0.0), reference_apr, term_months)
    out["est_payment_to_income"] = np.where(
        inc_safe.notna() & (inc_safe > 0), 12.0 * est_payment / inc_safe, np.nan
    )

    # -- debt burden --------------------------------------------------------
    dti = pd.to_numeric(out.get("dti"), errors="coerce")
    dti = dti.where(dti != _DTI_SENTINEL)
    # LendingClub's DTI excludes the loan being applied for. What a lender actually underwrites
    # against is the burden *after* this loan is booked, which is the sum of the two.
    out["dti_clean"] = dti.clip(lower=0, upper=60)
    out["dti_post_loan"] = (out["dti_clean"] + 100.0 * out["est_payment_to_income"]).clip(upper=120)

    # Revolving balance relative to income. Captures existing unsecured exposure that a
    # DTI computed from minimum payments understates.
    revol = pd.to_numeric(out.get("revol_bal"), errors="coerce")
    out["revol_bal_to_income"] = (revol / inc_safe).astype(float)

    # -- credit history depth ----------------------------------------------
    # Months since the first credit line, measured *as at the application date* rather than
    # as at today. Computing it against today's date would leak the vintage into the feature.
    if "earliest_cr_line" in out.columns and "issue_period" in out.columns:
        out["credit_history_months"] = _months_between(
            out["earliest_cr_line"], out["issue_period"]
        )
    else:
        out["credit_history_months"] = np.nan

    # Share of accounts currently open. A borrower with 30 accounts of which 3 are open looks
    # different from one with 4 accounts of which 3 are open, at the same `open_acc`.
    open_acc = pd.to_numeric(out.get("open_acc"), errors="coerce")
    total_acc = pd.to_numeric(out.get("total_acc"), errors="coerce")
    out["open_acc_ratio"] = (open_acc / total_acc.where(total_acc > 0)).astype(float)

    # -- scorecard ----------------------------------------------------------
    lo = pd.to_numeric(out.get("fico_range_low"), errors="coerce")
    hi = pd.to_numeric(out.get("fico_range_high"), errors="coerce")
    # The bureau reports a band; the midpoint is the usual single-number summary.
    out["fico_mid"] = (lo + hi) / 2.0

    # -- utilisation --------------------------------------------------------
    # Revolving utilisation above 100% happens (over-limit accounts) and is real, but the
    # extreme tail is mostly data error. Clipped rather than dropped.
    ru = pd.to_numeric(out.get("revol_util"), errors="coerce")
    out["revol_util_clipped"] = ru.clip(lower=0, upper=150)

    # -- adverse history flags ---------------------------------------------
    # Binary flags alongside the counts. A logistic regression cannot express "any derogatory
    # record at all" from a count without the flag, and that step change is the real signal.
    out["has_delinq_2yrs"] = _gt_zero(out.get("delinq_2yrs"))
    out["has_pub_rec"] = _gt_zero(out.get("pub_rec"))
    out["has_bankruptcy"] = _gt_zero(out.get("pub_rec_bankruptcies"))
    out["has_collections"] = _gt_zero(out.get("collections_12_mths_ex_med"))
    out["has_derogatory"] = (
        out[["has_pub_rec", "has_bankruptcy", "has_collections"]].max(axis=1).astype("Int8")
    )

    # -- credit-seeking intensity ------------------------------------------
    # Enquiries in the last six months. A classic early-warning indicator: a borrower
    # applying to many lenders at once is often already under strain, and the enquiry is
    # visible before any missed payment is.
    inq = pd.to_numeric(out.get("inq_last_6mths"), errors="coerce")
    out["inq_last_6mths_capped"] = inq.clip(upper=10)

    return out


def derived_feature_names(include_payment_burden: bool = True) -> tuple[str, ...]:
    """Names added by :func:`add_derived_features`, in a stable order."""
    names = [
        "loan_to_income",
        "log_annual_inc",
        "dti_clean",
        "revol_bal_to_income",
        "credit_history_months",
        "open_acc_ratio",
        "fico_mid",
        "revol_util_clipped",
        "has_delinq_2yrs",
        "has_pub_rec",
        "has_bankruptcy",
        "has_collections",
        "has_derogatory",
        "inq_last_6mths_capped",
    ]
    if include_payment_burden:
        names += ["est_payment_to_income", "dti_post_loan"]
    return tuple(names)


#: Features whose value changes when the lender chooses a different exposure. The decision
#: simulator recomputes exactly these when scoring a counterfactual loan amount, and nothing else.
EXPOSURE_DEPENDENT_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "loan_to_income",
    "est_payment_to_income",
    "dti_post_loan",
)


def _gt_zero(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, dtype="Int8")
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric > 0).astype("Int8").where(numeric.notna())


def _months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole months from ``start`` to ``end`` for monthly Period series."""
    if start is None or end is None:
        return pd.Series(np.nan)
    try:
        delta = (end.astype("period[M]") - start.astype("period[M]")).apply(
            lambda x: x.n if pd.notna(x) else np.nan
        )
    except (TypeError, ValueError):
        return pd.Series(np.nan, index=end.index)
    return pd.to_numeric(delta, errors="coerce").clip(lower=0)
