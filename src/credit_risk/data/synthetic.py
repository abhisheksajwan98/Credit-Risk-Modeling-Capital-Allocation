"""Schema-faithful synthetic LendingClub extract.

Why this exists
---------------
The real data is a 1.4 GB Kaggle download behind an account. Without a fallback, nothing in this
repository could be run, tested or reviewed until that download completes. This module emits a
file with the *same column names, dtypes, string formats, categorical levels, vintage-dependent
missingness and cashflow accounting* as the real extract, so that `prepare_data.py` and everything
downstream take exactly the same code path.

What it is not
--------------
It is **not** a source of findings. The risk structure here was written by hand; any AUC, lift or
policy gain measured on it is a statement about this generator, not about consumer credit. Every
artefact built from it carries ``data_source: synthetic`` in its manifest, and
``scripts/evaluate.py`` refuses to promote synthetic results into ``results/reports/``.

The cohort effects (a per-ZIP3 and per-employer random effect on default odds) are deliberately
*modest and configurable*. They exist so the graph code has structure to find and can be tested
end to end. Turning them up would manufacture a positive result for Layer B, which is precisely
the failure mode this project is trying to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.data import schema
from credit_risk.utils.runtime import get_logger

LOG = get_logger("data.synthetic")

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_PURPOSES = (
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "small_business", "car", "medical", "moving", "vacation", "house", "wedding",
    "renewable_energy", "educational",
)
_PURPOSE_P = np.array(
    [0.58, 0.23, 0.06, 0.045, 0.02, 0.011, 0.011, 0.011, 0.007, 0.006, 0.005, 0.004, 0.002, 0.002]
)

_PURPOSE_P = _PURPOSE_P / _PURPOSE_P.sum()
_STATES = (
    "CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA", "NC", "MI", "MD", "MA", "AZ",
    "WA", "CO", "MN", "IN", "MO", "TN", "CT", "WI", "AL", "SC", "OR", "LA", "KY", "OK", "KS",
    "NV", "UT", "AR", "MS", "NM", "NE", "WV", "NH", "HI", "RI", "MT", "DE", "AK", "WY", "DC",
    "SD", "VT", "ND", "ME", "ID", "IA",
)

_EMP_LENGTHS = (
    "10+ years", "< 1 year", "2 years", "3 years", "1 year", "5 years", "4 years",
    "6 years", "7 years", "8 years", "9 years", "n/a",
)
_EMP_LENGTH_P = np.array(
    [0.33, 0.09, 0.09, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.02]
)

_EMP_LENGTH_P = _EMP_LENGTH_P / _EMP_LENGTH_P.sum()
_EMPLOYERS = (
    "Teacher", "Manager", "Registered Nurse", "Owner", "Supervisor", "Sales", "RN", "Driver",
    "Project Manager", "Engineer", "Office Manager", "General Manager", "Director", "Technician",
    "Accountant", "Analyst", "Attorney", "Police Officer", "Server", "Mechanic", "Consultant",
    "Administrative Assistant", "Operations Manager", "Truck Driver", "Electrician", "Clerk",
    "US Army", "Walmart", "Bank of America", "UPS", "AT&T", "Wells Fargo", "IBM", "Kaiser",
    "US Postal Service", "Verizon", "Home Depot", "Target", "State of California", "FedEx",
)

_HOME_OWNERSHIP = ("MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY")
_HOME_P = np.array([0.49, 0.40, 0.10, 0.005, 0.003, 0.002])

_HOME_P = _HOME_P / _HOME_P.sum()
_VERIFICATION = ("Source Verified", "Verified", "Not Verified")
_VERIFICATION_P = np.array([0.37, 0.32, 0.31])

_VERIFICATION_P = _VERIFICATION_P / _VERIFICATION_P.sum()
_GRADES = ("A", "B", "C", "D", "E", "F", "G")
#: Base annual interest rate per grade; sub-grade adds a within-grade step.
_GRADE_BASE_RATE = {"A": 7.0, "B": 10.5, "C": 13.8, "D": 17.2, "E": 20.3, "F": 24.1, "G": 26.8}


@dataclass(frozen=True)
class SyntheticSpec:
    """Knobs for the generator. Defaults produce a ~15% default rate on 36-month loans."""

    n_loans: int = 60_000
    start_period: str = "2007-06"
    end_period: str = "2018-12"
    seed: int = 20260826
    n_zip3: int = 900
    #: Std-dev of the per-ZIP3 random effect on the default log-odds.
    zip_effect_sd: float = 0.18
    #: Std-dev of the per-employer random effect on the default log-odds.
    emp_effect_sd: float = 0.12
    #: Extra log-odds added per year of vintage from 2013 onward (mimics the observed
    #: deterioration of later LendingClub vintages). Set to 0 to disable drift.
    vintage_drift_per_year: float = 0.10
    base_default_logit: float = -2.05


def _period_index(spec: SyntheticSpec) -> pd.PeriodIndex:
    return pd.period_range(spec.start_period, spec.end_period, freq="M")


def _vintage_weights(periods: pd.PeriodIndex) -> np.ndarray:
    """Approximate LendingClub's origination growth: near-exponential 2007-2015, then flat."""
    years = np.array([p.year + (p.month - 1) / 12.0 for p in periods])
    growth = np.exp(0.55 * np.clip(years - 2007.5, 0, 8.0))
    growth *= np.where(years >= 2016.0, 0.92, 1.0)
    return growth / growth.sum()


def _fmt_period(periods: pd.PeriodIndex) -> np.ndarray:
    """LendingClub writes dates as ``Dec-2015``."""
    return np.array([f"{_MONTHS[p.month - 1]}-{p.year}" for p in periods], dtype=object)


def generate(spec: SyntheticSpec | None = None) -> pd.DataFrame:
    """Generate a synthetic accepted-loans frame with the real column set."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    n = spec.n_loans

    periods = _period_index(spec)
    issue_idx = rng.choice(len(periods), size=n, p=_vintage_weights(periods))
    issue_period = periods[issue_idx]
    issue_year_frac = np.array([p.year + (p.month - 1) / 12.0 for p in issue_period])

    # --- borrower latent risk ------------------------------------------------
    # One latent variable drives FICO, DTI, utilisation, the assigned grade and the default
    # draw, which is what produces realistic correlations between the observable features.
    latent = rng.normal(0.0, 1.0, n)

    zip3_id = rng.integers(10, 10 + spec.n_zip3, size=n)
    zip_effect_by_id = rng.normal(0.0, spec.zip_effect_sd, size=10 + spec.n_zip3)
    zip_effect = zip_effect_by_id[zip3_id]

    emp_idx = rng.integers(0, len(_EMPLOYERS), size=n)
    emp_effect_by_id = rng.normal(0.0, spec.emp_effect_sd, size=len(_EMPLOYERS))
    emp_effect = emp_effect_by_id[emp_idx]
    emp_title = np.array(_EMPLOYERS, dtype=object)[emp_idx]
    # ~6% of applicants disclose no employment string, as in the real file.
    emp_title = np.where(rng.random(n) < 0.06, None, emp_title)

    fico_low = np.clip(np.round((690 - 32 * latent + rng.normal(0, 18, n)) / 5) * 5, 660, 845)
    annual_inc = np.round(
        np.exp(np.clip(11.05 - 0.16 * latent + rng.normal(0, 0.52, n), 8.5, 14.2)), 2
    )
    dti = np.clip(17.5 + 3.6 * latent + rng.normal(0, 6.5, n), 0.0, 45.0).round(2)
    revol_util = np.clip(52.0 + 9.5 * latent + rng.normal(0, 22, n), 0.0, 145.0).round(1)

    term_months = np.where(rng.random(n) < 0.72, 36, 60)
    loan_amnt = np.clip(
        np.round((12_500 + 2_200 * latent + rng.normal(0, 7_800, n)) / 25) * 25, 500, 40_000
    )

    # --- grade / pricing -----------------------------------------------------
    grade_score = latent + rng.normal(0, 0.45, n)
    grade_idx = np.clip(np.digitize(grade_score, [-1.05, -0.35, 0.25, 0.85, 1.45, 2.05]), 0, 6)
    grade = np.array(_GRADES, dtype=object)[grade_idx]
    sub_step = rng.integers(1, 6, size=n)
    sub_grade = np.array([f"{g}{s}" for g, s in zip(grade, sub_step, strict=True)], dtype=object)
    int_rate = np.round(
        np.array([_GRADE_BASE_RATE[g] for g in grade])
        + 0.62 * (sub_step - 3)
        + np.where(term_months == 60, 1.35, 0.0)
        + rng.normal(0, 0.35, n),
        2,
    ).clip(5.31, 30.99)

    monthly_rate = int_rate / 100.0 / 12.0
    installment = np.round(
        loan_amnt * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-term_months)), 2
    )

    # --- default outcome -----------------------------------------------------
    drift = spec.vintage_drift_per_year * np.clip(issue_year_frac - 2013.0, 0.0, None)
    logit = (
        spec.base_default_logit
        + 0.80 * latent
        + 0.30 * (term_months == 60)
        + 0.012 * (dti - 17.5)
        + zip_effect
        + emp_effect
        + drift
    )
    pd_true = 1.0 / (1.0 + np.exp(-logit))
    defaulted = rng.random(n) < pd_true

    # --- maturity and status -------------------------------------------------
    data_cutoff = pd.Period("2018-12", freq="M")
    months_observed = np.array([(data_cutoff - p).n for p in issue_period])
    matured = months_observed >= term_months

    # Time of default, in months from origination: early-skewed, as in real portfolios.
    default_month = np.clip(
        np.round(rng.gamma(shape=2.6, scale=5.0, size=n) + 3).astype(int), 3, term_months
    )
    resolved_by_cutoff = np.where(defaulted, default_month + 5 <= months_observed, matured)

    loan_status = np.empty(n, dtype=object)
    loan_status[:] = "Current"
    loan_status[defaulted & resolved_by_cutoff] = "Charged Off"
    loan_status[~defaulted & matured] = "Fully Paid"
    # Non-matured, non-defaulted loans stay Current; a slice is late, as in the real file.
    still_running = ~matured & ~(defaulted & resolved_by_cutoff)
    late_draw = rng.random(n)
    loan_status[still_running & (late_draw < 0.012)] = "Late (31-120 days)"
    loan_status[still_running & (late_draw >= 0.012) & (late_draw < 0.020)] = "In Grace Period"
    # A handful of prepayments resolve early regardless of maturity.
    prepaid = ~defaulted & ~matured & (months_observed > 12) & (rng.random(n) < 0.28)
    loan_status[prepaid] = "Fully Paid"

    # --- cashflows, consistent with the status above -------------------------
    paid_months = np.where(
        loan_status == "Fully Paid",
        np.where(matured, term_months, np.minimum(months_observed, term_months)),
        np.where(
            loan_status == "Charged Off",
            default_month,
            np.minimum(months_observed, term_months),
        ),
    ).astype(float)

    total_pymnt = np.round(installment * paid_months, 2)

    # Split payments into principal and interest using the *exact* amortisation identity rather
    # than an approximate share. After k payments on a loan of P at monthly rate r over T months,
    # the principal repaid is
    #
    #     P * ((1+r)^k - 1) / ((1+r)^T - 1)
    #
    # which is bounded above by P by construction. An approximate split can imply that a
    # charged-off loan repaid more principal than was lent, which is impossible and would corrupt
    # the empirical LGD that the decision layer estimates from these cashflows.
    monthly = int_rate / 100.0 / 12.0
    growth_k = np.power(1.0 + monthly, paid_months)
    growth_t = np.power(1.0 + monthly, term_months)
    principal_fraction = np.where(
        growth_t > 1.0, (growth_k - 1.0) / (growth_t - 1.0), paid_months / term_months
    )
    principal_fraction = np.clip(principal_fraction, 0.0, 1.0)

    total_rec_prncp = np.round(loan_amnt * principal_fraction, 2)
    total_rec_int = np.round(np.maximum(total_pymnt - total_rec_prncp, 0.0), 2)

    # Fully paid loans repay principal exactly, by definition.
    fp = loan_status == "Fully Paid"
    total_rec_prncp = np.where(fp, loan_amnt.astype(float), total_rec_prncp)
    total_pymnt = np.round(total_rec_prncp + total_rec_int, 2)

    out_prncp = np.where(
        np.isin(loan_status, ["Fully Paid", "Charged Off"]),
        0.0,
        np.round(np.maximum(loan_amnt - total_rec_prncp, 0.0), 2),
    )

    co = loan_status == "Charged Off"
    recoveries = np.where(
        co,
        np.round(np.maximum(loan_amnt - total_rec_prncp, 0) * rng.beta(1.2, 7.0, n), 2),
        0.0,
    )
    collection_recovery_fee = np.round(recoveries * 0.16, 2)
    total_rec_late_fee = np.where(co, np.round(rng.gamma(1.1, 12.0, n), 2), 0.0)

    last_pymnt_period = np.array(
        [p + int(m) for p, m in zip(issue_period, np.maximum(paid_months, 1), strict=True)]
    )
    last_pymnt_period = np.minimum(last_pymnt_period, data_cutoff)

    earliest_cr_period = np.array(
        [
            p - int(m)
            for p, m in zip(
                issue_period,
                np.clip(rng.gamma(4.0, 42.0, n) + 24, 24, 620).astype(int),
                strict=True,
            )
        ]
    )

    # --- assemble ------------------------------------------------------------
    df = pd.DataFrame(
        {
            "id": np.arange(1_000_000, 1_000_000 + n),
            "member_id": np.full(n, np.nan),
            "loan_amnt": loan_amnt,
            "funded_amnt": loan_amnt,
            "funded_amnt_inv": np.round(loan_amnt * rng.uniform(0.95, 1.0, n) / 25) * 25,
            "term": np.where(term_months == 36, " 36 months", " 60 months"),
            "int_rate": int_rate,
            "installment": installment,
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_title": emp_title,
            "emp_length": rng.choice(_EMP_LENGTHS, size=n, p=_EMP_LENGTH_P),
            "home_ownership": rng.choice(_HOME_OWNERSHIP, size=n, p=_HOME_P),
            "annual_inc": annual_inc,
            "verification_status": rng.choice(_VERIFICATION, size=n, p=_VERIFICATION_P),
            "issue_d": _fmt_period(issue_period),
            "loan_status": loan_status,
            "pymnt_plan": np.where(rng.random(n) < 0.002, "y", "n"),
            "url": None,
            "desc": None,
            "purpose": rng.choice(_PURPOSES, size=n, p=_PURPOSE_P),
            "title": None,
            "zip_code": np.array([f"{z:03d}xx" for z in zip3_id], dtype=object),
            "addr_state": rng.choice(_STATES, size=n),
            "dti": dti,
            "delinq_2yrs": rng.poisson(0.28 + 0.10 * np.clip(latent, 0, None), n),
            "earliest_cr_line": _fmt_period(pd.PeriodIndex(earliest_cr_period, freq="M")),
            "fico_range_low": fico_low,
            "fico_range_high": fico_low + 4,
            "inq_last_6mths": rng.poisson(0.72 + 0.22 * np.clip(latent, 0, None), n),
            "mths_since_last_delinq": np.where(
                rng.random(n) < 0.51, np.nan, rng.integers(1, 120, n)
            ),
            "mths_since_last_record": np.where(
                rng.random(n) < 0.85, np.nan, rng.integers(1, 129, n)
            ),
            "open_acc": np.clip(rng.poisson(11.5, n), 1, 90),
            "pub_rec": rng.poisson(0.19, n),
            "revol_bal": np.round(np.exp(np.clip(9.4 + 0.22 * latent + rng.normal(0, 1.0, n), 0, 13))),
            "revol_util": revol_util,
            "total_acc": np.clip(rng.poisson(25.0, n), 2, 170),
            "initial_list_status": np.where(issue_year_frac < 2012.9, "f", np.where(rng.random(n) < 0.6, "w", "f")),
            "out_prncp": out_prncp,
            "out_prncp_inv": out_prncp,
            "total_pymnt": total_pymnt,
            "total_pymnt_inv": np.round(total_pymnt * 0.98, 2),
            "total_rec_prncp": total_rec_prncp,
            "total_rec_int": total_rec_int,
            "total_rec_late_fee": total_rec_late_fee,
            "recoveries": recoveries,
            "collection_recovery_fee": collection_recovery_fee,
            "last_pymnt_d": _fmt_period(pd.PeriodIndex(last_pymnt_period, freq="M")),
            "last_pymnt_amnt": np.round(installment * rng.uniform(0.5, 4.0, n), 2),
            "next_pymnt_d": None,
            "last_credit_pull_d": _fmt_period(pd.PeriodIndex(last_pymnt_period, freq="M")),
            # Post-origination FICO: collapses for defaulters. Present precisely so that the
            # leakage tests have a realistic offender to catch.
            "last_fico_range_high": np.where(
                defaulted, np.clip(fico_low - rng.integers(60, 190, n), 300, 850), fico_low + 24
            ),
            "last_fico_range_low": np.where(
                defaulted, np.clip(fico_low - rng.integers(65, 195, n), 300, 850), fico_low + 20
            ),
            "collections_12_mths_ex_med": rng.poisson(0.02, n),
            "mths_since_last_major_derog": np.where(
                rng.random(n) < 0.74, np.nan, rng.integers(1, 180, n)
            ),
            "policy_code": 1,
            "application_type": np.where(issue_year_frac < 2015.8, "Individual",
                                         np.where(rng.random(n) < 0.05, "Joint App", "Individual")),
            "acc_now_delinq": rng.poisson(0.005, n),
            "tot_coll_amt": np.where(rng.random(n) < 0.86, 0.0, np.round(rng.gamma(1.3, 260, n), 0)),
            "tot_cur_bal": np.round(np.exp(np.clip(11.2 + rng.normal(0, 1.3, n), 0, 15))),
            "total_rev_hi_lim": np.round(np.exp(np.clip(10.2 + rng.normal(0, 0.85, n), 0, 14))),
            "acc_open_past_24mths": rng.poisson(4.5, n),
            "avg_cur_bal": np.round(np.exp(np.clip(9.3 + rng.normal(0, 1.1, n), 0, 14))),
            "bc_open_to_buy": np.round(np.exp(np.clip(8.6 + rng.normal(0, 1.5, n), 0, 13))),
            "bc_util": np.clip(58.0 + 8.0 * latent + rng.normal(0, 26, n), 0, 200).round(1),
            "chargeoff_within_12_mths": rng.poisson(0.01, n),
            "delinq_amnt": np.where(rng.random(n) < 0.995, 0.0, np.round(rng.gamma(1.2, 900, n))),
            "mo_sin_old_il_acct": np.clip(rng.gamma(4.0, 34.0, n), 2, 700).round(0),
            "mo_sin_old_rev_tl_op": np.clip(rng.gamma(4.2, 44.0, n), 3, 900).round(0),
            "mo_sin_rcnt_rev_tl_op": np.clip(rng.gamma(1.5, 9.0, n), 0, 400).round(0),
            "mo_sin_rcnt_tl": np.clip(rng.gamma(1.4, 6.0, n), 0, 300).round(0),
            "mort_acc": rng.poisson(1.8, n),
            "mths_since_recent_bc": np.where(rng.random(n) < 0.04, np.nan, rng.integers(0, 300, n)),
            "mths_since_recent_bc_dlq": np.where(
                rng.random(n) < 0.77, np.nan, rng.integers(0, 180, n)
            ),
            "mths_since_recent_inq": np.where(
                rng.random(n) < 0.11, np.nan, rng.integers(0, 26, n)
            ),
            "mths_since_recent_revol_delinq": np.where(
                rng.random(n) < 0.67, np.nan, rng.integers(0, 180, n)
            ),
            "num_accts_ever_120_pd": rng.poisson(0.48, n),
            "num_actv_bc_tl": rng.poisson(3.7, n),
            "num_actv_rev_tl": rng.poisson(5.7, n),
            "num_bc_sats": rng.poisson(4.7, n),
            "num_bc_tl": rng.poisson(8.0, n),
            "num_il_tl": rng.poisson(8.4, n),
            "num_op_rev_tl": rng.poisson(8.2, n),
            "num_rev_accts": rng.poisson(14.4, n),
            "num_rev_tl_bal_gt_0": rng.poisson(5.6, n),
            "num_sats": rng.poisson(11.6, n),
            "num_tl_120dpd_2m": rng.poisson(0.001, n),
            "num_tl_30dpd": rng.poisson(0.004, n),
            "num_tl_90g_dpd_24m": rng.poisson(0.09, n),
            "num_tl_op_past_12m": rng.poisson(2.1, n),
            "pct_tl_nvr_dlq": np.clip(94.0 - 3.0 * np.clip(latent, 0, None) + rng.normal(0, 9, n), 0, 100).round(1),
            "percent_bc_gt_75": np.clip(43.0 + 9.0 * latent + rng.normal(0, 30, n), 0, 100).round(1),
            "pub_rec_bankruptcies": rng.poisson(0.13, n),
            "tax_liens": rng.poisson(0.05, n),
            "tot_hi_cred_lim": np.round(np.exp(np.clip(11.9 + rng.normal(0, 1.0, n), 0, 15))),
            "total_bal_ex_mort": np.round(np.exp(np.clip(10.6 + rng.normal(0, 0.95, n), 0, 14))),
            "total_bc_limit": np.round(np.exp(np.clip(9.8 + rng.normal(0, 1.0, n), 0, 14))),
            "total_il_high_credit_limit": np.round(np.exp(np.clip(10.4 + rng.normal(0, 1.1, n), 0, 14))),
            "disbursement_method": np.where(issue_year_frac < 2016.0, "Cash",
                                            np.where(rng.random(n) < 0.15, "DirectPay", "Cash")),
            "debt_settlement_flag": np.where(co & (rng.random(n) < 0.09), "Y", "N"),
        }
    )

    # Columns that exist in the real file but are structurally absent in this window.
    for col in schema.LATE_ADDITION_NUMERIC:
        df[col] = np.where(issue_year_frac >= 2015.95, rng.gamma(2.0, 6.0, n).round(1), np.nan)
    for col in schema.JOINT_APPLICATION_COLUMNS:
        if col == "verification_status_joint":
            df[col] = None
        else:
            df[col] = np.nan
    for col in (
        "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
        "hardship_loan_status", "settlement_status",
    ):
        df[col] = np.where(rng.random(n) < 0.004, "Y", "N") if col == "hardship_flag" else None
    for col in (
        "deferral_term", "hardship_amount", "hardship_length", "hardship_dpd",
        "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount", "settlement_amount", "settlement_percentage",
        "settlement_term",
    ):
        df[col] = np.nan
    for col in (
        "hardship_start_date", "hardship_end_date", "payment_plan_start_date",
        "debt_settlement_flag_date", "settlement_date",
    ):
        df[col] = None

    # Vintage-dependent availability: LendingClub introduced this bureau block partway through
    # its history. Reproduced so the availability-drift guard in features/build.py has a real
    # offender to catch rather than a hypothetical one.
    pre_block = issue_year_frac < 2012.6
    for col in schema.VINTAGE_SENSITIVE_NUMERIC:
        if col in df.columns:
            df.loc[pre_block, col] = np.nan
    for col in ("tot_cur_bal", "total_rev_hi_lim", "tot_coll_amt"):
        df.loc[issue_year_frac < 2012.0, col] = np.nan

    LOG.info(
        "generated %s synthetic loans | %s to %s | overall charge-off rate %.1f%%",
        f"{n:,}",
        spec.start_period,
        spec.end_period,
        100.0 * (df["loan_status"] == "Charged Off").mean(),
    )
    return df


def write_synthetic_raw(
    out_dir: str | Path,
    spec: SyntheticSpec | None = None,
    filename: str = "accepted_2007_to_2018Q4.csv.gz",
) -> Path:
    """Generate and write a synthetic extract to ``out_dir`` in the real file's format."""
    spec = spec or SyntheticSpec()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(spec)
    path = out_dir / filename
    df.to_csv(path, index=False, compression="gzip")
    (out_dir / "SYNTHETIC").write_text(
        "Files in this directory were produced by credit_risk.data.synthetic.\n"
        "They are NOT real LendingClub data. Results computed from them are pipeline\n"
        "checks only and must not be reported as findings. See docs/DATASET.md.\n",
        encoding="utf-8",
    )
    LOG.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path
