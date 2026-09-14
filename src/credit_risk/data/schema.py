"""LendingClub schema: what each column *is*, and — more importantly — when it becomes knowable.

This module is the single source of truth for leakage control. Every column in the accepted-loans
file is assigned to exactly one role. The feature builder may only ever draw from
:data:`DECISION_TIME_NUMERIC` and :data:`DECISION_TIME_CATEGORICAL`; anything in
:data:`POST_ORIGINATION` is available to the *outcome* and *economics* code and to nothing else.

The distinction that matters, and the one most LendingClub write-ups get wrong:

    A column is a valid feature if and only if its value was already determined at the moment
    the lender had to decide. "Correlates with default" is not the test. `last_fico_range_low`
    correlates enormously with default and is worthless, because you only learn it afterwards.

`tests/test_leakage.py` asserts that no post-origination column can reach a model matrix.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Identity / bookkeeping
# ---------------------------------------------------------------------------

ID_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "member_id",  # scrubbed to null in the public files -- no borrower identity exists
    "url",
)

TARGET_SOURCE_COLUMN: Final[str] = "loan_status"
TARGET_COLUMN: Final[str] = "default"
DECISION_TIME_COLUMN: Final[str] = "issue_d"

# ---------------------------------------------------------------------------
# Outcome mapping
# ---------------------------------------------------------------------------

#: Terminal statuses only. Anything absent from this map is non-terminal and, inside a fully
#: matured issue window, is a data anomaly rather than censoring.
LC_STATUS_MAP: Final[dict[str, int]] = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

NON_TERMINAL_STATUSES: Final[tuple[str, ...]] = (
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Issued",
)

# ---------------------------------------------------------------------------
# POST-ORIGINATION — the deny-list.
# ---------------------------------------------------------------------------

#: Columns whose value is only determined after money changed hands. These build the target and
#: the realised cashflows. They must never appear in a feature matrix.
POST_ORIGINATION: Final[tuple[str, ...]] = (
    # --- repayment performance -------------------------------------------------
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    # --- post-origination bureau refresh --------------------------------------
    # The single most seductive leak in this dataset: a FICO pulled *after* origination
    # collapses for borrowers who are about to default. AUC ~0.95 and completely useless.
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    # --- distress / workout programmes ----------------------------------------
    "pymnt_plan",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    # --- the outcome itself ----------------------------------------------------
    "loan_status",
)

#: Post-origination columns the economics layer is explicitly allowed to read, to build
#: realised cashflow and an empirical LGD. Kept as a named subset so the permission is visible.
CASHFLOW_COLUMNS: Final[tuple[str, ...]] = (
    "total_pymnt",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "out_prncp",
)

# ---------------------------------------------------------------------------
# Exposure / pricing — decision-time, but reserved for the economics layer
# ---------------------------------------------------------------------------

#: `funded_amnt` is the exposure actually taken. It is knowable at decision time (it is the
#: decision), but using it as a *PD feature* conflates the action with the risk, so it is
#: routed to the economics layer only. `loan_amnt` (the amount requested) is the feature.
EXPOSURE_COLUMNS: Final[tuple[str, ...]] = ("funded_amnt", "funded_amnt_inv")

#: LendingClub's own underwriting output. Legitimately available at decision time, but a model
#: trained on these is partly reverse-engineering an existing scorecard rather than underwriting
#: from primary evidence. Held out of the `primary` feature set and included in `with_lc_grade`.
LC_UNDERWRITING_COLUMNS: Final[tuple[str, ...]] = ("grade", "sub_grade", "int_rate")

# ---------------------------------------------------------------------------
# Decision-time features
# ---------------------------------------------------------------------------

DECISION_TIME_NUMERIC: Final[tuple[str, ...]] = (
    # loan request
    "loan_amnt",
    "installment",
    # income
    "annual_inc",
    "dti",
    # bureau: scorecard
    "fico_range_low",
    "fico_range_high",
    # bureau: depth
    "open_acc",
    "total_acc",
    # bureau: delinquency
    "delinq_2yrs",
    "delinq_amnt",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "pub_rec",
    "pub_rec_bankruptcies",
    "tax_liens",
    "acc_now_delinq",
    "chargeoff_within_12_mths",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    # bureau: utilisation / balances
    "revol_bal",
    "revol_util",
    "tot_cur_bal",
    "total_rev_hi_lim",
    # bureau: recent activity
    "inq_last_6mths",
)

#: Bureau block that LendingClub added (and partially backfilled) mid-history. Genuinely
#: predictive where present, but its *availability* tracks vintage, so a model can learn
#: "this block is populated" as a proxy for origination date. Gated behind a config flag and
#: additionally subject to the automatic availability-drift guard in features/build.py.
VINTAGE_SENSITIVE_NUMERIC: Final[tuple[str, ...]] = (
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
)

#: Introduced around 2015-12 and essentially absent before it. Excluded by default: in the
#: 2010-2015 modelling window they are ~100% null, so they carry no signal and only risk
#: acting as a vintage indicator.
LATE_ADDITION_NUMERIC: Final[tuple[str, ...]] = (
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
)

#: Joint-application and secondary-applicant fields. Joint applications only appear from ~2015-10,
#: so these are near-entirely null in the modelling window.
JOINT_APPLICATION_COLUMNS: Final[tuple[str, ...]] = (
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
)

DECISION_TIME_CATEGORICAL: Final[tuple[str, ...]] = (
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
)

#: Date columns parsed as dates rather than treated as categoricals.
DATE_COLUMNS: Final[tuple[str, ...]] = (
    "issue_d",
    "earliest_cr_line",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d",
)

#: Raw columns consumed to build relational keys and derived features, then dropped.
RELATIONAL_SOURCE_COLUMNS: Final[tuple[str, ...]] = ("zip_code", "emp_title")

#: Dropped: free text that is redundant with `purpose`, near-constant flags, or borrower prose
#: that was discontinued in 2013-11 and would therefore act as a vintage indicator.
DROPPED_COLUMNS: Final[tuple[str, ...]] = (
    "desc",
    "title",
    "policy_code",
    "disbursement_method",
)

#: Missing here means "no such event has ever been recorded", which is informative and must not
#: be mean-imputed. Handled with an explicit indicator plus a high sentinel.
INFORMATIVE_MISSING_NUMERIC: Final[tuple[str, ...]] = (
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_revol_delinq",
    "mths_since_recent_inq",
    "mths_since_recent_bc",
)

# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

#: Full deny-list the feature builder screens against.
FORBIDDEN_AS_FEATURES: Final[frozenset[str]] = frozenset(
    POST_ORIGINATION
    + ID_COLUMNS
    + EXPOSURE_COLUMNS
    + DROPPED_COLUMNS
    + (TARGET_COLUMN, TARGET_SOURCE_COLUMN)
)


def feature_columns(
    feature_set: str = "primary",
    include_vintage_sensitive: bool = True,
    include_late_additions: bool = False,
    include_joint: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(numeric, categorical)`` raw column names for a named feature set.

    Parameters
    ----------
    feature_set:
        ``"primary"`` excludes LendingClub's own grade/sub-grade/interest rate, asking
        "can we underwrite from primary evidence?". ``"with_lc_grade"`` includes them,
        asking "can we beat the incumbent scorecard?". Both are reported.
    include_vintage_sensitive:
        Include the mid-history bureau block. Still subject to the availability-drift guard.
    include_late_additions:
        Include fields introduced around 2015-12. Off by default: ~100% null in the window.
    include_joint:
        Include joint-application fields. Off by default: near-entirely null in the window.
    """
    if feature_set not in {"primary", "with_lc_grade"}:
        raise ValueError(f"unknown feature_set {feature_set!r}; expected primary|with_lc_grade")

    numeric = list(DECISION_TIME_NUMERIC)
    categorical = list(DECISION_TIME_CATEGORICAL)

    if include_vintage_sensitive:
        numeric += list(VINTAGE_SENSITIVE_NUMERIC)
    if include_late_additions:
        numeric += list(LATE_ADDITION_NUMERIC)
    if include_joint:
        numeric += [c for c in JOINT_APPLICATION_COLUMNS if not c.startswith("verification")]
        categorical += ["verification_status_joint"]
    if feature_set == "with_lc_grade":
        numeric += ["int_rate"]
        categorical += ["grade", "sub_grade"]

    return tuple(dict.fromkeys(numeric)), tuple(dict.fromkeys(categorical))


def assert_no_leakage(columns: list[str] | tuple[str, ...]) -> None:
    """Raise if any forbidden column appears in a proposed feature matrix.

    Called by the feature builder on every build. It raises rather than warns: a warning in a
    log that nobody reads is not a control.
    """
    offenders = sorted(set(columns) & FORBIDDEN_AS_FEATURES)
    if offenders:
        raise ValueError(
            "Post-origination or forbidden columns reached the feature matrix: "
            f"{offenders}. See docs/LEAKAGE.md."
        )
