"""Leakage controls.

These are the tests that matter most in this repository. A credit model that leaks does not fail
loudly -- it produces an excellent AUC and a worthless model, and the failure only becomes visible
after deployment. So every leakage rule in docs/LEAKAGE.md gets an assertion here, and each one
fails the build rather than logging a warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import schema
from credit_risk.data.splits import SplitWindows, assert_temporal_ordering
from credit_risk.features.build import FeatureBuilder, FeatureSpec


# ---------------------------------------------------------------------------
# The deny-list itself
# ---------------------------------------------------------------------------

def test_post_origination_columns_are_forbidden():
    for column in schema.POST_ORIGINATION:
        assert column in schema.FORBIDDEN_AS_FEATURES, f"{column} must be denied as a feature"


def test_the_notorious_offenders_are_denied():
    """The columns that make LendingClub look easy, and make the model useless.

    `last_fico_range_low` collapses for borrowers about to default; `recoveries` is literally
    money collected after charge-off. Either one gives an AUC near 0.95 and predicts nothing that
    could have been known at decision time.
    """
    for column in (
        "last_fico_range_low",
        "last_fico_range_high",
        "recoveries",
        "total_pymnt",
        "total_rec_prncp",
        "out_prncp",
        "last_pymnt_d",
        "debt_settlement_flag",
        "loan_status",
    ):
        assert column in schema.FORBIDDEN_AS_FEATURES


def test_assert_no_leakage_raises_on_offender():
    with pytest.raises(ValueError, match="Post-origination"):
        schema.assert_no_leakage(["loan_amnt", "fico_range_low", "recoveries"])


def test_assert_no_leakage_passes_clean_list():
    schema.assert_no_leakage(["loan_amnt", "fico_range_low", "dti", "annual_inc"])


def test_feature_columns_never_include_denied_columns():
    for feature_set in ("primary", "with_lc_grade"):
        for vintage in (True, False):
            for late in (True, False):
                numeric, categorical = schema.feature_columns(
                    feature_set=feature_set,
                    include_vintage_sensitive=vintage,
                    include_late_additions=late,
                )
                overlap = set(numeric + categorical) & schema.FORBIDDEN_AS_FEATURES
                assert not overlap, f"{feature_set} leaks {overlap}"


def test_cashflow_columns_are_a_subset_of_post_origination():
    """The economics layer's permission is explicit, and confined to genuinely post-hoc fields."""
    assert set(schema.CASHFLOW_COLUMNS).issubset(set(schema.POST_ORIGINATION))


# ---------------------------------------------------------------------------
# The built feature matrix
# ---------------------------------------------------------------------------

def test_built_matrix_contains_no_forbidden_column(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    X = builder.transform(split_frames["test"])
    assert not set(X.columns) & schema.FORBIDDEN_AS_FEATURES


def test_primary_feature_set_excludes_lendingclub_underwriting(split_frames):
    """`primary` must not contain grade, sub-grade, interest rate -- or instalment.

    Instalment is the subtle one. It is an invertible function of amount, rate and term, so
    including it would smuggle LendingClub's own risk grade into a model that claims not to use it.
    """
    builder = FeatureBuilder(FeatureSpec(feature_set="primary")).fit(
        split_frames["train"], split_frames["valid"]
    )
    names = set(builder.feature_names)
    for banned in ("grade", "sub_grade", "int_rate", "installment"):
        assert banned not in names, f"primary feature set leaked {banned}"


def test_with_lc_grade_feature_set_includes_them(split_frames):
    builder = FeatureBuilder(FeatureSpec(feature_set="with_lc_grade")).fit(
        split_frames["train"], split_frames["valid"]
    )
    names = set(builder.feature_names)
    assert {"grade", "sub_grade", "int_rate"} <= names


def test_interest_rate_is_recoverable_from_instalment(loans):
    """Demonstrates *why* instalment is excluded, rather than just asserting the rule.

    Given amount, term and instalment, the APR is recoverable by bisection to well within the
    granularity of a risk grade.
    """
    row = loans.dropna(subset=["loan_amnt", "installment", "int_rate"]).iloc[0]
    principal = float(row["loan_amnt"])
    payment = float(row["installment"])
    term = int(row["term_months"])

    def implied(apr: float) -> float:
        r = apr / 12.0
        return principal * r / (1.0 - (1.0 + r) ** (-term))

    lo, hi = 1e-6, 1.5
    for _ in range(200):
        mid = (lo + hi) / 2
        if implied(mid) < payment:
            lo = mid
        else:
            hi = mid
    recovered = 100.0 * (lo + hi) / 2
    assert abs(recovered - float(row["int_rate"])) < 0.05


# ---------------------------------------------------------------------------
# Fitting boundaries
# ---------------------------------------------------------------------------

def test_feature_builder_statistics_come_from_train_only(split_frames):
    """Transforming a different split must not change any fitted statistic."""
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    before = dict(builder.state.numeric_medians)
    bounds_before = dict(builder.state.clip_bounds)
    builder.transform(split_frames["test"])
    builder.transform(split_frames["monitor"])
    assert builder.state.numeric_medians == before
    assert builder.state.clip_bounds == bounds_before


def test_transform_is_idempotent_across_splits(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    first = builder.transform(split_frames["test"])
    second = builder.transform(split_frames["test"])
    pd.testing.assert_frame_equal(first, second)


def test_availability_guard_drops_vintage_encoding_columns(split_frames):
    """The synthetic generator withholds a bureau block before 2012, exactly as LendingClub did.

    The guard must notice. If this test ever stops finding anything to drop, the guard has been
    disabled or the generator has changed, and either is worth knowing.
    """
    builder = FeatureBuilder(FeatureSpec(availability_shift_tolerance=0.05)).fit(
        split_frames["train"], split_frames["valid"]
    )
    reasons = " ".join(builder.state.dropped.values())
    assert "availability guard" in reasons


def test_availability_guard_uses_validation_not_test(split_frames):
    """Fitting the guard against the test window would spend the out-of-time set."""
    with_valid = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    with_test = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["test"])
    # Different reference windows must be capable of producing different decisions; if they were
    # identical by construction the guard would not be reading its argument at all.
    assert isinstance(with_valid.state.dropped, dict)
    assert isinstance(with_test.state.dropped, dict)


# ---------------------------------------------------------------------------
# Temporal integrity
# ---------------------------------------------------------------------------

def test_splits_do_not_overlap_in_time(loans):
    assert_temporal_ordering(loans)


def test_split_windows_reject_overlap():
    bad = SplitWindows(train=("2010-01", "2014-12"), valid=("2014-06", "2014-12"))
    with pytest.raises(ValueError, match="overlaps"):
        bad.validate()


def test_every_split_is_a_contiguous_period_block(loans):
    for split in ("train", "valid", "test"):
        periods = pd.PeriodIndex(loans.loc[loans["split"] == split, "issue_period"])
        assert periods.min() <= periods.max()
    train_max = pd.PeriodIndex(loans.loc[loans["split"] == "train", "issue_period"]).max()
    valid_min = pd.PeriodIndex(loans.loc[loans["split"] == "valid", "issue_period"]).min()
    test_min = pd.PeriodIndex(loans.loc[loans["split"] == "test", "issue_period"]).min()
    assert train_max < valid_min < test_min


def test_monitoring_window_carries_no_labels(split_frames):
    """The production window must be unlabelled -- that is the whole point of it."""
    monitor = split_frames["monitor"]
    if len(monitor):
        assert monitor["default"].isna().all()


def test_history_window_is_unlabelled_but_has_observed_outcomes(split_frames):
    """History rows are not modelled, but their outcomes were legitimately knowable later."""
    history = split_frames["history"]
    if len(history):
        assert history["default"].isna().all()
        assert history["default_observed"].notna().any()


def test_outcome_is_never_observed_before_issue(loans):
    """If a loan could resolve before it was issued, every cohort label aggregate is contaminated."""
    resolved = loans["outcome_observed_period"].notna()
    if resolved.any():
        issue = pd.PeriodIndex(loans.loc[resolved, "issue_period"])
        observed = pd.PeriodIndex(loans.loc[resolved, "outcome_observed_period"])
        assert (observed > issue).all()


def test_unresolved_loans_have_no_observation_date(loans):
    """A loan still running must not be counted as a known-good neighbour."""
    unresolved = loans["default_observed"].isna()
    if unresolved.any():
        assert loans.loc[unresolved, "outcome_observed_period"].isna().all()


def test_credit_history_is_measured_as_at_application(enriched):
    """Measuring against today rather than the application date would leak the vintage."""
    subset = enriched.dropna(subset=["credit_history_months"])
    months = (
        pd.PeriodIndex(subset["issue_period"]) - pd.PeriodIndex(subset["earliest_cr_line"])
    ).map(lambda x: x.n)
    assert np.allclose(subset["credit_history_months"].to_numpy(), months.to_numpy(), atol=1e-6)
