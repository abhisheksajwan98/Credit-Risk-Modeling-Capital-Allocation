"""Metrics, calibration, feature building and the models themselves.

The centrepiece is :func:`test_auc_is_blind_to_calibration`, which demonstrates the claim EXP02
rests on: a monotone transform of the score leaves every ranking metric untouched while making the
probabilities arbitrarily wrong. If that test ever fails, the argument for the whole calibration
layer has gone with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.evaluation.metrics import (
    compute_metrics,
    expected_calibration_error,
    gains_table,
    ks_statistic,
    reliability_table,
)
from credit_risk.features.build import FeatureBuilder, FeatureSpec, to_numeric_matrix
from credit_risk.models.baseline import LogisticBaseline, LogisticConfig
from credit_risk.models.boosting import BoostingConfig, BoostingModel
from credit_risk.models.calibration import ProbabilityCalibrator, compare_calibrations


# ---------------------------------------------------------------------------
# The calibration argument
# ---------------------------------------------------------------------------

def test_auc_is_blind_to_calibration(rng):
    """Cube every probability. Ranking metrics do not move; probability metrics collapse.

    This is exactly why a model reported only by AUC can be badly wrong about the level of risk,
    and why the decision layer -- which multiplies PD by an exposure -- needs calibration rather
    than ranking.
    """
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)

    honest = compute_metrics(y, p)
    distorted = compute_metrics(y, p**3)  # strictly monotone, so the ordering is identical

    assert distorted.roc_auc == pytest.approx(honest.roc_auc, abs=1e-9)
    assert distorted.ks == pytest.approx(honest.ks, abs=1e-9)
    assert distorted.brier > honest.brier * 1.2
    assert distorted.ece > honest.ece * 2


def test_calibration_slope_detects_overconfidence(rng):
    """Over-spread predictions show up as a calibration slope below 1."""
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)
    logit = np.log(p / (1 - p))
    overconfident = 1.0 / (1.0 + np.exp(-1.8 * logit))  # stretched away from the base rate
    metrics = compute_metrics(y, overconfident)
    assert metrics.calibration_slope < 0.9


def test_level_bias_ratios_point_in_opposite_directions(rng):
    """The two ratios are reciprocals, and their names must match their direction.

    `predicted_observed_ratio` above 1 means the model OVER-states risk. The conventional
    credit-risk O/E (observed over expected) above 1 means it UNDER-states. These were previously
    conflated: one field computed predicted/observed under the name `observed_expected_ratio`.
    """
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)

    over = compute_metrics(y, np.clip(p * 2.0, 1e-6, 1 - 1e-6))
    assert over.predicted_observed_ratio > 1.5      # predicts twice the risk
    assert over.observed_expected_ratio < 0.7       # so observed/expected is well below 1
    assert over.observed_expected_ratio == pytest.approx(1.0 / over.predicted_observed_ratio)

    under = compute_metrics(y, np.clip(p * 0.5, 1e-6, 1 - 1e-6))
    assert under.predicted_observed_ratio < 0.7
    assert under.observed_expected_ratio > 1.4


def test_perfect_predictions_score_perfectly():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(y, p)
    assert metrics.roc_auc == pytest.approx(1.0)
    assert metrics.brier < 1e-6


def test_ece_is_zero_for_a_calibrated_model(rng):
    n = 60_000
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    ece, mce = expected_calibration_error(y, p, n_bins=20)
    assert ece < 0.01
    assert mce < 0.05


def test_ks_matches_a_manual_computation(rng):
    n = 5_000
    p = rng.uniform(0, 1, n)
    y = (rng.random(n) < p).astype(int)
    assert 0.0 <= ks_statistic(y, p) <= 1.0


def test_reliability_and_gains_tables_are_coherent(rng):
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)

    reliability = reliability_table(y, p, n_bins=10)
    assert reliability["n"].sum() == n
    assert reliability["mean_predicted"].is_monotonic_increasing

    gains = gains_table(y, p, n_bands=10)
    assert gains["cum_population_share"].iloc[-1] == pytest.approx(1.0)
    assert gains["cum_bad_share"].iloc[-1] == pytest.approx(1.0)
    # Riskiest decile first, so the observed rate should fall across bands.
    assert gains["observed_rate"].iloc[0] > gains["observed_rate"].iloc[-1]


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------

def test_identity_calibrator_is_a_no_op(rng):
    p = rng.uniform(0.01, 0.99, 1000)
    assert np.allclose(ProbabilityCalibrator("identity").transform(p), p)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrators_fix_a_distorted_model(rng, method):
    n = 40_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)
    distorted = p**2

    half = n // 2
    calibrator = ProbabilityCalibrator(method).fit(y[:half], distorted[:half])
    fixed = calibrator.transform(distorted[half:])

    before = compute_metrics(y[half:], distorted[half:])
    after = compute_metrics(y[half:], fixed)
    assert after.brier < before.brier
    assert after.ece < before.ece


def test_calibration_preserves_ranking(rng):
    """Platt scaling is strictly monotone, so AUC must be untouched to numerical precision."""
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)
    fixed = ProbabilityCalibrator("platt").fit_transform(y, p**2)
    assert compute_metrics(y, fixed).roc_auc == pytest.approx(
        compute_metrics(y, p**2).roc_auc, abs=1e-6
    )


def test_calibrator_rejects_single_class(rng):
    with pytest.raises(ValueError, match="both classes"):
        ProbabilityCalibrator("platt").fit(np.zeros(100), rng.uniform(size=100))


def test_calibrator_requires_fit_before_transform(rng):
    with pytest.raises(RuntimeError, match="before fit"):
        ProbabilityCalibrator("isotonic").transform(rng.uniform(size=10))


def test_compare_calibrations_returns_every_method(rng):
    n = 20_000
    p = rng.beta(2, 8, n)
    y = (rng.random(n) < p).astype(int)
    half = n // 2
    comparison = compare_calibrations(y[:half], p[:half], y[half:], p[half:])
    assert set(comparison.per_method) == {"identity", "platt", "isotonic"}
    assert "Brier" in comparison.verdict() or "reduced" in comparison.verdict()
    assert len(comparison.table()) == 3


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def test_transform_produces_no_missing_values(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    X = builder.transform(split_frames["test"])
    assert int(X.isna().sum().sum()) == 0


def test_informative_missingness_gets_an_indicator(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    X = builder.transform(split_frames["test"])
    indicators = [c for c in X.columns if c.endswith("_missing")]
    assert indicators, "months-since fields should produce explicit missingness indicators"
    for column in indicators:
        assert set(np.unique(X[column])) <= {0, 1}


def test_unseen_categorical_level_maps_to_other(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    frame = split_frames["test"].copy()
    column = builder.categorical_features[0]
    frame[column] = "a-level-that-never-occurred"
    X = builder.transform(frame)
    assert (X[column].astype(str) == "__other__").all()


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        FeatureBuilder(FeatureSpec()).transform(None)


def test_numeric_matrix_shapes_match_across_splits(split_frames):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    X_train = builder.transform(split_frames["train"])
    X_test = builder.transform(split_frames["test"])
    matrix_train, names, means, stds = to_numeric_matrix(
        X_train, builder.categorical_features, builder.state.categorical_levels
    )
    matrix_test, names_test, _, _ = to_numeric_matrix(
        X_test, builder.categorical_features, builder.state.categorical_levels,
        means=means, stds=stds,
    )
    assert names == names_test
    assert matrix_train.shape[1] == matrix_test.shape[1]
    assert np.isfinite(matrix_test).all()


def test_numeric_matrix_uses_supplied_statistics(split_frames):
    """Recomputing standardisation per split would leak each split's own distribution."""
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    X_train = builder.transform(split_frames["train"])
    _, _, means, stds = to_numeric_matrix(
        X_train, builder.categorical_features, builder.state.categorical_levels
    )
    again, _, means_again, _ = to_numeric_matrix(
        X_train, builder.categorical_features, builder.state.categorical_levels,
        means=means, stds=stds,
    )
    assert np.allclose(means, means_again)


def test_feature_builder_round_trips_through_disk(split_frames, tmp_path):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    path = builder.save(tmp_path / "fb.pkl")
    restored = FeatureBuilder.load(path)
    assert restored.feature_names == builder.feature_names
    import pandas as pd

    pd.testing.assert_frame_equal(
        restored.transform(split_frames["test"]), builder.transform(split_frames["test"])
    )
    assert (path.with_suffix(".json")).exists()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def test_logistic_baseline_learns_something(split_frames, labels):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    model = LogisticBaseline(
        builder.state.numeric_features + builder.state.indicator_features,
        builder.categorical_features,
        LogisticConfig(),
    ).fit(builder.transform(split_frames["train"]), labels["train"])
    scores = model.predict_proba(builder.transform(split_frames["test"]))
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert compute_metrics(labels["test"], scores).roc_auc > 0.55
    assert len(model.coefficients()) > 10


def test_boosting_learns_and_reports_importance(split_frames, labels):
    builder = FeatureBuilder(FeatureSpec()).fit(split_frames["train"], split_frames["valid"])
    model = BoostingModel(
        builder.categorical_features, BoostingConfig(n_estimators=200, early_stopping_rounds=30)
    ).fit(
        builder.transform(split_frames["train"]), labels["train"],
        builder.transform(split_frames["valid"]), labels["valid"], seed=0,
    )
    scores = model.predict_proba(builder.transform(split_frames["test"]))
    assert compute_metrics(labels["test"], scores).roc_auc > 0.55
    importance = model.feature_importance()
    assert len(importance) > 10
    assert importance["importance"].iloc[0] >= importance["importance"].iloc[-1]


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="before fit"):
        BoostingModel([]).predict_proba(None)
    with pytest.raises(RuntimeError, match="before fit"):
        LogisticBaseline([], []).predict_proba(None)
