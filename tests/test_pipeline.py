"""Data preparation, config, monitoring, explainability and the GNN."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import schema
from credit_risk.data.download import detect_data_source
from credit_risk.data.prepare import normalise_employment, parse_lc_month
from credit_risk.data.splits import SplitWindows, assign_split
from credit_risk.data.synthetic import SyntheticSpec, generate
from credit_risk.utils.config import Config, load_config, parse_overrides


# ---------------------------------------------------------------------------
# Synthetic generator: it must be schema-faithful or it tests nothing
# ---------------------------------------------------------------------------

def test_synthetic_frame_covers_the_real_schema():
    df = generate(SyntheticSpec(n_loans=2_000, seed=3))
    required = set(
        schema.DECISION_TIME_NUMERIC
        + schema.DECISION_TIME_CATEGORICAL
        + schema.POST_ORIGINATION
        + schema.EXPOSURE_COLUMNS
        + schema.LC_UNDERWRITING_COLUMNS
        + schema.RELATIONAL_SOURCE_COLUMNS
    )
    missing = required - set(df.columns)
    assert not missing, f"synthetic generator is missing real columns: {sorted(missing)}"


def test_synthetic_formats_match_lendingclub():
    df = generate(SyntheticSpec(n_loans=500, seed=4))
    assert set(df["term"].unique()) <= {" 36 months", " 60 months"}
    assert df["issue_d"].str.match(r"^[A-Z][a-z]{2}-\d{4}$").all()
    assert df["zip_code"].str.match(r"^\d{3}xx$").all()
    assert df["member_id"].isna().all()  # scrubbed in the real public files


def test_synthetic_is_deterministic():
    a = generate(SyntheticSpec(n_loans=500, seed=9))
    b = generate(SyntheticSpec(n_loans=500, seed=9))
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_cashflows_are_internally_consistent():
    df = generate(SyntheticSpec(n_loans=3_000, seed=5))
    paid = df[df["loan_status"] == "Fully Paid"]
    assert np.allclose(paid["total_rec_prncp"], paid["loan_amnt"], atol=1.0)
    charged = df[df["loan_status"] == "Charged Off"]
    assert (charged["total_rec_prncp"] <= charged["loan_amnt"] + 1.0).all()
    assert (charged["recoveries"] >= 0).all()


def test_synthetic_reproduces_the_vintage_missingness_trap():
    """The bureau block must be absent in early vintages, as it is in the real file."""
    df = generate(SyntheticSpec(n_loans=6_000, seed=6))
    year = df["issue_d"].str.slice(-4).astype(int)
    early, late = year <= 2011, year >= 2014
    column = "num_sats"
    assert df.loc[early, column].isna().mean() > 0.9
    assert df.loc[late, column].isna().mean() < 0.2


def test_synthetic_post_origination_fico_leaks_by_design():
    """`last_fico_range_low` must be a strong leak, so the leakage tests have a real offender."""
    df = generate(SyntheticSpec(n_loans=5_000, seed=8))
    terminal = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])]
    bad = terminal["loan_status"] == "Charged Off"
    assert terminal.loc[bad, "last_fico_range_low"].mean() < (
        terminal.loc[~bad, "last_fico_range_low"].mean() - 50
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_lc_month_handles_the_real_format_and_nulls():
    parsed = parse_lc_month(pd.Series(["Dec-2015", "Jan-2008", None, ""]))
    assert str(parsed.iloc[0]) == "2015-12"
    assert str(parsed.iloc[1]) == "2008-01"
    assert parsed.iloc[2] is pd.NaT or pd.isna(parsed.iloc[2])


def test_normalise_employment_collapses_variants():
    values = pd.Series(["  Registered  NURSE ", "registered nurse", "R.N.", None, "n/a"])
    normalised = normalise_employment(values)
    assert normalised.iloc[0] == normalised.iloc[1] == "registered nurse"
    assert pd.isna(normalised.iloc[3])
    assert pd.isna(normalised.iloc[4])


def test_emp_title_semantics_break_is_flagged(loans):
    """LendingClub swapped employer name for job title on 2013-09-23; the flag must record it."""
    if "emp_title_is_employer_name" not in loans.columns:
        pytest.skip("flag not present")
    early = loans["issue_period"] < pd.Period("2013-10", freq="M")
    assert (loans.loc[early, "emp_title_is_employer_name"]).all()
    assert not (loans.loc[~early, "emp_title_is_employer_name"]).any()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_assign_split_maps_periods_to_windows():
    windows = SplitWindows()
    periods = pd.Series(
        pd.PeriodIndex(["2008-06", "2011-03", "2014-09", "2015-07", "2017-02", "2020-01"],
                       freq="M")
    )
    assigned = assign_split(periods, windows)
    assert assigned.tolist() == [
        "history", "train", "valid", "test", "monitor", "excluded"
    ]


def test_modelled_window_spans_train_to_test():
    start, end = SplitWindows().modelled_window
    assert str(start) == "2010-01"
    assert str(end) == "2015-12"


def test_target_mapping_covers_terminal_statuses_only():
    assert set(schema.LC_STATUS_MAP.values()) == {0, 1}
    for status in schema.NON_TERMINAL_STATUSES:
        assert status not in schema.LC_STATUS_MAP


def test_prepare_report_counts_are_consistent(prepare_report, loans):
    assert prepare_report.rows_read >= prepare_report.rows_after_term_filter
    assert sum(prepare_report.split_counts.values()) == len(loans)


def test_default_rate_is_plausible(prepare_report):
    for split, rate in prepare_report.default_rate_by_split.items():
        assert 0.03 < rate < 0.45, f"{split} default rate {rate} is implausible"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_supports_attribute_and_item_access():
    config = Config({"model": {"num_leaves": 31}, "seed": 7})
    assert config.model.num_leaves == 31
    assert config["model"]["num_leaves"] == 31
    assert config.get_path("model.num_leaves") == 31
    assert config.get_path("model.missing", "fallback") == "fallback"


def test_config_missing_key_error_lists_alternatives():
    with pytest.raises(AttributeError, match="available keys"):
        Config({"a": 1}).nope


def test_parse_overrides_builds_nested_typed_values():
    result = parse_overrides(["model.num_leaves=64", "seed=7", "flag=true", "x.y.z=0.5"])
    assert result == {
        "model": {"num_leaves": 64},
        "seed": 7,
        "flag": True,
        "x": {"y": {"z": 0.5}},
    }


def test_parse_overrides_rejects_malformed():
    with pytest.raises(ValueError, match="key.path=value"):
        parse_overrides(["nonsense"])


def test_shipped_configs_all_parse():
    import glob

    from credit_risk.utils.config import project_root

    files = glob.glob(str(project_root() / "configs" / "**" / "*.yaml"), recursive=True)
    assert files
    for path in files:
        assert len(load_config(path)) > 0


def test_experiment_config_inherits_from_base():
    config = load_config("configs/experiments/exp06_rl.yaml")
    assert "economics" in config          # from the base
    assert "experiment" in config         # from the child
    assert config.experiment.id == "EXP06"


def test_detect_data_source_flags_synthetic(tmp_path):
    assert detect_data_source(tmp_path) == "absent"
    (tmp_path / "accepted.csv.gz").write_bytes(b"x")
    assert detect_data_source(tmp_path) == "real"
    (tmp_path / "SYNTHETIC").write_text("marker")
    assert detect_data_source(tmp_path) == "synthetic"


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

def test_psi_is_zero_for_identical_distributions(rng):
    from credit_risk.monitoring.drift import population_stability_index

    values = rng.normal(size=20_000)
    assert abs(population_stability_index(values, values)) < 1e-6


def test_psi_grows_with_the_size_of_the_shift(rng):
    from credit_risk.monitoring.drift import classify_psi, population_stability_index

    reference = rng.normal(size=20_000)
    small = population_stability_index(reference, reference + 0.1)
    large = population_stability_index(reference, reference + 1.5)
    assert small < large
    assert classify_psi(small) in {"stable", "moderate"}
    assert classify_psi(large) == "significant"


def test_prediction_drift_reports_approval_rate_impact(rng):
    from credit_risk.monitoring.drift import prediction_drift

    reference = rng.beta(2, 10, 10_000)
    current = rng.beta(3, 10, 10_000)  # riskier population
    report = prediction_drift(reference, current, thresholds=(0.15,))
    assert report.current_mean > report.reference_mean
    # A riskier population means fewer approvals under a fixed cut-off.
    assert report.approval_rate_shift["tau=0.15"] < 0


def test_calibration_drift_tracks_vintages(loans, rng):
    from credit_risk.monitoring.drift import calibration_drift

    labelled = loans[loans["split"].isin(["train", "valid", "test"])].copy()
    labelled["_pd"] = np.clip(
        0.15 + 0.05 * rng.standard_normal(len(labelled)), 0.01, 0.9
    )
    table = calibration_drift(labelled, "_pd", freq="Y")
    assert len(table) >= 2
    assert {"vintage", "observed_expected", "roc_auc", "brier"} <= set(table.columns)
    assert table["observed_expected"].notna().all()


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def test_reason_phrases_are_mapped_not_invented():
    from credit_risk.explainability.reasons import REASON_PHRASES, _phrase_for

    assert _phrase_for("fico_mid", 1.0) == REASON_PHRASES["fico_mid"][0]
    assert _phrase_for("fico_mid", -1.0) == REASON_PHRASES["fico_mid"][1]
    # An unmapped feature falls back to its own name rather than borrowed prose.
    assert "totally_unknown_column" in _phrase_for("totally_unknown_column", 1.0)


def test_missingness_indicator_flips_the_sense():
    """`mths_since_last_delinq_missing` being positive means *no* delinquency on file."""
    from credit_risk.explainability.reasons import _phrase_for

    assert _phrase_for("mths_since_last_delinq_missing", 1.0) == "No recent delinquency"


def test_explanation_deduplicates_collinear_phrases():
    from credit_risk.explainability.reasons import explain_row

    explanation = explain_row(
        shap_values=np.array([0.5, 0.4, -0.3]),
        feature_names=["revol_util", "revol_util_clipped", "fico_mid"],
        feature_values=pd.Series(dtype=float),
        pd_estimate=0.2,
        decision="Decline",
    )
    phrases = [r.phrase for r in explanation.reasons_against + explanation.reasons_for]
    assert len(phrases) == len(set(phrases))


def test_explanation_renders_and_bands(rng):
    from credit_risk.explainability.reasons import explain_row, risk_band

    assert risk_band(0.01) == "Very low"
    assert risk_band(0.5) == "Very high"
    explanation = explain_row(
        shap_values=rng.normal(size=6),
        feature_names=["fico_mid", "dti_clean", "loan_to_income", "revol_util", "pub_rec",
                       "credit_history_months"],
        feature_values=pd.Series(dtype=float),
        pd_estimate=0.068,
        decision="Approve",
        exposure=12_000.0,
    )
    rendered = explanation.render()
    assert "6.8%" in rendered
    assert "$12,000" in rendered
    assert "Principal reasons" in rendered


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------

@pytest.mark.needs_torch
def test_graphsage_trains_and_predicts(enriched, split_frames):
    pytest.importorskip("torch")  # guard only; the symbol itself is not needed here
    from credit_risk.features.build import FeatureBuilder, FeatureSpec, to_numeric_matrix
    from credit_risk.graph.construction import build_graph
    from credit_risk.models.gnn import GNNConfig, GraphSAGETrainer

    df = enriched.reset_index(drop=True)
    graph = build_graph(df)
    masks = {s: (df["split"] == s).to_numpy() for s in ("train", "valid", "test")}

    builder = FeatureBuilder(FeatureSpec()).fit(df[masks["train"]], df[masks["valid"]])
    X = builder.transform(df)
    _, _, means, stds = to_numeric_matrix(
        X[masks["train"]], builder.categorical_features, builder.state.categorical_levels
    )
    features, names, _, _ = to_numeric_matrix(
        X, builder.categorical_features, builder.state.categorical_levels, means=means, stds=stds
    )

    y = df["default"].astype("float32").to_numpy()
    idx = {s: np.flatnonzero(masks[s]) for s in masks}
    trainer = GraphSAGETrainer(
        GNNConfig(max_epochs=3, patience=3, batch_size=256, hidden_dim=16, embed_dim=8)
    )
    trainer.fit(
        features, graph.neighbour_index, graph.neighbour_mask,
        idx["train"], y[idx["train"]], idx["valid"], y[idx["valid"]],
    )
    scores = trainer.predict(idx["test"])
    assert len(scores) == len(idx["test"])
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert trainer.embeddings(idx["test"][:64]).shape == (64, 8)


@pytest.mark.needs_torch
def test_graphsage_default_does_not_distort_the_base_rate():
    """`balance_classes` must default off, or the scores cannot feed the decision layer."""
    from credit_risk.models.gnn import GNNConfig

    assert GNNConfig().balance_classes is False


@pytest.mark.needs_torch
def test_sage_layer_ignores_masked_neighbours():
    """A padded neighbour slot must contribute nothing to the aggregation."""
    torch = pytest.importorskip("torch")
    from credit_risk.models.gnn import SAGELayer

    layer = SAGELayer(in_dim=4, out_dim=3, n_relations=1)
    h_self = torch.zeros(2, 4)
    neighbours = torch.randn(2, 1, 3, 4)
    mask = torch.tensor([[[True, False, False]], [[True, False, False]]])

    # Corrupting a masked-out slot must not change the output.
    first = layer(h_self, neighbours, mask)
    neighbours[:, :, 1:, :] = 1e6
    second = layer(h_self, neighbours, mask)
    assert torch.allclose(first, second)
