"""Home Credit Default Risk adapter -- Layer A only.

Why this exists
---------------
Home Credit was the nominated primary dataset and was rejected for the reasons in
``docs/DATASET.md``. It is retained here as a *secondary* loader so that EXP01-EXP03 can be
reproduced on a completely different schema, which is worth demonstrating: it shows the pipeline
is not fitted to one file's quirks.

What it deliberately does not support
-------------------------------------
**Layer B (graph) and Layer C (decision/RL) are not available on this dataset**, and the loader
refuses rather than improvising:

* *No graph.* Home Credit's tables are strictly hierarchical -- a borrower and *their own* bureau
  records and prior applications. There are no borrower-to-borrower links, so any graph would be a
  k-nearest-neighbour construction in feature space. That is a smoothing regulariser, not
  relational financial information, and running it would answer a different question from the one
  the project asks.
* *No decision layer.* There is no interest rate, no recovery amount and no realised cashflow, so
  every quantity in an expected-profit calculation would have to be invented.

**Splits are grouped-random, not out-of-time**, because they cannot be otherwise: Home Credit
contains no absolute calendar date at all. Every temporal field (``DAYS_BIRTH``, ``DAYS_CREDIT``,
``DAYS_DECISION``, ...) is expressed in days relative to each individual application, so there is
no axis to sort on. A random split lets the model see the same credit cycle it is scored on, which
inflates every metric relative to what deployment would give. That is stated on every result this
adapter produces rather than left as a footnote -- it is the single biggest reason the dataset was
not chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.utils.runtime import get_logger

LOG = get_logger("data.home_credit")

TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"

#: Home Credit's own leakage risks are milder than LendingClub's -- the competition file was
#: curated -- but these are still not decision-time attributes of the applicant.
HOME_CREDIT_DENYLIST: tuple[str, ...] = (
    "SK_ID_CURR",
    "TARGET",
    # Bureau-side identifiers that index other tables rather than describing the applicant.
    "SK_ID_BUREAU",
    "SK_ID_PREV",
)

#: Sentinel Home Credit uses for "not employed": 365243 days is exactly 1000 years and is not a
#: measurement. Leaving it in place puts a 1000-year employment history into the model.
DAYS_EMPLOYED_SENTINEL = 365_243


@dataclass
class HomeCreditReport:
    rows: int = 0
    n_features: int = 0
    default_rate: float = float("nan")
    split_counts: dict[str, int] = field(default_factory=dict)
    sentinel_rows_fixed: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "n_features": self.n_features,
            "default_rate": self.default_rate,
            "split_counts": self.split_counts,
            "sentinel_rows_fixed": self.sentinel_rows_fixed,
            "notes": self.notes,
        }


def load_application(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load ``application_train.csv``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Home Credit is a separate download:\n"
            "  kaggle competitions download -c home-credit-default-risk\n"
            "Accepting the competition rules on Kaggle is required first. Note that the "
            "competition's data terms permit academic and non-commercial use; see docs/DATASET.md."
        )
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def aggregate_bureau(path: str | Path) -> pd.DataFrame:
    """Aggregate ``bureau.csv`` to one row per applicant.

    Hierarchical aggregation, *not* a graph. Each applicant's own external credit records are
    collapsed into counts and means. This is the same operation the LendingClub cohort features
    perform over a peer group, and the contrast is instructive: there, the aggregation crosses
    borrowers; here it never leaves one.
    """
    bureau = pd.read_csv(path, low_memory=False)
    numeric = bureau.select_dtypes(include=[np.number]).columns.difference([ID_COLUMN])

    grouped = bureau.groupby(ID_COLUMN)
    aggregated = grouped[list(numeric)].agg(["mean", "max", "sum"])
    aggregated.columns = [f"BUREAU_{a}_{b}".upper() for a, b in aggregated.columns]
    aggregated["BUREAU_RECORD_COUNT"] = grouped.size()

    if "CREDIT_ACTIVE" in bureau.columns:
        active = (
            bureau.assign(_active=(bureau["CREDIT_ACTIVE"] == "Active").astype(int))
            .groupby(ID_COLUMN)["_active"]
            .sum()
        )
        aggregated["BUREAU_ACTIVE_COUNT"] = active

    return aggregated.reset_index()


def prepare_home_credit(
    application_path: str | Path,
    bureau_path: str | Path | None = None,
    valid_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 42,
    nrows: int | None = None,
) -> tuple[pd.DataFrame, HomeCreditReport]:
    """Produce a modelling table with the same column contract as the LendingClub path.

    Returns a frame carrying ``default`` and ``split`` so that :class:`FeatureBuilder`-equivalent
    code and the metric suite work unchanged.
    """
    report = HomeCreditReport()
    df = load_application(application_path, nrows=nrows)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"{application_path} has no {TARGET_COLUMN} column. This adapter expects "
            "application_train.csv, not application_test.csv (which is unlabelled)."
        )

    # -- the sentinel that ruins DAYS_EMPLOYED -------------------------------
    if "DAYS_EMPLOYED" in df.columns:
        sentinel = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        report.sentinel_rows_fixed = int(sentinel.sum())
        if sentinel.any():
            # Replace with NaN and keep an explicit flag: "not employed" is information, and
            # 365243 days is not a duration.
            df.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
            df["DAYS_EMPLOYED_ANOMALY"] = sentinel.astype(int)
            report.notes.append(
                f"DAYS_EMPLOYED == {DAYS_EMPLOYED_SENTINEL} on {report.sentinel_rows_fixed:,} "
                "rows: replaced with NaN plus an explicit indicator. Left as-is it encodes a "
                "1000-year employment history."
            )

    if bureau_path is not None:
        bureau = aggregate_bureau(bureau_path)
        df = df.merge(bureau, on=ID_COLUMN, how="left")
        report.notes.append(f"merged {bureau.shape[1] - 1} aggregated bureau features")

    df["default"] = df[TARGET_COLUMN].astype("Float64")

    # -- splits: random, because no temporal axis exists ---------------------
    rng = np.random.default_rng(seed)
    draws = rng.random(len(df))
    df["split"] = np.where(
        draws < test_fraction,
        "test",
        np.where(draws < test_fraction + valid_fraction, "valid", "train"),
    )
    report.notes.append(
        "Splits are RANDOM, not out-of-time. Home Credit contains no absolute calendar date, so "
        "a temporal split is impossible. Metrics from this adapter are therefore optimistic "
        "relative to deployment and are not comparable with the LendingClub results."
    )

    report.rows = len(df)
    report.default_rate = float(df["default"].mean())
    report.split_counts = df["split"].value_counts().to_dict()
    report.n_features = len(home_credit_features(df)[0]) + len(home_credit_features(df)[1])

    LOG.info(
        "prepared %s Home Credit applications | default rate %.4f | splits %s",
        f"{len(df):,}",
        report.default_rate,
        report.split_counts,
    )
    LOG.warning(
        "Home Credit splits are RANDOM: no calendar date exists in this dataset, so "
        "out-of-time validation is impossible. Results are optimistic."
    )
    return df, report


def home_credit_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` feature columns, excluding identifiers and the target."""
    denied = set(HOME_CREDIT_DENYLIST) | {"default", "split"}
    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in denied
    ]
    categorical = [
        c for c in df.select_dtypes(include=["object", "string", "category"]).columns
        if c not in denied
    ]
    return numeric, categorical


def build_matrix(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    medians: dict[str, float] | None = None,
    levels: dict[str, list[str]] | None = None,
    max_levels: int = 30,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, list[str]]]:
    """Minimal fit/transform for the Home Credit schema.

    A separate, smaller implementation rather than a reuse of
    :class:`credit_risk.features.build.FeatureBuilder`: that class is built around LendingClub's
    named columns and its availability guard is defined against a temporal split, which does not
    exist here. Forcing one class to serve both would make the LendingClub path harder to read for
    the sake of a secondary demonstration.
    """
    out = pd.DataFrame(index=df.index)
    fitted_medians = dict(medians or {})
    fitted_levels = dict(levels or {})

    for column in numeric:
        series = pd.to_numeric(df[column], errors="coerce")
        if column not in fitted_medians:
            fitted_medians[column] = float(series.median()) if series.notna().any() else 0.0
        out[column] = series.fillna(fitted_medians[column]).astype("float32")

    for column in categorical:
        values = df[column].astype("string").fillna("__missing__")
        if column not in fitted_levels:
            keep = values.value_counts().head(max_levels).index.tolist()
            if "__missing__" not in keep:
                keep.append("__missing__")
            keep.append("__other__")
            fitted_levels[column] = keep
        allowed = fitted_levels[column]
        out[column] = pd.Categorical(values.where(values.isin(allowed), "__other__"),
                                     categories=allowed)

    return out, fitted_medians, fitted_levels


def generate_synthetic_home_credit(n_rows: int = 5_000, seed: int = 0) -> pd.DataFrame:
    """A tiny Home-Credit-shaped frame, for smoke-testing the adapter without the download."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=n_rows)
    frame = pd.DataFrame(
        {
            ID_COLUMN: np.arange(100_000, 100_000 + n_rows),
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n_rows,
                                             p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["M", "F", "XNA"], n_rows, p=[0.34, 0.65, 0.01]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n_rows),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n_rows),
            "CNT_CHILDREN": rng.poisson(0.4, n_rows),
            "AMT_INCOME_TOTAL": np.round(np.exp(11.9 - 0.1 * latent + rng.normal(0, 0.5, n_rows))),
            "AMT_CREDIT": np.round(np.exp(13.1 + 0.1 * latent + rng.normal(0, 0.5, n_rows))),
            "AMT_ANNUITY": np.round(np.exp(10.2 + rng.normal(0, 0.4, n_rows))),
            "DAYS_BIRTH": -rng.integers(7_500, 25_000, n_rows),
            "DAYS_EMPLOYED": np.where(
                rng.random(n_rows) < 0.18,
                DAYS_EMPLOYED_SENTINEL,  # the sentinel, on purpose
                -rng.integers(30, 15_000, n_rows),
            ),
            "DAYS_REGISTRATION": -rng.integers(100, 20_000, n_rows),
            "DAYS_ID_PUBLISH": -rng.integers(50, 6_000, n_rows),
            "EXT_SOURCE_1": np.clip(0.5 - 0.15 * latent + rng.normal(0, 0.15, n_rows), 0, 1),
            "EXT_SOURCE_2": np.clip(0.5 - 0.18 * latent + rng.normal(0, 0.15, n_rows), 0, 1),
            "EXT_SOURCE_3": np.clip(0.5 - 0.16 * latent + rng.normal(0, 0.15, n_rows), 0, 1),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n_rows),
            "NAME_EDUCATION_TYPE": rng.choice(
                ["Secondary / secondary special", "Higher education"], n_rows
            ),
            "ORGANIZATION_TYPE": rng.choice(
                ["Business Entity Type 3", "XNA", "Self-employed", "Other"], n_rows
            ),
            "REGION_RATING_CLIENT": rng.integers(1, 4, n_rows),
        }
    )
    logit = -2.4 + 0.85 * latent
    frame[TARGET_COLUMN] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return frame
