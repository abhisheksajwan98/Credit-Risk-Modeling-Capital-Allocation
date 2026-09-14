"""Raw LendingClub CSV -> a clean, split-assigned modelling table.

This stage is deterministic and does no fitting. Anything that learns from data (encoders,
imputation statistics, cohort rates, calibration maps) belongs downstream, where it can be fitted
on ``train`` only. Keeping that boundary sharp is what makes the leakage guarantees checkable.

Output: ``data/processed/loans.parquet`` plus a measured profile in
``results/reports/data_profile.md``. The profile is written from the data rather than from
assumptions, because every "known fact" about this dataset that circulates online is worth
re-checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.data import schema
from credit_risk.data.splits import SplitWindows, assign_split
from credit_risk.utils.runtime import get_logger

LOG = get_logger("data.prepare")

#: LendingClub writes months as ``Dec-2015``.
_LC_DATE_FORMAT = "%b-%Y"

#: Months between the last payment received and the charge-off being booked. LendingClub charges
#: off at 150 days delinquent, so a defaulted loan's outcome becomes *known* roughly five months
#: after its final payment. Used to date when a neighbour's label was observable -- see
#: docs/GRAPH_DESIGN.md. Not a feature; it only ever gates information availability.
CHARGE_OFF_LAG_MONTHS = 5


@dataclass
class PrepareReport:
    """Counts collected while preparing, written out so drops are visible rather than silent."""

    rows_read: int = 0
    rows_after_term_filter: int = 0
    rows_after_window_filter: int = 0
    rows_dropped_non_terminal: int = 0
    rows_dropped_unmapped_status: int = 0
    rows_dropped_bad_id: int = 0
    split_counts: dict[str, int] = field(default_factory=dict)
    default_rate_by_split: dict[str, float] = field(default_factory=dict)
    default_rate_by_vintage: dict[str, float] = field(default_factory=dict)
    missingness: dict[str, float] = field(default_factory=dict)
    availability_shift: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows_read": self.rows_read,
            "rows_after_term_filter": self.rows_after_term_filter,
            "rows_after_window_filter": self.rows_after_window_filter,
            "rows_dropped_non_terminal": self.rows_dropped_non_terminal,
            "rows_dropped_unmapped_status": self.rows_dropped_unmapped_status,
            "rows_dropped_bad_id": self.rows_dropped_bad_id,
            "split_counts": self.split_counts,
            "default_rate_by_split": self.default_rate_by_split,
            "default_rate_by_vintage": self.default_rate_by_vintage,
            "missingness_top": dict(
                sorted(self.missingness.items(), key=lambda kv: -kv[1])[:40]
            ),
            "availability_shift_top": dict(
                sorted(self.availability_shift.items(), key=lambda kv: -abs(kv[1]))[:25]
            ),
            "notes": self.notes,
        }


def parse_lc_month(series: pd.Series) -> pd.Series:
    """Parse ``Dec-2015``-style strings into monthly Periods, tolerating nulls and stray formats."""
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, format=_LC_DATE_FORMAT, errors="coerce")
    # A minority of mirrored files use ISO-like values; catch those without silently mangling.
    unparsed = dt.isna() & s.notna()
    if unparsed.any():
        dt.loc[unparsed] = pd.to_datetime(s[unparsed], errors="coerce")
    return dt.dt.to_period("M")


def normalise_employment(series: pd.Series) -> pd.Series:
    """Lower-case, strip punctuation and collapse whitespace in the free-text employment field.

    Row-wise and deterministic, so it is safe here. Reducing the result to a bounded vocabulary
    *is* a fitted operation and therefore happens downstream on ``train`` only.

    Caveat carried through the whole project: LendingClub's ``emp_title`` holds **employer name
    before 2013-09-23 and job title after**, so this column changes meaning mid-history. It is
    used only as a secondary graph relation and is ablated separately.
    """
    s = series.astype("string").str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9\s&/-]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace({"": pd.NA, "n/a": pd.NA, "na": pd.NA, "none": pd.NA})


def _term_months(series: pd.Series) -> pd.Series:
    """``" 36 months"`` -> ``36``."""
    return (
        series.astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype("Float64")
        .astype("Int16")
    )


def _emp_length_years(series: pd.Series) -> pd.Series:
    """``"10+ years"`` -> 10, ``"< 1 year"`` -> 0, ``"n/a"`` -> NA. Ordinal, so keep it numeric."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NA, index=series.index, dtype="Float64")
    out[s.str.startswith("10+", na=False)] = 10.0
    out[s.str.startswith("<", na=False)] = 0.0
    plain = s.str.extract(r"^(\d+)\s+years?$", expand=False)
    out[plain.notna()] = plain[plain.notna()].astype(float)
    return out


def _read_raw(
    path: Path, usecols: list[str] | None, chunksize: int, term_filter: int | None
) -> tuple[pd.DataFrame, PrepareReport]:
    """Read the (large, gzipped) accepted-loans CSV in chunks, filtering as we go.

    The real file is ~2.26M rows by 151 columns. Filtering to a single term inside the chunk loop
    keeps peak memory around a gigabyte instead of several.
    """
    report = PrepareReport()
    frames: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        chunksize=chunksize,
        low_memory=False,
        usecols=usecols,
        na_values=["", "n/a", "N/A", "null", "NULL"],
        keep_default_na=True,
    )
    for chunk in reader:
        report.rows_read += len(chunk)
        # LendingClub's original exports carried trailing summary rows with a null id.
        bad_id = chunk["id"].isna() if "id" in chunk.columns else pd.Series(False, index=chunk.index)
        if "id" in chunk.columns:
            numeric_id = pd.to_numeric(chunk["id"], errors="coerce")
            bad_id = bad_id | numeric_id.isna()
        if bad_id.any():
            report.rows_dropped_bad_id += int(bad_id.sum())
            chunk = chunk.loc[~bad_id]
        if term_filter is not None and "term" in chunk.columns:
            chunk = chunk.loc[_term_months(chunk["term"]) == term_filter]
        if len(chunk):
            frames.append(chunk)

    if not frames:
        raise ValueError(f"no rows survived reading {path}; check the term filter and file format")
    df = pd.concat(frames, ignore_index=True)
    report.rows_after_term_filter = len(df)
    LOG.info(
        "read %s rows, %s remain after term filter", f"{report.rows_read:,}", f"{len(df):,}"
    )
    return df, report


def prepare_loans(
    raw_path: str | Path,
    windows: SplitWindows | None = None,
    term_months: int | None = 36,
    chunksize: int = 250_000,
    usecols: list[str] | None = None,
) -> tuple[pd.DataFrame, PrepareReport]:
    """Produce the modelling table.

    Steps, in order, each of which is a decision documented in ``docs/DATASET.md``:

    1. Read and drop malformed rows.
    2. Restrict to a single loan term, so every loan has the same outcome horizon.
    3. Parse dates; derive ``issue_period``.
    4. Assign splits from ``issue_period`` alone.
    5. Map ``loan_status`` to a binary target on labelled splits; drop non-terminal rows there.
    6. Derive relational keys and the outcome-observation date.
    """
    windows = windows or SplitWindows()
    windows.validate()
    raw_path = Path(raw_path)

    df, report = _read_raw(raw_path, usecols, chunksize, term_months)

    # --- dates ---------------------------------------------------------------
    for col in schema.DATE_COLUMNS:
        if col in df.columns:
            df[col] = parse_lc_month(df[col])
    if "issue_d" not in df.columns:
        raise ValueError("issue_d is required: it is the decision timestamp for every split")
    df = df.loc[df["issue_d"].notna()].copy()
    df["issue_period"] = df["issue_d"]

    # --- splits, assigned from the decision date and nothing else ------------
    df["split"] = assign_split(df["issue_period"], windows)
    df = df.loc[df["split"] != "excluded"].copy()
    report.rows_after_window_filter = len(df)

    # --- target --------------------------------------------------------------
    status = df[schema.TARGET_SOURCE_COLUMN].astype("string").str.strip()
    mapped = status.map(schema.LC_STATUS_MAP)
    labelled = df["split"].isin(windows.labelled_splits)

    non_terminal = labelled & status.isin(schema.NON_TERMINAL_STATUSES)
    unmapped = labelled & mapped.isna() & ~non_terminal
    report.rows_dropped_non_terminal = int(non_terminal.sum())
    report.rows_dropped_unmapped_status = int(unmapped.sum())
    if report.rows_dropped_unmapped_status:
        report.notes.append(
            "Unmapped loan_status values inside the matured window: "
            + ", ".join(sorted(status[unmapped].dropna().unique().tolist())[:10])
        )

    drop_mask = non_terminal | unmapped
    if drop_mask.any():
        share = 100.0 * drop_mask.sum() / max(labelled.sum(), 1)
        msg = (
            f"dropped {drop_mask.sum():,} labelled rows ({share:.2f}%) with a non-terminal or "
            f"unmapped status inside a window that should be fully matured"
        )
        LOG.warning(msg)
        report.notes.append(msg)
        if share > 0.5:
            report.notes.append(
                "WARNING: this exceeds the 0.5% tolerance in docs/DATASET.md section 4. "
                "The maturity assumption should be revisited rather than accepted."
            )
        df = df.loc[~drop_mask].copy()
        mapped = mapped.loc[df.index]

    # Two distinct columns, because they answer two distinct questions.
    #
    # `default_observed` — did this loan resolve, and how? Populated for every row with a terminal
    # status, including the pre-2010 history window and any monitoring-window loan that has
    # already run off. This is what a later underwriter could legitimately have seen, so it is
    # what the graph's neighbour-label aggregates draw on.
    #
    # `default` — the modelling target. Populated only on train/valid/test, where the maturity
    # argument in docs/DATASET.md section 4 actually holds. Monitoring-window rows are excluded
    # even when they happen to have resolved, because keeping only the ones that resolved early
    # would select for fast defaults and bias the rate upward.
    df["default_observed"] = mapped.astype("Float64")
    df[schema.TARGET_COLUMN] = mapped.astype("Float64")
    df.loc[~df["split"].isin(windows.labelled_splits), schema.TARGET_COLUMN] = pd.NA

    # --- derived typed columns ----------------------------------------------
    if "term" in df.columns:
        df["term_months"] = _term_months(df["term"])
    if "emp_length" in df.columns:
        df["emp_length_years"] = _emp_length_years(df["emp_length"])

    # --- relational keys -----------------------------------------------------
    if "zip_code" in df.columns:
        df["zip3"] = (
            df["zip_code"].astype("string").str.strip().str.slice(0, 3).replace({"": pd.NA})
        )
    if "emp_title" in df.columns:
        df["emp_norm"] = normalise_employment(df["emp_title"])
        # Flag the semantics break so downstream code can ablate around it.
        df["emp_title_is_employer_name"] = df["issue_period"] < pd.Period("2013-10", freq="M")

    # --- when did the outcome become observable? -----------------------------
    df["outcome_observed_period"] = _outcome_observed_period(df)

    df = df.sort_values("issue_period", kind="mergesort").reset_index(drop=True)
    df["loan_idx"] = np.arange(len(df), dtype=np.int64)

    _fill_report(df, report, windows)
    LOG.info(
        "prepared %s loans | splits: %s",
        f"{len(df):,}",
        {k: f"{v:,}" for k, v in report.split_counts.items()},
    )
    return df, report


def _outcome_observed_period(df: pd.DataFrame) -> pd.Series:
    """The month by which each loan's outcome was knowable to an observer.

    A loan that was fully paid is resolved when its last payment arrived. A charged-off loan is
    resolved roughly :data:`CHARGE_OFF_LAG_MONTHS` after its last payment, since LendingClub books
    the charge-off at 150 days delinquent.

    This column exists for one purpose: to decide whether a *neighbour's label* was available at
    the time of another borrower's application. It is never a feature, and it is derived from
    post-origination fields, so it is confined to graph construction.

    A loan that has **not** reached a terminal status by the data cutoff has no observation date
    at all -- its outcome is still unknown. Leaving this as NA for those rows is what stops an
    unresolved loan from being counted as a "good" neighbour simply because it has not defaulted
    yet, which would bias every cohort default rate downward.
    """
    if "last_pymnt_d" not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="period[M]")

    resolved = df["default_observed"].notna() if "default_observed" in df.columns else None
    if resolved is None:
        resolved = pd.Series(True, index=df.index)

    is_default = df.get("default_observed")
    lag = pd.Series(0, index=df.index, dtype="int64")
    if is_default is not None:
        lag = pd.Series(
            np.where(is_default.fillna(0).astype(float) > 0.5, CHARGE_OFF_LAG_MONTHS, 0),
            index=df.index,
        )

    base = df["last_pymnt_d"].copy()
    # Fall back to contractual maturity where the payment date is missing but the loan resolved.
    if "term_months" in df.columns:
        contractual = df["issue_period"] + df["term_months"].fillna(36).astype(int)
        base = base.fillna(contractual)

    valid = base.notna() & resolved
    out = pd.Series(pd.NaT, index=df.index, dtype="period[M]")
    if valid.any():
        shifted = pd.PeriodIndex(
            [max(p + int(l), i + 1) for p, l, i in zip(base[valid], lag[valid], df.loc[valid, "issue_period"], strict=True)], freq="M"
        )
        out.loc[valid] = shifted
    return out


def _fill_report(df: pd.DataFrame, report: PrepareReport, windows: SplitWindows) -> None:
    report.split_counts = df["split"].value_counts().to_dict()

    target = df[schema.TARGET_COLUMN]
    for split in windows.labelled_splits:
        sub = target[df["split"] == split].dropna()
        if len(sub):
            report.default_rate_by_split[split] = float(sub.mean())

    vintage = df["issue_period"].dt.year if hasattr(df["issue_period"], "dt") else None
    if vintage is not None:
        by_year = target.groupby(vintage).mean(numeric_only=False)
        report.default_rate_by_vintage = {
            str(int(k)): float(v) for k, v in by_year.dropna().items()
        }

    report.missingness = {
        col: float(df[col].isna().mean()) for col in df.columns if df[col].isna().any()
    }

    # Availability shift: how much a column's missingness changes between the training window and
    # the out-of-time window. A large shift means the column partly encodes the vintage, which a
    # model will happily exploit and which will not reproduce in production.
    tr = df["split"] == "train"
    te = df["split"] == "test"
    if tr.any() and te.any():
        for col in df.columns:
            if df[col].dtype.kind in "OSU" and col not in schema.VINTAGE_SENSITIVE_NUMERIC:
                continue
            shift = float(df.loc[te, col].isna().mean() - df.loc[tr, col].isna().mean())
            if abs(shift) > 0.02:
                report.availability_shift[col] = shift


def write_processed(
    df: pd.DataFrame, out_dir: str | Path, filename: str = "loans.parquet"
) -> Path:
    """Persist the modelling table. Period columns are stored as strings for parquet portability."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename

    out = df.copy()
    if "id" in out.columns:
        out["id"] = out["id"].astype("string")
    for col in out.columns:
        if isinstance(out[col].dtype, pd.PeriodDtype):
            out[col] = out[col].astype("string")
    out.to_parquet(path, index=False, compression="snappy")
    LOG.info("wrote %s (%.1f MB)", path, path.stat().st_size / 1024**2)
    return path


def read_processed(path: str | Path, period_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load the modelling table, restoring Period dtypes for the given columns."""
    df = pd.read_parquet(path)
    default_periods = (
        "issue_period",
        "outcome_observed_period",
        "issue_d",
        "earliest_cr_line",
        "last_pymnt_d",
        "last_credit_pull_d",
        "next_pymnt_d",
    )
    for col in period_columns or default_periods:
        if col in df.columns and df[col].dtype.kind in "OSU":
            df[col] = pd.PeriodIndex(df[col].astype("string").fillna(pd.NA), freq="M")
    return df
