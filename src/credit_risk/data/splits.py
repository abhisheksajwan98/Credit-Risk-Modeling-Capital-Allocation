"""Out-of-time split assignment.

Every split in this project is defined by a contiguous window of ``issue_d``. There is no random
splitting anywhere, and no function here accepts a ``shuffle`` argument.

Why it has to be this way
-------------------------
Credit performance moves with the vintage and the macro cycle. If you split at random, a model
sees loans issued in the same month as the ones it is scored on, and it silently exploits
information about that month's credit conditions that it could not have had at decision time.
The result is an AUC that looks excellent and a model that degrades the moment it is deployed.
The out-of-time window is the only estimate of deployment performance worth quoting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SplitName = Literal["history", "train", "valid", "test", "monitor", "excluded"]

SPLIT_ORDER: tuple[str, ...] = ("history", "train", "valid", "test", "monitor")


@dataclass(frozen=True)
class SplitWindows:
    """Inclusive month windows, expressed as ``YYYY-MM`` strings.

    ``history`` rows are never modelled. They exist so that loans issued at the very start of the
    training window still have past neighbours available for the graph.

    ``monitor`` rows have no mature outcome by the data cutoff and carry no label. That is
    deliberate rather than a shortcoming: it is exactly the production situation, where features
    and predictions are available immediately and outcomes only years later.
    """

    history: tuple[str, str] = ("2007-06", "2009-12")
    train: tuple[str, str] = ("2010-01", "2014-06")
    valid: tuple[str, str] = ("2014-07", "2014-12")
    test: tuple[str, str] = ("2015-01", "2015-12")
    monitor: tuple[str, str] = ("2016-01", "2018-12")

    @classmethod
    def from_config(cls, cfg) -> SplitWindows:
        def win(name: str, default: tuple[str, str]) -> tuple[str, str]:
            node = cfg.get(name) if hasattr(cfg, "get") else None
            if node is None:
                return default
            return (str(node["start"]), str(node["end"]))

        return cls(
            history=win("history", cls.history),
            train=win("train", cls.train),
            valid=win("valid", cls.valid),
            test=win("test", cls.test),
            monitor=win("monitor", cls.monitor),
        )

    def as_periods(self) -> dict[str, tuple[pd.Period, pd.Period]]:
        out = {}
        for name in SPLIT_ORDER:
            start, end = getattr(self, name)
            out[name] = (pd.Period(start, freq="M"), pd.Period(end, freq="M"))
        return out

    def validate(self) -> None:
        """Assert the windows are ordered, non-overlapping and contiguous in intent."""
        periods = self.as_periods()
        previous_end: pd.Period | None = None
        previous_name = ""
        for name in SPLIT_ORDER:
            start, end = periods[name]
            if start > end:
                raise ValueError(f"split {name!r} has start {start} after end {end}")
            if previous_end is not None and start <= previous_end:
                raise ValueError(
                    f"split {name!r} starts at {start}, which overlaps {previous_name!r} "
                    f"ending at {previous_end}. Overlapping windows leak future information."
                )
            previous_end, previous_name = end, name

    @property
    def modelled_window(self) -> tuple[pd.Period, pd.Period]:
        """The full labelled range: start of train through end of test."""
        periods = self.as_periods()
        return periods["train"][0], periods["test"][1]

    @property
    def labelled_splits(self) -> tuple[str, ...]:
        return ("train", "valid", "test")


def assign_split(issue_period: pd.Series, windows: SplitWindows) -> pd.Series:
    """Map a Series of monthly Periods to split names.

    Anything outside every window becomes ``"excluded"`` rather than raising, so that the caller
    can count and report what was dropped instead of silently losing rows.
    """
    windows.validate()
    out = pd.Series("excluded", index=issue_period.index, dtype="object")
    for name, (start, end) in windows.as_periods().items():
        mask = (issue_period >= start) & (issue_period <= end)
        out[mask] = name
    return out


def assert_temporal_ordering(df: pd.DataFrame, split_col: str = "split",
                             period_col: str = "issue_period") -> None:
    """Fail if any training row was issued after any test row.

    This is the cheapest possible guard against a split being silently rebuilt as random, and it
    is called from ``tests/test_leakage.py``.
    """
    present = [s for s in SPLIT_ORDER if s in set(df[split_col].unique())]
    bounds: dict[str, tuple] = {}
    for name in present:
        sub = df.loc[df[split_col] == name, period_col]
        if len(sub):
            bounds[name] = (sub.min(), sub.max())

    for earlier, later in zip(present, present[1:], strict=False):
        if earlier not in bounds or later not in bounds:
            continue
        if bounds[earlier][1] >= bounds[later][0]:
            raise AssertionError(
                f"temporal ordering violated: {earlier} ends at {bounds[earlier][1]} but "
                f"{later} starts at {bounds[later][0]}. Splits must not overlap in time."
            )
