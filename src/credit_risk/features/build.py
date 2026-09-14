"""Fitted feature construction.

Everything that learns anything from data lives here and is fitted on ``train`` only. The builder
is stateful on purpose: ``fit`` records categorical vocabularies, imputation values and clip bounds,
and ``transform`` applies them unchanged to validation, test and production traffic. Refitting on
a later window would quietly rewrite history.

Two controls run on every build:

* :func:`credit_risk.data.schema.assert_no_leakage` raises if a post-origination column has
  reached the matrix.
* The **availability guard** drops any feature whose *missingness* moves materially between
  train and validation. LendingClub introduced bureau fields partway through its history, so
  "is this column populated?" is partly a statement about the origination date. A model will
  exploit that happily and it will not reproduce in production. The guard is measured against
  the validation window, never the test window -- looking at test missingness to make a
  modelling decision would spend the out-of-time set.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.data import schema
from credit_risk.features.financial import add_derived_features, derived_feature_names
from credit_risk.utils.runtime import get_logger

LOG = get_logger("features.build")

#: Sentinel for "no such event has ever been recorded" in months-since fields. Paired with an
#: explicit indicator column so a model can separate "never happened" from "happened long ago",
#: which mean-imputation destroys.
MONTHS_SINCE_SENTINEL = 999.0


@dataclass
class FeatureSpec:
    """Declarative description of a feature set."""

    feature_set: str = "primary"
    include_vintage_sensitive: bool = True
    include_late_additions: bool = False
    include_joint: bool = False
    include_derived: bool = True
    #: Graph features are appended by the graph layer; named here so the builder can accept them.
    extra_numeric: tuple[str, ...] = ()
    #: Missingness shift (train -> valid, absolute) above which a feature is dropped.
    availability_shift_tolerance: float = 0.05
    #: Categorical levels kept; the remainder collapse to "__other__".
    max_categorical_levels: int = 40
    #: Winsorisation applied to numeric features, as train-window quantiles.
    clip_quantiles: tuple[float, float] = (0.001, 0.999)

    def raw_columns(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return schema.feature_columns(
            feature_set=self.feature_set,
            include_vintage_sensitive=self.include_vintage_sensitive,
            include_late_additions=self.include_late_additions,
            include_joint=self.include_joint,
        )


@dataclass
class FittedState:
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    indicator_features: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)


class FeatureBuilder:
    """Fit on train, transform everywhere.

    ``installment`` deserves a note. It is knowable at decision time, but it is an invertible
    function of ``loan_amnt``, ``int_rate`` and ``term`` -- given the other three you can recover
    the interest rate to within rounding, and the interest rate *is* LendingClub's risk grade.
    So including ``installment`` in the ``primary`` feature set would smuggle the incumbent
    scorecard into a model that claims not to use it. It is therefore excluded from ``primary``
    and available to the economics layer, which legitimately needs it.
    """

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        self.state = FittedState()
        self._fitted = False

    # -- fitting ----------------------------------------------------------
    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> FeatureBuilder:
        numeric_raw, categorical_raw = self.spec.raw_columns()

        frame = self._pre_transform(train)
        numeric = [c for c in numeric_raw if c in frame.columns]
        categorical = [c for c in categorical_raw if c in frame.columns]

        if self.spec.include_derived:
            numeric += [c for c in derived_feature_names() if c in frame.columns]
        numeric += [c for c in self.spec.extra_numeric if c in frame.columns]
        numeric = list(dict.fromkeys(numeric))

        # `installment` is excluded from `primary` because the interest rate is recoverable
        # from it; see the class docstring.
        if self.spec.feature_set == "primary":
            for banned in ("installment", "est_payment_to_income", "dti_post_loan"):
                if banned in numeric:
                    numeric.remove(banned)
                    self.state.dropped[banned] = (
                        "excluded from 'primary': interest rate is recoverable from instalment"
                    )

        # Near-constant columns carry no information and destabilise the linear baseline.
        for col in list(numeric):
            series = pd.to_numeric(frame[col], errors="coerce")
            if series.notna().sum() == 0:
                numeric.remove(col)
                self.state.dropped[col] = "entirely missing in the training window"
            elif series.nunique(dropna=True) <= 1:
                numeric.remove(col)
                self.state.dropped[col] = "constant in the training window"

        # -- availability guard, measured against validation ---------------
        if valid is not None and len(valid):
            valid_frame = self._pre_transform(valid)
            for col in list(numeric):
                if col not in valid_frame.columns:
                    continue
                shift = float(valid_frame[col].isna().mean() - frame[col].isna().mean())
                if abs(shift) > self.spec.availability_shift_tolerance:
                    numeric.remove(col)
                    self.state.dropped[col] = (
                        f"availability guard: missingness moves {shift:+.1%} from train to "
                        f"valid, so the column partly encodes the vintage"
                    )

        # -- fitted statistics ---------------------------------------------
        for col in numeric:
            series = pd.to_numeric(frame[col], errors="coerce")
            lo, hi = series.quantile(self.spec.clip_quantiles).tolist()
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                lo, hi = float(series.min()), float(series.max())
            self.state.clip_bounds[col] = (float(lo), float(hi))
            self.state.numeric_medians[col] = float(series.clip(lo, hi).median())

        for col in categorical:
            counts = frame[col].astype("string").fillna("__missing__").value_counts()
            levels = counts.head(self.spec.max_categorical_levels).index.tolist()
            if "__missing__" not in levels:
                levels.append("__missing__")
            levels.append("__other__")
            self.state.categorical_levels[col] = levels

        indicators = [
            f"{col}_missing"
            for col in schema.INFORMATIVE_MISSING_NUMERIC
            if col in numeric
        ]

        self.state.numeric_features = numeric
        self.state.categorical_features = categorical
        self.state.indicator_features = indicators
        self._fitted = True

        LOG.info(
            "fitted feature builder [%s]: %d numeric, %d categorical, %d indicators, %d dropped",
            self.spec.feature_set,
            len(numeric),
            len(categorical),
            len(indicators),
            len(self.state.dropped),
        )
        for col, reason in self.state.dropped.items():
            LOG.debug("dropped %s: %s", col, reason)
        return self

    # -- transforming -----------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.transform called before fit")

        frame = self._pre_transform(df)
        out = pd.DataFrame(index=frame.index)

        for col in self.state.numeric_features:
            series = pd.to_numeric(frame.get(col), errors="coerce")
            if series is None:
                series = pd.Series(np.nan, index=frame.index)
            lo, hi = self.state.clip_bounds[col]
            series = series.clip(lo, hi)
            if col in schema.INFORMATIVE_MISSING_NUMERIC:
                # "Never happened" is information. Flag it, then push the value to a sentinel
                # far beyond any observed month count rather than to the median.
                out[f"{col}_missing"] = series.isna().astype("int8")
                series = series.fillna(min(MONTHS_SINCE_SENTINEL, hi if hi > 0 else 999.0))
            else:
                series = series.fillna(self.state.numeric_medians[col])
            out[col] = series.astype("float32")

        for col in self.state.categorical_features:
            levels = self.state.categorical_levels[col]
            values = frame.get(col)
            values = (
                pd.Series(pd.NA, index=frame.index, dtype="string")
                if values is None
                else values.astype("string")
            )
            values = values.fillna("__missing__")
            values = values.where(values.isin(levels), "__other__")
            out[col] = pd.Categorical(values, categories=levels)

        ordered = (
            self.state.numeric_features
            + [c for c in self.state.indicator_features if c in out.columns]
            + self.state.categorical_features
        )
        out = out[[c for c in ordered if c in out.columns]]

        schema.assert_no_leakage(list(out.columns))
        return out

    def fit_transform(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> pd.DataFrame:
        return self.fit(train, valid).transform(train)

    # -- helpers ----------------------------------------------------------
    def _pre_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return add_derived_features(df) if self.spec.include_derived else df.copy()

    @property
    def feature_names(self) -> list[str]:
        return (
            self.state.numeric_features
            + self.state.indicator_features
            + self.state.categorical_features
        )

    @property
    def categorical_features(self) -> list[str]:
        return list(self.state.categorical_features)

    def describe_dropped(self) -> pd.DataFrame:
        return pd.DataFrame(
            sorted(self.state.dropped.items()), columns=["feature", "reason"]
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"spec": self.spec, "state": self.state}, fh)
        # A human-readable sidecar, so the fitted vocabulary can be inspected without unpickling.
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "feature_set": self.spec.feature_set,
                    "n_numeric": len(self.state.numeric_features),
                    "n_categorical": len(self.state.categorical_features),
                    "features": self.feature_names,
                    "dropped": self.state.dropped,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureBuilder:
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        builder = cls(payload["spec"])
        builder.state = payload["state"]
        builder._fitted = True
        return builder


def to_numeric_matrix(
    X: pd.DataFrame,
    categorical_features: list[str] | None = None,
    fitted_levels: dict[str, list[str]] | None = None,
    standardise: bool = True,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Flatten a mixed-dtype feature frame into a dense float32 matrix.

    Needed by the GNN, which aggregates numeric vectors across neighbours and so cannot consume
    pandas categoricals the way LightGBM does. Categoricals are one-hot encoded against the
    *fitted* level list so that train, validation and test always produce identically-shaped and
    identically-ordered columns.

    Standardisation statistics must be passed in when transforming anything other than the
    training window -- recomputing them per split would let each split's own distribution leak
    into its own features.

    Returns ``(matrix, column_names, means, stds)``.
    """
    categorical_features = categorical_features or []
    blocks: list[np.ndarray] = []
    names: list[str] = []

    numeric_cols = [c for c in X.columns if c not in categorical_features]
    if numeric_cols:
        blocks.append(X[numeric_cols].to_numpy(dtype=np.float32))
        names.extend(numeric_cols)

    for col in categorical_features:
        if col not in X.columns:
            continue
        levels = (
            fitted_levels[col]
            if fitted_levels and col in fitted_levels
            else list(pd.Categorical(X[col]).categories)
        )
        values = X[col].astype("string").fillna("__missing__")
        onehot = np.zeros((len(X), len(levels)), dtype=np.float32)
        lookup = {level: i for i, level in enumerate(levels)}
        indices = values.map(lookup)
        valid = indices.notna().to_numpy()
        onehot[np.flatnonzero(valid), indices[valid].to_numpy(dtype=int)] = 1.0
        blocks.append(onehot)
        names.extend([f"{col}={level}" for level in levels])

    matrix = np.hstack(blocks) if blocks else np.zeros((len(X), 0), dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if standardise:
        if means is None or stds is None:
            means = matrix.mean(axis=0)
            stds = matrix.std(axis=0)
        safe = np.where(stds > 1e-6, stds, 1.0)
        matrix = ((matrix - means) / safe).astype(np.float32)
    else:
        means = np.zeros(matrix.shape[1], dtype=np.float32) if means is None else means
        stds = np.ones(matrix.shape[1], dtype=np.float32) if stds is None else stds

    return matrix, names, np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)
