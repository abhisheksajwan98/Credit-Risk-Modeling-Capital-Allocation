"""Logistic regression baseline.

This is the reference every other model is judged against, and it is not a formality. Regulated
lenders still run scorecards built on penalised logistic regression because the coefficients are
inspectable, the model extrapolates predictably, and an adverse-action reason can be derived from
it directly. If a gradient-boosted model cannot beat this by a margin that survives the out-of-time
window, the extra complexity is not worth defending.

Two properties are worth noting for the calibration experiment: logistic regression trained by
maximum likelihood on the natural class balance is *already close to calibrated* on the training
distribution, because the log-loss it minimises is a proper scoring rule. Class re-weighting breaks
that -- it deliberately distorts the base rate -- which is why re-weighting is treated as a
hypothesis to test rather than a default.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.utils.runtime import get_logger

LOG = get_logger("models.baseline")


@dataclass
class LogisticConfig:
    C: float = 0.1
    penalty: str = "l2"
    solver: str = "lbfgs"
    max_iter: int = 2000
    #: ``None`` keeps the natural base rate, which preserves calibration. ``"balanced"`` is
    #: available so EXP01 can measure what re-weighting actually costs.
    class_weight: str | None = None
    tol: float = 1e-4


class LogisticBaseline:
    """Scale + one-hot + penalised logistic regression."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        config: LogisticConfig | None = None,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.config = config or LogisticConfig()
        self.pipeline: Pipeline | None = None

    def _build(self) -> Pipeline:
        cfg = self.config
        transformers = []
        if self.numeric_features:
            # The feature builder has already imputed and winsorised; scaling is what the
            # lbfgs solver needs to converge and what makes coefficients comparable.
            transformers.append(("numeric", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",  # avoid the dummy trap under an intercept
                        sparse_output=False,
                        min_frequency=None,
                    ),
                    self.categorical_features,
                )
            )
        return Pipeline(
            [
                ("prep", ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)),
                (
                    "model",
                    LogisticRegression(
                        # `penalty` is not passed: L2 is sklearn's default and naming it
                        # explicitly triggers a deprecation in scikit-learn >= 1.8.
                        C=cfg.C,
                        solver=cfg.solver,
                        max_iter=cfg.max_iter,
                        class_weight=cfg.class_weight,
                        tol=cfg.tol,
                        n_jobs=None,
                    ),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> LogisticBaseline:
        self.pipeline = self._build()
        # One-hot needs plain strings, not pandas Categorical with unused levels.
        self.pipeline.fit(self._prepare(X), np.asarray(y).astype(int))
        LOG.info(
            "logistic baseline fitted on %s rows, %d expanded columns",
            f"{len(X):,}",
            self.pipeline.named_steps["prep"].transform(self._prepare(X.head(2))).shape[1],
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LogisticBaseline.predict_proba called before fit")
        return self.pipeline.predict_proba(self._prepare(X))[:, 1]

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.categorical_features:
            if col in out.columns:
                out[col] = out[col].astype("string").fillna("__missing__")
        return out

    def coefficients(self) -> pd.DataFrame:
        """Standardised coefficients, largest absolute effect first.

        Because the numeric block is standardised, these are directly comparable: a coefficient of
        0.4 means one standard deviation of that feature moves the log-odds of default by 0.4.
        """
        if self.pipeline is None:
            raise RuntimeError("coefficients requested before fit")
        names = self.pipeline.named_steps["prep"].get_feature_names_out()
        coefs = self.pipeline.named_steps["model"].coef_[0]
        return (
            pd.DataFrame({"feature": names, "coefficient": coefs})
            .assign(abs_coefficient=lambda d: d["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> LogisticBaseline:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
