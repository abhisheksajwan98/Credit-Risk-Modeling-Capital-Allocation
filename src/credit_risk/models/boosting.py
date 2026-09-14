"""LightGBM model.

One boosting framework, not two. Running XGBoost alongside would produce a near-identical number
and double the code that has to be explained; where the two differ it is down to defaults, not to
anything about credit risk. LightGBM is chosen for native categorical handling (no one-hot
explosion on `addr_state` and `sub_grade`) and for fast CPU histogram training.

Hyper-parameter search is deliberately small: a random search over a handful of regularisation
knobs, scored on the validation window. The objective of this project is a defensible model, not a
leaderboard position, and a 400-trial sweep against a validation set is mostly a machine for
overfitting that set.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.utils.runtime import default_n_jobs, get_logger

LOG = get_logger("models.boosting")


@dataclass
class BoostingConfig:
    objective: str = "binary"
    #: Optimising log-loss rather than AUC keeps the raw scores closer to probabilities, which
    #: matters because Layer C consumes them. Calibration still follows; this just starts closer.
    metric: str = "binary_logloss"
    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 200
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    n_estimators: int = 3000
    early_stopping_rounds: int = 100
    #: ``None`` preserves the base rate and therefore the calibration. Set to ``"balanced"``
    #: only to measure the cost in EXP01.
    class_weight: str | None = None
    #: -1 means every core. `default_n_jobs()` honours the thread cap instead, so a run
    #: launched with --max-threads does not quietly claim the whole machine anyway.
    n_jobs: int = field(default_factory=default_n_jobs)
    verbose: int = -1

    def lgb_params(self, seed: int) -> dict[str, Any]:
        params = asdict(self)
        for key in ("early_stopping_rounds", "class_weight"):
            params.pop(key, None)
        params["random_state"] = seed
        return params


@dataclass
class SearchSpace:
    """Small random search over the knobs that actually matter at this sample size."""

    num_leaves: tuple[int, ...] = (15, 31, 63, 127)
    min_child_samples: tuple[int, ...] = (50, 100, 200, 500)
    learning_rate: tuple[float, ...] = (0.02, 0.03, 0.05)
    colsample_bytree: tuple[float, ...] = (0.6, 0.8, 1.0)
    subsample: tuple[float, ...] = (0.7, 0.85, 1.0)
    reg_lambda: tuple[float, ...] = (0.0, 1.0, 5.0, 20.0)
    n_trials: int = 15

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            "num_leaves": int(rng.choice(self.num_leaves)),
            "min_child_samples": int(rng.choice(self.min_child_samples)),
            "learning_rate": float(rng.choice(self.learning_rate)),
            "colsample_bytree": float(rng.choice(self.colsample_bytree)),
            "subsample": float(rng.choice(self.subsample)),
            "reg_lambda": float(rng.choice(self.reg_lambda)),
        }


@dataclass
class BoostingResult:
    best_iteration: int
    valid_auc: float
    params: dict[str, Any] = field(default_factory=dict)
    search_trials: list[dict[str, Any]] = field(default_factory=list)


class BoostingModel:
    """LightGBM with native categoricals, early stopping and an optional small random search."""

    def __init__(
        self,
        categorical_features: list[str] | None = None,
        config: BoostingConfig | None = None,
    ) -> None:
        self.categorical_features = list(categorical_features or [])
        self.config = config or BoostingConfig()
        self.model: lgb.LGBMClassifier | None = None
        self.result: BoostingResult | None = None

    # -- fitting ----------------------------------------------------------
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame,
        y_valid: pd.Series | np.ndarray,
        seed: int = 42,
        search: SearchSpace | None = None,
    ) -> BoostingModel:
        """Fit with early stopping on the validation window.

        Early stopping is what chooses the number of trees, so the validation window is genuinely
        used for model selection here. That is exactly why it must be a *separate, later* window
        from training and an *earlier* one than test.
        """
        Xtr, Xva = self._prepare(X_train), self._prepare(X_valid)
        ytr, yva = np.asarray(y_train).astype(int), np.asarray(y_valid).astype(int)
        rng = np.random.default_rng(seed)

        trials: list[dict[str, Any]] = []
        best_params: dict[str, Any] = {}
        best_auc, best_iter = -np.inf, self.config.n_estimators

        candidates = [{}] if search is None else [{} ] + [
            search.sample(rng) for _ in range(search.n_trials)
        ]

        for i, override in enumerate(candidates):
            params = self.config.lgb_params(seed) | override
            model = lgb.LGBMClassifier(
                **params, class_weight=self.config.class_weight
            )
            model.fit(
                Xtr,
                ytr,
                eval_set=[(Xva, yva)],
                eval_metric=["auc", "binary_logloss"],
                categorical_feature=self._present_categoricals(Xtr),
                callbacks=[
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            auc = float(roc_auc_score(yva, model.predict_proba(Xva)[:, 1]))
            trials.append({"trial": i, "valid_auc": auc, **override})
            LOG.info(
                "trial %d/%d valid AUC=%.5f best_iter=%s %s",
                i,
                len(candidates) - 1,
                auc,
                model.best_iteration_,
                override or "(defaults)",
            )
            if auc > best_auc:
                best_auc, best_params = auc, override
                best_iter = model.best_iteration_ or self.config.n_estimators
                self.model = model

        self.result = BoostingResult(
            best_iteration=int(best_iter),
            valid_auc=float(best_auc),
            params=self.config.lgb_params(seed) | best_params,
            search_trials=trials,
        )
        LOG.info(
            "selected: valid AUC=%.5f at %d trees %s",
            best_auc,
            best_iter,
            best_params or "(defaults)",
        )
        return self

    def refit_on(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        seed: int = 42,
    ) -> BoostingModel:
        """Refit on a combined window using the already-selected tree count.

        Offered but **not** used in the headline results. Refitting on train+valid before scoring
        test is common practice and usually helps, but it means the reported model is not the one
        the validation metrics describe. Kept available and off by default so the choice is
        explicit rather than accidental.
        """
        if self.result is None:
            raise RuntimeError("refit_on called before fit")
        params = dict(self.result.params)
        params["n_estimators"] = self.result.best_iteration
        model = lgb.LGBMClassifier(**params, class_weight=self.config.class_weight)
        Xp = self._prepare(X)
        model.fit(Xp, np.asarray(y).astype(int),
                  categorical_feature=self._present_categoricals(Xp))
        self.model = model
        return self

    # -- inference --------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("BoostingModel.predict_proba called before fit")
        return self.model.predict_proba(self._prepare(X))[:, 1]

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Feature importance.

        ``gain`` rather than ``split``: split counts reward high-cardinality features for being
        splittable, not for being useful. Neither is a substitute for SHAP, which is why the
        explainability layer exists -- importance says which features the model used overall,
        not which ones drove a particular decision.
        """
        if self.model is None:
            raise RuntimeError("feature_importance requested before fit")
        booster = self.model.booster_
        return (
            pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "importance": booster.feature_importance(importance_type=importance_type),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # -- helpers ----------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.categorical_features:
            if col in out.columns and not isinstance(out[col].dtype, pd.CategoricalDtype):
                out[col] = out[col].astype("category")
        return out

    def _present_categoricals(self, X: pd.DataFrame) -> list[str]:
        return [c for c in self.categorical_features if c in X.columns]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> BoostingModel:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
