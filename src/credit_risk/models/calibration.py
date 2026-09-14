"""Probability calibration.

The interview question this module exists to answer:

    *Why is a well-ranked model not necessarily a good probability model?*

Because AUC is invariant to any strictly monotone transform of the score. Take a perfectly
calibrated model and cube every prediction: the ordering of borrowers is untouched, so AUC, Gini
and KS do not move at all, while every predicted PD is now wrong and the expected loss computed
from them is wrong with it. Ranking answers "who is riskier than whom". Calibration answers "how
risky, in units you can multiply by an exposure". Layer C multiplies, so it needs the second.

Gradient boosting on log-loss is closer to calibrated than most people assume, but it is still
typically over-confident at the extremes -- the calibration slope comes in below 1. Anything that
distorts the base rate (class weighting, undersampling the majority) breaks calibration outright,
which is precisely why those techniques are treated as hypotheses in EXP01 rather than defaults.

**The calibrator is fitted on the validation window, never on training data.** A calibrator fitted
on the same rows the model was trained on sees the model's in-sample, over-fitted scores and
learns a map that is wrong everywhere else.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from credit_risk.evaluation.metrics import RiskMetrics, compute_metrics
from credit_risk.utils.runtime import get_logger

LOG = get_logger("models.calibration")

Method = Literal["identity", "platt", "isotonic"]

_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(p / (1.0 - p))


class ProbabilityCalibrator:
    """Maps raw model scores to calibrated probabilities.

    Methods
    -------
    ``identity``
        No transform. The control condition -- without it there is no way to show calibration
        changed anything.
    ``platt``
        A one-parameter-plus-intercept logistic regression on the predicted log-odds. Parametric,
        so it is stable on small validation sets and cannot invent non-monotonicity, but it can
        only apply a smooth sigmoidal correction.
    ``isotonic``
        A non-decreasing step function fitted by pool-adjacent-violators. Strictly more flexible
        and usually wins on Brier when the validation set is large, at the cost of a piecewise
        constant output -- it maps whole score ranges onto one probability, which coarsens the
        ranking slightly. With ~80k validation rows here, isotonic has plenty of data.
    """

    def __init__(self, method: Method = "isotonic") -> None:
        if method not in ("identity", "platt", "isotonic"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method: Method = method
        self._platt: LogisticRegression | None = None
        self._isotonic: IsotonicRegression | None = None
        self._fitted = False

    def fit(self, y_true, y_prob) -> ProbabilityCalibrator:
        y = np.asarray(pd.Series(y_true).astype(float))
        p = np.asarray(pd.Series(y_prob).astype(float))
        mask = np.isfinite(y) & np.isfinite(p)
        y, p = y[mask], p[mask]
        if len(np.unique(y)) < 2:
            raise ValueError("calibration needs both classes present in the fitting window")

        if self.method == "platt":
            self._platt = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
            self._platt.fit(_logit(p).reshape(-1, 1), y)
            LOG.info(
                "platt fitted: slope=%.4f intercept=%.4f",
                float(self._platt.coef_[0][0]),
                float(self._platt.intercept_[0]),
            )
        elif self.method == "isotonic":
            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
            )
            self._isotonic.fit(p, y)
            LOG.info("isotonic fitted on %s validation rows", f"{len(p):,}")

        self._fitted = True
        return self

    def transform(self, y_prob) -> np.ndarray:
        p = np.asarray(pd.Series(y_prob).astype(float))
        if self.method == "identity":
            return np.clip(p, _EPS, 1 - _EPS)
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator.transform called before fit")
        if self.method == "platt":
            assert self._platt is not None
            return self._platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        assert self._isotonic is not None
        return np.clip(self._isotonic.predict(p), _EPS, 1 - _EPS)

    def fit_transform(self, y_true, y_prob) -> np.ndarray:
        return self.fit(y_true, y_prob).transform(y_prob)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: str | Path) -> ProbabilityCalibrator:
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


@dataclass
class CalibrationComparison:
    """Result of EXP02: what each calibration method did on the out-of-time window."""

    per_method: dict[str, RiskMetrics]
    fitted: dict[str, ProbabilityCalibrator]
    calibrated_test: dict[str, np.ndarray]

    def table(self) -> pd.DataFrame:
        rows = []
        for name, metrics in self.per_method.items():
            row = metrics.to_dict()
            row.pop("extras", None)
            row["method"] = name
            rows.append(row)
        frame = pd.DataFrame(rows).set_index("method")
        # Ranking metrics first, then the probability-quality metrics that are the point.
        order = [
            "n", "positive_rate", "mean_prediction", "roc_auc", "pr_auc", "ks",
            "brier", "log_loss", "ece", "mce", "calibration_slope",
            "calibration_intercept", "predicted_observed_ratio",
        ]
        return frame[[c for c in order if c in frame.columns]]

    def verdict(self, reference: str = "identity") -> str:
        """One-line reading of whether calibration was worth doing."""
        if reference not in self.per_method:
            return "no reference method available"
        base = self.per_method[reference]
        best = min(
            (m for k, m in self.per_method.items() if k != reference),
            key=lambda m: m.brier,
            default=None,
        )
        if best is None:
            return "only the reference method was evaluated"
        best_name = next(k for k, m in self.per_method.items() if m is best)
        d_brier = 100.0 * (base.brier - best.brier) / base.brier if base.brier else float("nan")
        d_auc = best.roc_auc - base.roc_auc
        return (
            f"{best_name} reduced Brier by {d_brier:.2f}% "
            f"({base.brier:.5f} -> {best.brier:.5f}) and moved ROC-AUC by {d_auc:+.5f}; "
            f"ECE {base.ece:.4f} -> {best.ece:.4f}, "
            f"calibration slope {base.calibration_slope:.3f} -> {best.calibration_slope:.3f}"
        )


def compare_calibrations(
    y_valid,
    p_valid,
    y_test,
    p_test,
    methods: tuple[Method, ...] = ("identity", "platt", "isotonic"),
) -> CalibrationComparison:
    """Fit each calibrator on validation, score all of them on the out-of-time window.

    The AUC column is the one to watch alongside Brier: it should barely move. That is the whole
    demonstration -- calibration changes the numbers without changing the ordering, so any metric
    that only sees the ordering cannot detect the improvement.
    """
    per_method: dict[str, RiskMetrics] = {}
    fitted: dict[str, ProbabilityCalibrator] = {}
    calibrated: dict[str, np.ndarray] = {}

    for method in methods:
        calibrator = ProbabilityCalibrator(method)
        if method != "identity":
            calibrator.fit(y_valid, p_valid)
        p_cal = calibrator.transform(p_test)
        per_method[method] = compute_metrics(y_test, p_cal)
        fitted[method] = calibrator
        calibrated[method] = p_cal
        LOG.info("[%s] %s", method, per_method[method].summary_line())

    return CalibrationComparison(per_method, fitted, calibrated)
