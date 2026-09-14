"""Risk-model metrics, split into two families that are reported separately throughout.

**Discrimination** — can the model rank borrowers? ROC-AUC, PR-AUC, KS, Gini.
**Calibration** — are the predicted probabilities *numbers*? Brier, log-loss, ECE, MCE.

Keeping them apart is not pedantry. A model that ranks perfectly can be systematically wrong about
the level of risk: multiply every predicted PD by three and the AUC does not move by a thousandth,
while the expected loss the business plans against triples. Layer C consumes probabilities as
quantities, so the calibration family is the one that decides whether the decision engine is
solvent. This is the point of EXP02.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


@dataclass
class RiskMetrics:
    """Discrimination and calibration for one model on one split."""

    n: int
    positive_rate: float
    mean_prediction: float
    # discrimination
    roc_auc: float
    pr_auc: float
    gini: float
    ks: float
    # calibration
    brier: float
    log_loss: float
    ece: float
    mce: float
    calibration_slope: float
    calibration_intercept: float
    #: mean predicted / observed default rate. 1.0 is unbiased; 1.2 means 20% over-prediction.
    #: Named for what it computes. Note this is the *reciprocal* of the ratio credit risk
    #: conventionally calls "O/E" (observed over expected), which is exposed as
    #: :attr:`observed_expected_ratio` below. The two were previously conflated: the field was
    #: named `observed_expected_ratio` while computing predicted/observed.
    predicted_observed_ratio: float
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_expected_ratio"] = self.observed_expected_ratio
        return payload

    @property
    def observed_expected_ratio(self) -> float:
        """Observed over expected, the conventional credit-risk A/E ratio.

        Above 1.0 means the model **under-states** risk. This is the reciprocal of
        :attr:`predicted_observed_ratio`.
        """
        if not np.isfinite(self.predicted_observed_ratio) or self.predicted_observed_ratio == 0:
            return float("nan")
        return 1.0 / self.predicted_observed_ratio

    def summary_line(self) -> str:
        return (
            f"n={self.n:,} rate={self.positive_rate:.4f} | "
            f"AUC={self.roc_auc:.4f} PR-AUC={self.pr_auc:.4f} KS={self.ks:.4f} | "
            f"Brier={self.brier:.5f} ECE={self.ece:.4f} P/O={self.predicted_observed_ratio:.3f}"
        )


def _as_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(pd.Series(y_true).astype(float))
    p = np.asarray(pd.Series(y_prob).astype(float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        raise ValueError("no finite (y_true, y_prob) pairs to score")
    return y[mask], np.clip(p[mask], 1e-9, 1 - 1e-9)


def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: the largest gap between the good and bad cumulative distributions.

    Still the headline discrimination number in most credit-risk shops, so it is reported
    alongside AUC even though the two are near-monotonically related.
    """
    y, p = _as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(np.abs(tpr - fpr)))


def expected_calibration_error(
    y_true, y_prob, n_bins: int = 20, strategy: str = "quantile"
) -> tuple[float, float]:
    """Return ``(ECE, MCE)``: the average and worst absolute gap between predicted and observed.

    Quantile bins by default. Uniform bins put most of the mass of a skewed PD distribution into
    the first bucket and then report a flatteringly small error.
    """
    y, p = _as_arrays(y_true, y_prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 3:
        return float("nan"), float("nan")

    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    gaps, weights = [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        gaps.append(abs(p[mask].mean() - y[mask].mean()))
        weights.append(mask.sum())
    if not gaps:
        return float("nan"), float("nan")
    gaps_arr, weights_arr = np.asarray(gaps), np.asarray(weights, dtype=float)
    return float((gaps_arr * weights_arr).sum() / weights_arr.sum()), float(gaps_arr.max())


def calibration_slope_intercept(y_true, y_prob) -> tuple[float, float]:
    """Regress the outcome on the predicted log-odds (a "calibration-in-the-large" check).

    A perfectly calibrated model gives slope 1, intercept 0. Slope below 1 means the predictions
    are too spread out -- over-confident at both ends, which is the usual signature of an
    uncalibrated gradient-boosted model.
    """
    y, p = _as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1.0 - p))
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    model.fit(logit.reshape(-1, 1), y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def compute_metrics(y_true, y_prob, n_bins: int = 20) -> RiskMetrics:
    """Full metric set for one model on one split."""
    y, p = _as_arrays(y_true, y_prob)
    single_class = len(np.unique(y)) < 2

    roc = float("nan") if single_class else float(roc_auc_score(y, p))
    pr = float("nan") if single_class else float(average_precision_score(y, p))
    ece, mce = expected_calibration_error(y, p, n_bins=n_bins)
    slope, intercept = calibration_slope_intercept(y, p)
    observed, predicted = float(y.mean()), float(p.mean())

    return RiskMetrics(
        n=int(len(y)),
        positive_rate=observed,
        mean_prediction=predicted,
        roc_auc=roc,
        pr_auc=pr,
        gini=float("nan") if single_class else 2.0 * roc - 1.0,
        ks=ks_statistic(y, p),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        ece=ece,
        mce=mce,
        calibration_slope=slope,
        calibration_intercept=intercept,
        predicted_observed_ratio=float(predicted / observed) if observed > 0 else float("nan"),
    )


def reliability_table(y_true, y_prob, n_bins: int = 20, strategy: str = "quantile") -> pd.DataFrame:
    """Binned predicted-vs-observed table, the data behind a reliability diagram."""
    y, p = _as_arrays(y_true, y_prob)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def gains_table(y_true, y_prob, n_bands: int = 10) -> pd.DataFrame:
    """Decile lift table, ordered from riskiest to safest.

    This is the artefact a credit committee actually reads: it answers "if we cut the worst
    decile, how much of the loss do we avoid and how much volume do we give up?".
    """
    y, p = _as_arrays(y_true, y_prob)
    order = np.argsort(-p)
    y_sorted, p_sorted = y[order], p[order]
    bands = np.array_split(np.arange(len(y_sorted)), n_bands)

    total_bad = y_sorted.sum()
    rows, cum_bad, cum_n = [], 0.0, 0
    for i, band in enumerate(bands, start=1):
        bad = float(y_sorted[band].sum())
        cum_bad += bad
        cum_n += len(band)
        rows.append(
            {
                "band": i,
                "n": len(band),
                "mean_predicted": float(p_sorted[band].mean()),
                "observed_rate": float(y_sorted[band].mean()),
                "bad_count": bad,
                "cum_bad_share": float(cum_bad / total_bad) if total_bad else float("nan"),
                "cum_population_share": float(cum_n / len(y_sorted)),
                "lift": float(
                    (y_sorted[band].mean() / y.mean()) if y.mean() > 0 else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_models(results: dict[str, RiskMetrics]) -> pd.DataFrame:
    """Tidy one-row-per-model comparison table, sorted by out-of-time ranking power."""
    frame = pd.DataFrame({name: m.to_dict() for name, m in results.items()}).T
    frame = frame.drop(columns=["extras"], errors="ignore")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.sort_values("roc_auc", ascending=False)
