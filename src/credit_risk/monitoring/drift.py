"""Production monitoring: feature drift, prediction drift, and calibration drift.

Why the three are separate, and why the order matters
-----------------------------------------------------
In a real credit portfolio you learn about problems in this sequence, and the gap between the
first and the last is measured in years:

1. **Feature drift** -- visible immediately. The applicant mix changed.
2. **Prediction drift** -- visible immediately. The score distribution moved, and with it the
   approval rate under a fixed cut-off.
3. **Calibration drift** -- visible only once outcomes mature. The scores no longer mean what
   they said they meant.

A 36-month loan booked today does not tell you whether it was a good decision until 2029. So the
only signals available in the moment are the first two, and the entire discipline of model
monitoring in lending exists because of that lag. This project mirrors the situation exactly: the
2016-2018 window is carried through the pipeline **without labels** and used as production
traffic, while calibration drift is measured on the vintages that have matured.

On PSI thresholds
-----------------
The conventional bands (<0.1 stable, 0.1-0.25 moderate, >0.25 significant) are industry
convention, not a statistical result, and they are sensitive to bin count and sample size. They
are reported here because they are what a model-risk committee expects to see, with the caveat
attached rather than omitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_risk.evaluation.metrics import compute_metrics
from credit_risk.utils.runtime import get_logger

LOG = get_logger("monitoring.drift")

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between a reference and a current distribution.

    ``sum (current_share - reference_share) * ln(current_share / reference_share)``

    Bin edges come from the **reference** quantiles, never from the pooled data. Re-binning on the
    combined sample would absorb part of the very shift the statistic is meant to detect.
    """
    ref = pd.Series(reference).dropna().to_numpy(dtype=float)
    cur = pd.Series(current).dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = np.maximum(ref_counts / max(ref_counts.sum(), 1), epsilon)
    cur_share = np.maximum(cur_counts / max(cur_counts.sum(), 1), epsilon)

    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def categorical_psi(
    reference: pd.Series, current: pd.Series, epsilon: float = 1e-6
) -> float:
    """PSI for a categorical column, over the union of observed levels."""
    ref = reference.astype("string").fillna("__missing__").value_counts(normalize=True)
    cur = current.astype("string").fillna("__missing__").value_counts(normalize=True)
    levels = ref.index.union(cur.index)
    ref_share = np.maximum(ref.reindex(levels).fillna(0.0).to_numpy(), epsilon)
    cur_share = np.maximum(cur.reindex(levels).fillna(0.0).to_numpy(), epsilon)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def classify_psi(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_SIGNIFICANT:
        return "moderate"
    return "significant"


@dataclass
class FeatureDriftReport:
    frame: pd.DataFrame
    reference_label: str
    current_label: str

    def significant(self) -> pd.DataFrame:
        return self.frame[self.frame["severity"] == "significant"]

    def summary(self) -> dict[str, Any]:
        return {
            "reference": self.reference_label,
            "current": self.current_label,
            "n_features": int(len(self.frame)),
            "n_moderate": int((self.frame["severity"] == "moderate").sum()),
            "n_significant": int((self.frame["severity"] == "significant").sum()),
            "max_psi": float(self.frame["psi"].max()) if len(self.frame) else float("nan"),
            "worst_features": self.frame.nlargest(5, "psi")["feature"].tolist(),
        }


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    reference_label: str = "train",
    current_label: str = "production",
    n_bins: int = 10,
) -> FeatureDriftReport:
    """PSI and a KS test per feature."""
    rows = []
    for col in numeric_features:
        if col not in reference.columns or col not in current.columns:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur = pd.to_numeric(current[col], errors="coerce").dropna()
        psi = population_stability_index(ref, cur, n_bins=n_bins)
        ks = ks_2samp(ref, cur).statistic if len(ref) and len(cur) else float("nan")
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "psi": psi,
                "ks": float(ks),
                "severity": classify_psi(psi),
                "reference_mean": float(ref.mean()) if len(ref) else float("nan"),
                "current_mean": float(cur.mean()) if len(cur) else float("nan"),
                "missing_shift": float(current[col].isna().mean() - reference[col].isna().mean()),
            }
        )

    for col in categorical_features:
        if col not in reference.columns or col not in current.columns:
            continue
        psi = categorical_psi(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "psi": psi,
                "ks": float("nan"),
                "severity": classify_psi(psi),
                "reference_mean": float("nan"),
                "current_mean": float("nan"),
                "missing_shift": float(current[col].isna().mean() - reference[col].isna().mean()),
            }
        )

    frame = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    report = FeatureDriftReport(frame, reference_label, current_label)
    LOG.info(
        "feature drift %s -> %s: %d significant, %d moderate of %d features",
        reference_label,
        current_label,
        report.summary()["n_significant"],
        report.summary()["n_moderate"],
        len(frame),
    )
    return report


@dataclass
class PredictionDriftReport:
    psi: float
    severity: str
    reference_mean: float
    current_mean: float
    reference_quantiles: dict[str, float]
    current_quantiles: dict[str, float]
    approval_rate_shift: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    thresholds: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30),
) -> PredictionDriftReport:
    """Drift in the score distribution, and what it does to the approval rate.

    The approval-rate column is the one that matters operationally. A PSI of 0.18 is an abstraction;
    "under our 15% cut-off, approvals fall from 54% to 41%" is a business event, and it is
    detectable the day it happens rather than three years later.
    """
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    psi = population_stability_index(ref, cur)

    quantile_points = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    approval_shift = {
        f"tau={t:g}": float((cur < t).mean() - (ref < t).mean()) for t in thresholds
    }

    report = PredictionDriftReport(
        psi=psi,
        severity=classify_psi(psi),
        reference_mean=float(ref.mean()),
        current_mean=float(cur.mean()),
        reference_quantiles={f"q{int(q*100)}": float(np.quantile(ref, q)) for q in quantile_points},
        current_quantiles={f"q{int(q*100)}": float(np.quantile(cur, q)) for q in quantile_points},
        approval_rate_shift=approval_shift,
    )
    LOG.info(
        "prediction drift: PSI %.4f (%s), mean PD %.4f -> %.4f",
        psi,
        report.severity,
        report.reference_mean,
        report.current_mean,
    )
    return report


def calibration_drift(
    df: pd.DataFrame,
    score_column: str,
    target_column: str = "default",
    period_column: str = "issue_period",
    freq: str = "Q",
) -> pd.DataFrame:
    """Track calibration by vintage on the windows that have matured.

    The column to read is ``observed_expected``. A value drifting above 1 means the model is
    under-predicting risk for that vintage -- exactly the failure that made 2015-2016 marketplace
    vintages painful for investors who assumed a model fitted on 2012 still applied.
    """
    frame = df.loc[pd.to_numeric(df[target_column], errors="coerce").notna()].copy()
    if not len(frame):
        return pd.DataFrame()

    periods = pd.PeriodIndex(frame[period_column], freq="M")
    frame["_vintage"] = periods.asfreq(freq).astype(str)

    rows = []
    for vintage, group in frame.groupby("_vintage", sort=True):
        y = pd.to_numeric(group[target_column], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(group[score_column], errors="coerce").to_numpy(dtype=float)
        if len(y) < 50 or len(np.unique(y)) < 2:
            continue
        metrics = compute_metrics(y, p)
        rows.append(
            {
                "vintage": vintage,
                "n": metrics.n,
                "observed_rate": metrics.positive_rate,
                "mean_predicted": metrics.mean_prediction,
                "observed_expected": (
                    metrics.positive_rate / metrics.mean_prediction
                    if metrics.mean_prediction > 0
                    else float("nan")
                ),
                "roc_auc": metrics.roc_auc,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "calibration_slope": metrics.calibration_slope,
            }
        )
    return pd.DataFrame(rows)


def policy_drift(
    df: pd.DataFrame,
    decision_column: str,
    period_column: str = "issue_period",
    freq: str = "Q",
) -> pd.DataFrame:
    """Approval mix over time -- the operational counterpart to prediction drift."""
    frame = df.copy()
    frame["_vintage"] = pd.PeriodIndex(frame[period_column], freq="M").asfreq(freq).astype(str)
    counts = (
        frame.groupby(["_vintage", decision_column]).size().unstack(fill_value=0)
    )
    return counts.div(counts.sum(axis=1), axis=0)


def monitoring_summary(
    feature_report: FeatureDriftReport,
    prediction_report: PredictionDriftReport,
    calibration_table: pd.DataFrame,
) -> str:
    """A short written verdict, of the kind that would open a monitoring pack."""
    feature_summary = feature_report.summary()
    lines = [
        f"Reference window: {feature_report.reference_label}; "
        f"current window: {feature_report.current_label}.",
        f"Feature drift: {feature_summary['n_significant']} of {feature_summary['n_features']} "
        f"features show significant drift (PSI > {PSI_SIGNIFICANT}); "
        f"worst: {', '.join(feature_summary['worst_features'][:3])}.",
        f"Prediction drift: PSI {prediction_report.psi:.4f} ({prediction_report.severity}); "
        f"mean PD {prediction_report.reference_mean:.4f} -> {prediction_report.current_mean:.4f}.",
    ]
    if prediction_report.approval_rate_shift:
        worst = max(prediction_report.approval_rate_shift.items(), key=lambda kv: abs(kv[1]))
        lines.append(
            f"Largest approval-rate shift: {worst[0]} moves {worst[1]:+.1%}."
        )
    if len(calibration_table):
        first, last = calibration_table.iloc[0], calibration_table.iloc[-1]
        lines.append(
            f"Calibration drift on matured vintages: observed/expected "
            f"{first['observed_expected']:.3f} ({first['vintage']}) -> "
            f"{last['observed_expected']:.3f} ({last['vintage']})."
        )
    lines.append(
        "Note: calibration drift is only measurable on matured vintages. For the current "
        "production window, feature and prediction drift are the only available signals -- "
        "outcomes will not be known for up to three years."
    )
    return "\n".join(lines)
