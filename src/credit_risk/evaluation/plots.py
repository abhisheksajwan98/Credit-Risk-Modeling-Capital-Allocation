"""Figures. Matplotlib only, no seaborn, no styling dependencies.

Each function returns the Figure so a caller can compose or save it. Every figure is written to
``results/figures/`` by the scripts and none of them are required for the numeric results -- a
headless run produces every table without touching this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless by default; scripts run without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.evaluation.metrics import gains_table, reliability_table  # noqa: E402

_FIGSIZE = (7.0, 5.0)


def save(fig: plt.Figure, path: str | Path, dpi: int = 140) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def reliability_diagram(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    n_bins: int = 20,
    title: str = "Reliability (out-of-time test window)",
) -> plt.Figure:
    """Predicted vs observed default rate, one line per model.

    The diagonal is perfect calibration. A line below it means the model over-predicts risk; above
    means it under-predicts. This is the figure that makes EXP02 legible at a glance -- the ROC
    curves for the same models are visually indistinguishable.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration", zorder=1)

    upper = 0.0
    for name, (y_true, y_prob) in curves.items():
        table = reliability_table(y_true, y_prob, n_bins=n_bins)
        ax.plot(
            table["mean_predicted"],
            table["observed_rate"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=name,
        )
        upper = max(upper, float(table[["mean_predicted", "observed_rate"]].to_numpy().max()))

    limit = min(1.0, upper * 1.1 + 0.01)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], title: str = "ROC") -> plt.Figure:
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, zorder=1)
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=1.4, label=f"{name} (AUC {auc:.4f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig


def gains_chart(y_true, y_prob, n_bands: int = 10, title: str = "Cumulative bad capture") -> plt.Figure:
    """The chart a credit committee reads: cut the worst decile, avoid how much of the loss?"""
    table = gains_table(y_true, y_prob, n_bands=n_bands)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    x = np.r_[0.0, table["cum_population_share"].to_numpy()]
    y = np.r_[0.0, table["cum_bad_share"].to_numpy()]
    ax.plot(x, y, marker="o", markersize=4, linewidth=1.5, label="model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="random")
    ax.set_xlabel("Share of population declined (riskiest first)")
    ax.set_ylabel("Share of defaults avoided")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig


def score_distribution(
    scores: dict[str, np.ndarray], bins: int = 60, title: str = "PD distribution"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for name, values in scores.items():
        ax.hist(values, bins=bins, histtype="step", density=True, linewidth=1.4, label=name)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig


def policy_comparison(
    summary: pd.DataFrame,
    metric: str = "mean_reward",
    error: str | None = "std_reward",
    title: str = "Policy comparison",
) -> plt.Figure:
    """Horizontal bars with an error bar, ordered worst to best."""
    frame = summary.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(frame) + 2.0))
    positions = np.arange(len(frame))
    errors = frame[error].to_numpy() if error and error in frame.columns else None
    ax.barh(positions, frame[metric].to_numpy(), xerr=errors, color="#4C72B0", alpha=0.85,
            error_kw={"ecolor": "#333333", "capsize": 3, "linewidth": 1})
    ax.set_yticks(positions)
    ax.set_yticklabels(frame.index, fontsize=9)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x", linewidth=0.5)
    return fig


def vintage_calibration(table: pd.DataFrame, title: str = "Calibration drift by vintage") -> plt.Figure:
    """Observed vs expected default rate over time, with the 1.0 reference line."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(table["vintage"], table["observed_expected"], marker="o", linewidth=1.5,
            label="observed / expected")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="perfectly calibrated")
    ax.set_ylabel("Observed / expected default rate")
    ax.set_xlabel("Vintage")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig


def drift_bars(frame: pd.DataFrame, top_k: int = 20, title: str = "Feature drift (PSI)") -> plt.Figure:
    top = frame.nlargest(top_k, "psi").sort_values("psi")
    colours = [
        "#C44E52" if s == "significant" else "#DD8452" if s == "moderate" else "#4C72B0"
        for s in top["severity"]
    ]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(top) + 1.8))
    ax.barh(np.arange(len(top)), top["psi"].to_numpy(), color=colours, alpha=0.9)
    ax.axvline(0.10, color="#888888", linestyle="--", linewidth=1)
    ax.axvline(0.25, color="#444444", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Population Stability Index")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x", linewidth=0.5)
    return fig


def shap_summary_bar(
    importance: pd.DataFrame, top_k: int = 20, title: str = "Mean |SHAP| contribution"
) -> plt.Figure:
    top = importance.head(top_k).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(top) + 1.8))
    ax.barh(np.arange(len(top)), top["mean_abs_shap"].to_numpy(), color="#55A868", alpha=0.9)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["feature"], fontsize=8)
    ax.set_xlabel("Mean absolute SHAP value (log-odds)")
    ax.set_title(title)
    ax.grid(alpha=0.25, axis="x", linewidth=0.5)
    return fig


def learning_curve(
    rewards: Sequence[float], window: int = 20, title: str = "RL training"
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    values = np.asarray(rewards, dtype=float)
    ax.plot(values, linewidth=0.8, alpha=0.35, label="episode reward")
    if len(values) >= window:
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(values)), smooth, linewidth=1.8,
                label=f"{window}-episode mean")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total reward")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    return fig
