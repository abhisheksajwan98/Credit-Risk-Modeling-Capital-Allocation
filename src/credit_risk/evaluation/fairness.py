"""A limited fairness slice.

Scope, stated first because it constrains everything below
-----------------------------------------------------------
LendingClub's public files contain **no protected attributes**. There is no race, sex, age,
marital status or national origin -- US fair-lending law (ECOA/Regulation B) generally bars
collecting most of them for non-mortgage consumer credit in the first place. So a genuine
disparate-impact analysis is not possible with this data, and this module does not pretend
otherwise.

What *is* possible, and what is done here, is to measure whether the model's errors and its
approval decisions fall unevenly across groups that are observable and that plausibly correlate
with protected characteristics:

* **Geography** (``addr_state``, ``zip3``) -- correlated with race and income in the US to a
  well-documented degree, which is the entire history of redlining.
* **Income band** -- not protected, but the group most affected by a credit decision.
* **Employment length** and **home ownership** -- correlated with age and wealth.

The right claim to make from this is narrow: *the model's error rates and approval rates differ
across these observable groups by this much*. Not that the model is fair, and not that it is
unfair. Both would require data this dataset does not have.

Why calibration-by-group is the metric to lead with
----------------------------------------------------
The fairness literature's impossibility result matters here in a practical way: when base rates
differ across groups, you cannot simultaneously equalise calibration, false-positive rates and
false-negative rates. Something has to give, and it is a policy choice which. For credit
specifically, **calibration within groups** is the property most directly tied to whether
individuals are being priced correctly, so it leads -- and the error-rate gaps are reported
alongside it rather than instead of it, so the trade-off is visible rather than hidden by the
choice of metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from credit_risk.evaluation.metrics import compute_metrics
from credit_risk.utils.runtime import get_logger

LOG = get_logger("evaluation.fairness")

#: Groups available in this dataset. None of these are protected attributes; each is a proxy whose
#: limitations are documented above.
DEFAULT_GROUP_COLUMNS: tuple[str, ...] = ("addr_state", "home_ownership", "emp_length", "purpose")


@dataclass
class GroupMetrics:
    group: str
    n: int
    share: float
    observed_rate: float
    mean_predicted: float
    observed_expected: float
    roc_auc: float
    approval_rate: float
    tpr: float
    fpr: float


def _rates_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    """Approval rate, and the error rates *among those the lender got wrong*.

    ``tpr`` here is the share of eventual defaulters correctly declined, and ``fpr`` the share of
    eventual repayers wrongly declined. The second is the one that harms individuals: a
    creditworthy applicant refused a loan.
    """
    declined = y_prob >= threshold
    approval_rate = float((~declined).mean())
    bad, good = y_true > 0.5, y_true <= 0.5
    tpr = float(declined[bad].mean()) if bad.any() else float("nan")
    fpr = float(declined[good].mean()) if good.any() else float("nan")
    return approval_rate, tpr, fpr


def group_analysis(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_column: str,
    threshold: float = 0.15,
    min_group_size: int = 200,
) -> pd.DataFrame:
    """Per-group calibration, discrimination and decision rates.

    Groups below ``min_group_size`` are pooled into ``__small__`` rather than reported. A default
    rate computed from forty loans is noise, and presenting it as a fairness finding would be
    worse than reporting nothing.
    """
    frame = pd.DataFrame(
        {
            "group": df[group_column].astype("string").fillna("__missing__").to_numpy(),
            "y": np.asarray(y_true, dtype=float),
            "p": np.asarray(y_prob, dtype=float),
        }
    )
    counts = frame["group"].value_counts()
    small = set(counts[counts < min_group_size].index)
    frame.loc[frame["group"].isin(small), "group"] = "__small__"

    rows: list[GroupMetrics] = []
    total = len(frame)
    for group, chunk in frame.groupby("group", sort=True):
        y = chunk["y"].to_numpy()
        p = chunk["p"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        metrics = compute_metrics(y, p)
        approval, tpr, fpr = _rates_at_threshold(y, p, threshold)
        rows.append(
            GroupMetrics(
                group=str(group),
                n=len(chunk),
                share=len(chunk) / total,
                observed_rate=metrics.positive_rate,
                mean_predicted=metrics.mean_prediction,
                observed_expected=metrics.observed_expected_ratio,
                roc_auc=metrics.roc_auc,
                approval_rate=approval,
                tpr=tpr,
                fpr=fpr,
            )
        )

    return (
        pd.DataFrame([r.__dict__ for r in rows])
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def disparity_summary(table: pd.DataFrame, reference: str | None = None) -> dict[str, float]:
    """Spread across groups, plus the ratio to a reference group.

    ``approval_rate_ratio_min`` is the closest thing here to the "four-fifths rule" heuristic used
    in US employment-discrimination screening -- a ratio below 0.8 is conventionally treated as
    worth investigating. It is applied to a **non-protected proxy** group here, so it is a
    diagnostic prompt, not a legal finding.
    """
    if table.empty:
        return {}
    working = table[table["group"] != "__small__"]
    if working.empty:
        return {}

    reference_row = (
        working.loc[working["group"] == reference]
        if reference and (working["group"] == reference).any()
        else working.nlargest(1, "n")
    )
    ref_approval = float(reference_row["approval_rate"].iloc[0])

    return {
        "n_groups": float(len(working)),
        "reference_group": reference_row["group"].iloc[0],
        "approval_rate_min": float(working["approval_rate"].min()),
        "approval_rate_max": float(working["approval_rate"].max()),
        "approval_rate_spread": float(
            working["approval_rate"].max() - working["approval_rate"].min()
        ),
        "approval_rate_ratio_min": (
            float(working["approval_rate"].min() / ref_approval) if ref_approval > 0 else float("nan")
        ),
        "fpr_spread": float(working["fpr"].max() - working["fpr"].min()),
        "tpr_spread": float(working["tpr"].max() - working["tpr"].min()),
        "observed_expected_min": float(working["observed_expected"].min()),
        "observed_expected_max": float(working["observed_expected"].max()),
        "calibration_spread": float(
            working["observed_expected"].max() - working["observed_expected"].min()
        ),
        "auc_min": float(working["roc_auc"].min()),
        "auc_max": float(working["roc_auc"].max()),
    }


def fairness_report(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    threshold: float = 0.15,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Run the slice across several proxy groupings and write a cautious verdict."""
    tables: dict[str, pd.DataFrame] = {}
    lines = [
        "Fairness slice. LendingClub publishes no protected attributes, so the groups below are "
        "observable proxies, and nothing here supports a claim about disparate impact in the "
        "legal sense. See the module docstring in evaluation/fairness.py.",
        "",
    ]

    for column in group_columns:
        if column not in df.columns:
            continue
        table = group_analysis(df, y_true, y_prob, column, threshold=threshold)
        if table.empty:
            continue
        tables[column] = table
        summary = disparity_summary(table)
        if not summary:
            continue
        lines.append(
            f"{column}: {int(summary['n_groups'])} groups. Approval rate ranges "
            f"{summary['approval_rate_min']:.1%} to {summary['approval_rate_max']:.1%} "
            f"(spread {summary['approval_rate_spread']:.1%}); calibration (observed/expected) "
            f"ranges {summary['observed_expected_min']:.3f} to "
            f"{summary['observed_expected_max']:.3f}; AUC ranges "
            f"{summary['auc_min']:.3f} to {summary['auc_max']:.3f}."
        )
        if np.isfinite(summary["approval_rate_ratio_min"]) and summary["approval_rate_ratio_min"] < 0.8:
            lines.append(
                f"  -> The lowest-approval group sits at "
                f"{summary['approval_rate_ratio_min']:.2f}x the reference group "
                f"({summary['reference_group']}). Below the 0.8 heuristic; on a protected "
                f"attribute this would warrant investigation."
            )
        LOG.info("fairness | %s | %s", column, summary)

    lines += [
        "",
        "Reading: a calibration spread near zero means the model's probabilities mean the same "
        "thing in every group, which is the property most directly tied to correct pricing. "
        "Approval-rate spread is expected to be non-zero whenever true risk differs between "
        "groups, and on its own is not evidence of a problem.",
    ]
    return tables, "\n".join(lines)
