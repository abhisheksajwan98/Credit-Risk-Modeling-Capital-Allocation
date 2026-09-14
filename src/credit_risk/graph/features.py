"""Graph-derived cohort features -- arm **B** of the Layer B comparison.

These are the honest competitor to the GNN. Before claiming a graph neural network adds value, it
has to beat what you get from simply aggregating the cohort's recent history into columns and
handing them to the same LightGBM model. Very often that is most of the benefit, and saying so is
more useful than a GNN result with no baseline underneath it.

The two information sets
------------------------
Neighbour information splits into two kinds with *different* availability rules, and this module
exists to keep them apart:

============================  ==========================================  =======================
Aggregate                     Condition for neighbour ``j`` of loan ``i``  Why
============================  ==========================================  =======================
Feature aggregates            ``issue[j] < issue[i]``                     The application existed
(mean FICO, mean DTI, count)                                              and was observable
Label aggregates              ``resolved[j] <= issue[i]``                 The outcome had actually
(cohort default rate)                                                     happened and been booked
============================  ==========================================  =======================

The second condition is far stricter than the first. A 36-month loan issued in 2013-01 that runs to
term does not resolve until 2016-01, so a 2014 applicant's underwriter cannot know how it ended.
Using ``issue[j] < issue[i]`` for a *label* aggregate is the single most common way graph credit
models leak, and it inflates AUC dramatically because the cohort default rate then contains
outcomes from the same period the model is being scored on.

Both aggregates are computed with per-cohort prefix sums and ``searchsorted``, which is O(m log m)
per cohort and avoids ever materialising the pairwise join.

A note on the two counts, because the relationship between them is not the obvious one.
``n_prior`` windows on the **issue** date and measures cohort *activity*: how many
applications from this cohort have been seen recently. ``n_resolved`` windows on the
**resolution** date and measures cohort *experience*: how many outcomes from this cohort are
known recently. These are different windows on different time axes, so
``n_resolved <= n_prior`` does **not** hold in general -- a neighbour issued forty months ago
sits outside the issue window but, having resolved ten months ago, sits inside the resolution
window. Both quantities are wanted, and collapsing them onto one axis would discard the more
informative of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.graph.construction import _NA_SENTINEL, DEFAULT_RELATIONS, RelationSpec
from credit_risk.utils.runtime import get_logger

LOG = get_logger("graph.features")


@dataclass
class CohortFeatureConfig:
    """Configuration for cohort aggregation."""

    relations: tuple[RelationSpec, ...] = DEFAULT_RELATIONS
    #: Months of history each aggregate looks back over. Recent cohort experience is the
    #: signal; a ZIP's behaviour in 2008 says little about the same ZIP in 2015.
    lookback_months: int = 36
    #: Empirical-Bayes smoothing weight. A cohort default rate computed from 3 loans is noise;
    #: shrinking it toward the portfolio prior with a pseudo-count of `alpha` makes small
    #: cohorts default to "average" instead of "0% or 100%".
    smoothing_alpha: float = 25.0
    #: Numeric columns averaged over the cohort's recent applications.
    feature_columns: tuple[str, ...] = ("fico_mid", "dti_clean", "loan_amnt", "annual_inc")


@dataclass
class CohortFeatureState:
    """The only fitted quantity: the portfolio prior, learned on the training window."""

    prior_default_rate: float = 0.15
    feature_names: list[str] = field(default_factory=list)


class CohortFeatureBuilder:
    """Builds time-respecting cohort aggregates.

    ``fit`` learns only the smoothing prior, from the training window. ``transform`` then runs over
    the **whole** frame, because a 2015 test loan's legitimate predecessors include 2012 training
    loans -- that is past information, not leakage. What makes it safe is the per-row as-of
    condition, not withholding rows.
    """

    def __init__(self, config: CohortFeatureConfig | None = None) -> None:
        self.config = config or CohortFeatureConfig()
        self.state = CohortFeatureState()
        self._fitted = False

    def fit(self, train: pd.DataFrame, target_column: str = "default") -> CohortFeatureBuilder:
        target = pd.to_numeric(train[target_column], errors="coerce").dropna()
        if len(target) == 0:
            raise ValueError("cannot fit cohort prior: no labelled rows in the training window")
        self.state.prior_default_rate = float(target.mean())
        self._fitted = True
        LOG.info(
            "cohort prior fitted on %s training rows: %.4f",
            f"{len(target):,}",
            self.state.prior_default_rate,
        )
        return self

    def transform(
        self,
        df: pd.DataFrame,
        period_column: str = "issue_period",
        observed_column: str = "outcome_observed_period",
        label_column: str = "default_observed",
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("CohortFeatureBuilder.transform called before fit")

        n = len(df)
        months = _period_to_int(df[period_column])
        observed = _period_to_int(df[observed_column], allow_na=True)
        labels = pd.to_numeric(df[label_column], errors="coerce").to_numpy(dtype=float)

        # A loan cannot resolve before it is issued. If this ever fires, the observation-date
        # logic is wrong and every label aggregate below is contaminated.
        resolved = ~np.isnan(observed)
        if resolved.any() and (observed[resolved] <= months[resolved]).any():
            bad = int((observed[resolved] <= months[resolved]).sum())
            raise AssertionError(
                f"{bad} loans resolve on or before their issue month; "
                "outcome_observed_period is miscomputed and label aggregates would leak"
            )

        out = pd.DataFrame(index=df.index)
        cfg = self.config

        for relation in cfg.relations:
            if relation.key_column not in df.columns:
                LOG.warning("cohort relation %r skipped: %r absent", relation.name, relation.key_column)
                continue

            prefix = f"cohort_{relation.name}"
            key_series = df[relation.key_column].astype("string")
            keys = key_series.fillna(_NA_SENTINEL).to_numpy(dtype=object)
            key_valid = key_series.notna().to_numpy()

            n_prior = np.zeros(n, dtype=np.float32)
            n_resolved = np.zeros(n, dtype=np.float32)
            n_bad = np.zeros(n, dtype=np.float32)
            feature_sums = {c: np.full(n, np.nan, dtype=np.float32) for c in cfg.feature_columns}

            for positions in _cohort_positions(keys, key_valid):
                if len(positions) < relation.min_cohort_size:
                    continue
                self._accumulate_cohort(
                    positions=positions,
                    months=months,
                    observed=observed,
                    labels=labels,
                    df=df,
                    n_prior=n_prior,
                    n_resolved=n_resolved,
                    n_bad=n_bad,
                    feature_sums=feature_sums,
                )

            alpha = cfg.smoothing_alpha
            prior = self.state.prior_default_rate
            smoothed = (n_bad + alpha * prior) / (n_resolved + alpha)

            out[f"{prefix}_n_prior"] = n_prior
            out[f"{prefix}_n_resolved"] = n_resolved
            out[f"{prefix}_default_rate"] = smoothed.astype(np.float32)
            # How much of the smoothed rate is real evidence rather than the prior. A model can
            # use this to discount the rate for thin cohorts, and a human can read it directly.
            out[f"{prefix}_evidence_weight"] = (n_resolved / (n_resolved + alpha)).astype(np.float32)
            # Lift over the portfolio prior: >1 means this cohort has run worse than average.
            out[f"{prefix}_default_lift"] = (smoothed / max(prior, 1e-9)).astype(np.float32)
            for col, sums in feature_sums.items():
                out[f"{prefix}_mean_{col}"] = sums

            LOG.info(
                "%s | mean prior loans %.1f | mean resolved %.1f | isolated %.1f%%",
                prefix,
                n_prior.mean(),
                n_resolved.mean(),
                100.0 * (n_prior == 0).mean(),
            )

        self.state.feature_names = list(out.columns)
        return out

    # -- internals ---------------------------------------------------------
    def _accumulate_cohort(
        self,
        positions: np.ndarray,
        months: np.ndarray,
        observed: np.ndarray,
        labels: np.ndarray,
        df: pd.DataFrame,
        n_prior: np.ndarray,
        n_resolved: np.ndarray,
        n_bad: np.ndarray,
        feature_sums: dict[str, np.ndarray],
    ) -> None:
        cfg = self.config
        window = cfg.lookback_months

        # ---- feature aggregates: gated on issue date -----------------------
        issue_order = np.argsort(months[positions], kind="stable")
        pos_by_issue = positions[issue_order]
        m_sorted = months[pos_by_issue]

        upper = np.searchsorted(m_sorted, m_sorted, side="left")  # strictly earlier only
        lower = np.searchsorted(m_sorted, m_sorted - window, side="left")
        counts = np.maximum(upper - lower, 0)
        n_prior[pos_by_issue] = counts

        for col in cfg.feature_columns:
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)[pos_by_issue]
            present = ~np.isnan(values)
            csum = np.concatenate([[0.0], np.cumsum(np.where(present, values, 0.0))])
            ccnt = np.concatenate([[0.0], np.cumsum(present.astype(float))])
            total = csum[upper] - csum[lower]
            count = ccnt[upper] - ccnt[lower]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, total / count, np.nan)
            feature_sums[col][pos_by_issue] = mean.astype(np.float32)

        # ---- label aggregates: gated on *resolution* date ------------------
        # Sorting by resolution rather than issue is the whole point. `searchsorted(..., "right")`
        # against the applicant's issue month admits exactly those neighbours whose outcome had
        # already been booked when the application landed.
        res_mask = ~np.isnan(observed[positions])
        res_positions = positions[res_mask]
        if len(res_positions) == 0:
            return
        res_order = np.argsort(observed[res_positions], kind="stable")
        pos_by_res = res_positions[res_order]
        o_sorted = observed[pos_by_res]
        lab_sorted = labels[pos_by_res]

        present = ~np.isnan(lab_sorted)
        csum_bad = np.concatenate([[0.0], np.cumsum(np.where(present, lab_sorted, 0.0))])
        ccnt_res = np.concatenate([[0.0], np.cumsum(present.astype(float))])

        target_months = months[positions]
        hi = np.searchsorted(o_sorted, target_months, side="right")
        lo = np.searchsorted(o_sorted, target_months - window, side="left")
        n_resolved[positions] = np.maximum(ccnt_res[hi] - ccnt_res[lo], 0)
        n_bad[positions] = np.maximum(csum_bad[hi] - csum_bad[lo], 0)


def _cohort_positions(keys: np.ndarray, valid: np.ndarray):
    """Yield positional index arrays, one per non-null cohort key."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    ordered_valid = valid[order]
    if not ordered_valid.any():
        return
    boundaries = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if not ordered_valid[start]:
            continue
        yield order[start:stop]


def _period_to_int(series: pd.Series, allow_na: bool = False) -> np.ndarray:
    periods = pd.PeriodIndex(series, freq="M")
    values = periods.year.to_numpy(dtype="float64") * 12 + periods.month.to_numpy(dtype="float64")
    na = periods.isna()
    if na.any():
        if not allow_na:
            raise ValueError("unexpected missing period values")
        values = values.astype(float)
        values[na] = np.nan
        return values
    return values.astype(np.int64) if not allow_na else values.astype(float)


def save_cohort_features(features: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(path, index=False, compression="snappy")
    return path
