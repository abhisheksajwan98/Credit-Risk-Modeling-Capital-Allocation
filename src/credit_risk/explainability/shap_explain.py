"""SHAP attribution for the tabular models, plus a gradient attribution for the GNN.

For LightGBM, ``TreeExplainer`` gives exact Shapley values in polynomial time, so there is no
sampling error to reason about. For the GNN there is no exact analogue, so the fallback is
gradient-times-input on the node's own features -- an honest approximation that is labelled as
one rather than presented as SHAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.utils.runtime import get_logger

LOG = get_logger("explainability.shap")


@dataclass
class ShapResult:
    values: np.ndarray  # (n_rows, n_features), in log-odds space for a binary LightGBM model
    feature_names: list[str]
    base_value: float
    frame: pd.DataFrame | None = None

    def global_importance(self) -> pd.DataFrame:
        """Mean absolute contribution per feature.

        Different from LightGBM's ``gain`` importance and usually more trustworthy: gain measures
        how much a feature improved the training objective, while this measures how much it moves
        actual predictions. A feature can score high on gain by being used in many low-impact
        splits.
        """
        mean_abs = np.abs(self.values).mean(axis=0)
        signed = self.values.mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "mean_abs_shap": mean_abs,
                    "mean_shap": signed,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def row(self, i: int) -> np.ndarray:
        return self.values[i]


def explain_boosting(
    model,
    X: pd.DataFrame,
    max_rows: int = 5000,
    seed: int = 0,
) -> ShapResult:
    """Exact tree SHAP for a :class:`~credit_risk.models.boosting.BoostingModel`.

    ``max_rows`` subsamples for speed. Tree SHAP is polynomial rather than exponential, but it is
    still the slowest step in the pipeline on a few hundred thousand rows, and the global picture
    is stable well before then.
    """
    import shap

    booster = getattr(model, "model", model)
    frame = model._prepare(X) if hasattr(model, "_prepare") else X

    if len(frame) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(frame), size=max_rows, replace=False)
        frame = frame.iloc[np.sort(idx)]

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(frame)
    # Different SHAP versions return either an array or a per-class list for binary models.
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).ravel()[-1])

    LOG.info("tree SHAP computed for %s rows, %d features", f"{len(frame):,}", values.shape[1])
    return ShapResult(
        values=values,
        feature_names=list(frame.columns),
        base_value=float(base),
        frame=frame,
    )


def explain_logistic(model, X: pd.DataFrame) -> ShapResult:
    """Contributions for the logistic baseline.

    For a linear model the Shapley value has a closed form: ``phi_j = beta_j * (x_j - E[x_j])``.
    No approximation and no explainer object needed -- which is worth pointing out, because it is
    exactly why regulated lenders liked scorecards in the first place.
    """
    pipeline = model.pipeline
    prep = pipeline.named_steps["prep"]
    linear = pipeline.named_steps["model"]

    design = prep.transform(model._prepare(X))
    names = list(prep.get_feature_names_out())
    coefs = linear.coef_[0]
    means = design.mean(axis=0)
    values = (design - means) * coefs

    return ShapResult(
        values=np.asarray(values),
        feature_names=names,
        base_value=float(linear.intercept_[0] + float(means @ coefs)),
    )


def explain_gnn_gradients(
    trainer,
    nodes: np.ndarray,
    feature_names: list[str],
    batch_size: int = 256,
) -> ShapResult:
    """Gradient-times-input attribution over a node's **own** features.

    Not SHAP, and labelled as such. It answers a narrower question -- how sensitive is this
    node's predicted logit to each of its own input features -- and deliberately does not attempt
    to attribute across the neighbourhood. Attributing to *neighbours* would require something
    like GNNExplainer, which is a substantial piece of machinery whose output is hard to defend
    in an adverse-action context.

    The interview-honest position: the GNN is explainable at the level of "which of your own
    attributes moved the score", and the cohort contribution is legible through the *graph
    features* (:mod:`credit_risk.graph.features`) rather than through the network. That is one
    concrete reason the cohort-feature arm of Layer B is worth having even if the GNN scores
    slightly better.
    """
    import torch

    model = trainer.model
    if model is None:
        raise RuntimeError("explain_gnn_gradients called before fit")
    model.eval()

    features = trainer._tensors["features"]
    index = trainer._tensors["index"]
    mask = trainer._tensors["mask"]

    attributions = []
    for start in range(0, len(nodes), batch_size):
        batch = torch.as_tensor(
            np.asarray(nodes[start : start + batch_size]), dtype=torch.long, device=trainer.device
        )
        perturbable = features.clone().requires_grad_(True)
        logits = model(perturbable, batch, index, mask)
        logits.sum().backward()
        grads = perturbable.grad[batch]
        attributions.append((grads * features[batch]).detach().cpu().numpy())

    values = np.concatenate(attributions) if attributions else np.zeros((0, len(feature_names)))
    return ShapResult(
        values=values,
        feature_names=list(feature_names),
        base_value=float("nan"),
    )


def save_importance(result: ShapResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.global_importance().to_csv(path, index=False)
    return path


def compare_importance(
    results: dict[str, ShapResult], top_k: int = 20
) -> pd.DataFrame:
    """Side-by-side global importance ranking across models.

    Useful as a sanity check rather than a metric: if two models with similar AUC rank their
    drivers completely differently, at least one of them is fitting noise, and that is worth
    knowing before either is trusted with a decision.
    """
    frames = []
    for name, result in results.items():
        frame = result.global_importance().head(top_k)[["feature", "mean_abs_shap"]]
        frame = frame.assign(model=name, rank=np.arange(1, len(frame) + 1))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.pivot_table(
        index="feature", columns="model", values="rank", aggfunc="min"
    ).sort_values(by=list(results.keys())[0])


def sample_explanations(
    shap_result: ShapResult,
    pd_estimates: np.ndarray,
    decisions: np.ndarray,
    exposures: np.ndarray,
    n: int = 5,
    seed: int = 0,
) -> list:
    """Produce human-readable explanations for a stratified sample of decisions."""
    from credit_risk.explainability.reasons import explain_row

    rng = np.random.default_rng(seed)
    n_rows = len(shap_result.values)
    # Stratify across the risk distribution so the sample is not all low-risk approvals.
    order = np.argsort(pd_estimates[:n_rows])
    picks = [order[int(q * (n_rows - 1))] for q in np.linspace(0.05, 0.95, n)]
    picks = [int(p) for p in picks]
    rng.shuffle(picks)

    frame = shap_result.frame
    out = []
    for i in picks:
        values = frame.iloc[i] if frame is not None else pd.Series(dtype=float)
        out.append(
            explain_row(
                shap_values=shap_result.values[i],
                feature_names=shap_result.feature_names,
                feature_values=values,
                pd_estimate=float(pd_estimates[i]),
                decision=str(decisions[i]),
                exposure=float(exposures[i]),
                baseline_pd=float(1 / (1 + np.exp(-shap_result.base_value)))
                if np.isfinite(shap_result.base_value)
                else float("nan"),
            )
        )
    return out
