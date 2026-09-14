"""GraphSAGE, written directly in PyTorch.

Why not torch-geometric
-----------------------
Two reasons, in order of importance. First, a hand-written sampler and aggregator is something that
can be explained line by line; ``NeighborLoader(...)`` is not, and "the library did it" is not an
answer to "how does message passing actually work?". Second, PyG's neighbour sampling depends on
``pyg-lib`` or ``torch-sparse``, whose Windows wheels are unreliable. The whole implementation below
is about 200 lines and has no dependency beyond ``torch``.

The mechanism
-------------
GraphSAGE learns to produce a node embedding by *aggregating over a sampled neighbourhood* rather
than by memorising a per-node vector. That is what makes it **inductive**: a borrower who was not
in the training graph still gets an embedding, because the model learned an aggregation function
rather than a lookup table. For credit scoring this is the only workable option -- every applicant
is new.

One layer, for node ``i``:

.. code-block:: text

    h_i' = sigma( W . [ h_i || mean_{j in N_1(i)} h_j || mean_{j in N_2(i)} h_j ] )

Concatenating the self-representation with the neighbourhood means (rather than averaging them
together) is what lets the model weigh "what this borrower looks like" against "what this
borrower's cohort looks like" instead of blurring the two. Each relation gets its own slice of the
weight matrix, so geography and employment are not forced to share a coefficient.

Stacking two layers gives each node a receptive field two hops wide: the borrower, their cohort,
and their cohort's cohort. More layers were not used -- at three hops on a cohort graph the
neighbourhood approaches the whole portfolio and every embedding converges to the same vector,
which is over-smoothing.

Leakage
-------
Every edge consumed here comes from :mod:`credit_risk.graph.construction` and is strictly
backward-looking. Node features are decision-time only and contain **no label information**, so
message passing moves *features* between borrowers, never outcomes. Cohort default rates, which do
carry label information, are gated separately and much more strictly in
:mod:`credit_risk.graph.features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from credit_risk.utils.runtime import get_logger, resolve_device

LOG = get_logger("models.gnn")

try:  # torch is an optional dependency; the tabular layers must work without it.
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


@dataclass
class GNNConfig:
    hidden_dim: int = 64
    embed_dim: int = 32
    n_layers: int = 2
    dropout: float = 0.2
    #: Neighbours sampled per relation at each hop, outermost hop last. Shrinking the second hop
    #: is standard practice: the two-hop neighbourhood grows as the product of the fan-outs, and
    #: distant neighbours contribute progressively less.
    fanout: tuple[int, ...] = (10, 5)
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 30
    patience: int = 5
    #: `l2_normalize` follows the GraphSAGE paper; it keeps embedding norms comparable across
    #: nodes with very different degrees, which matters here because ZIP cohorts are wildly uneven.
    l2_normalize: bool = True
    device: str = "auto"
    seed: int = 42
    #: Positive-class weight in the loss. **Off by default, deliberately.** Up-weighting defaults
    #: distorts the effective base rate in exactly the way resampling does, and the resulting
    #: scores are badly mis-calibrated (an observed/expected ratio near 3 in testing). Since these
    #: scores feed the decision layer, the default preserves the base rate and calibration is
    #: handled afterwards by `models.calibration`, the same as for LightGBM. Turn this on only to
    #: measure what it costs.
    balance_classes: bool = False


if TORCH_AVAILABLE:

    class SAGELayer(nn.Module):
        """One GraphSAGE layer with per-relation mean aggregation."""

        def __init__(self, in_dim: int, out_dim: int, n_relations: int, dropout: float = 0.0):
            super().__init__()
            self.n_relations = n_relations
            # (1 + R) blocks: the node itself plus one mean per relation.
            self.linear = nn.Linear(in_dim * (1 + n_relations), out_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            h_self: "torch.Tensor",  # (..., D)
            h_neighbours: "torch.Tensor",  # (..., R, K, D)
            mask: "torch.Tensor",  # (..., R, K)
        ) -> "torch.Tensor":
            # Masked mean over the K sampled neighbours, per relation. Nodes with no eligible
            # predecessor in a relation get a zero vector, which the concatenation keeps
            # distinguishable from "neighbours whose features happen to average to zero"
            # because the self-block is always present.
            weights = mask.unsqueeze(-1).to(h_neighbours.dtype)
            summed = (h_neighbours * weights).sum(dim=-2)
            counts = weights.sum(dim=-2).clamp(min=1.0)
            means = summed / counts  # (..., R, D)

            flat_means = means.flatten(start_dim=-2)  # (..., R*D)
            combined = torch.cat([h_self, flat_means], dim=-1)
            return self.linear(self.dropout(combined))

    class GraphSAGE(nn.Module):
        """Stacked GraphSAGE layers followed by an MLP head producing a default logit."""

        def __init__(self, in_dim: int, n_relations: int, config: GNNConfig):
            super().__init__()
            self.config = config
            self.n_relations = n_relations
            dims = [in_dim] + [config.hidden_dim] * (config.n_layers - 1) + [config.embed_dim]
            self.layers = nn.ModuleList(
                [
                    SAGELayer(dims[i], dims[i + 1], n_relations, config.dropout)
                    for i in range(config.n_layers)
                ]
            )
            self.head = nn.Sequential(
                nn.Linear(config.embed_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, 1),
            )

        def _activate(self, h: "torch.Tensor", last: bool) -> "torch.Tensor":
            if last:
                return nn.functional.normalize(h, p=2, dim=-1) if self.config.l2_normalize else h
            return torch.relu(h)

        def embed(
            self,
            features: "torch.Tensor",  # (n_nodes, D) full feature table on device
            batch_nodes: "torch.Tensor",  # (B,) int64
            neighbour_index: "torch.Tensor",  # (n_nodes, R, K) int64
            neighbour_mask: "torch.Tensor",  # (n_nodes, R, K) bool
        ) -> "torch.Tensor":
            """Compute embeddings for ``batch_nodes`` by explicit nested neighbour gathering.

            Written for two layers, which is what the config uses. The nesting is deliberately
            spelled out rather than looped: the shapes are the part worth being able to point at.
            """
            fan = self.config.fanout
            k1 = min(fan[0], neighbour_index.shape[2])

            # -- hop 1: neighbours of the batch --------------------------------
            idx1 = neighbour_index[batch_nodes][:, :, :k1]  # (B, R, K1)
            msk1 = neighbour_mask[batch_nodes][:, :, :k1]  # (B, R, K1)

            if len(self.layers) == 1:
                h = self.layers[0](features[batch_nodes], features[idx1], msk1)
                return self._activate(h, last=True)

            k2 = min(fan[1] if len(fan) > 1 else k1, neighbour_index.shape[2])

            # -- hop 2: neighbours of the hop-1 nodes --------------------------
            flat1 = idx1.reshape(-1)  # (B*R*K1,)
            idx2 = neighbour_index[flat1][:, :, :k2]  # (B*R*K1, R, K2)
            msk2 = neighbour_mask[flat1][:, :, :k2]
            # A padded hop-1 slot must not contribute a real hop-2 neighbourhood.
            msk2 = msk2 & msk1.reshape(-1)[:, None, None]

            # -- layer 1 on the hop-1 nodes ------------------------------------
            h1_neighbours = self.layers[0](features[flat1], features[idx2], msk2)
            h1_neighbours = self._activate(h1_neighbours, last=False)
            h1_neighbours = h1_neighbours.reshape(idx1.shape[0], idx1.shape[1], k1, -1)

            # -- layer 1 on the batch nodes themselves -------------------------
            h1_self = self.layers[0](features[batch_nodes], features[idx1], msk1)
            h1_self = self._activate(h1_self, last=False)

            # -- layer 2 -------------------------------------------------------
            h2 = self.layers[1](h1_self, h1_neighbours, msk1)
            return self._activate(h2, last=True)

        def forward(
            self,
            features: "torch.Tensor",
            batch_nodes: "torch.Tensor",
            neighbour_index: "torch.Tensor",
            neighbour_mask: "torch.Tensor",
        ) -> "torch.Tensor":
            return self.head(
                self.embed(features, batch_nodes, neighbour_index, neighbour_mask)
            ).squeeze(-1)


@dataclass
class GNNTrainResult:
    best_epoch: int
    best_valid_auc: float
    history: list[dict[str, float]] = field(default_factory=list)
    device: str = "cpu"
    notes: dict[str, Any] = field(default_factory=dict)


class GraphSAGETrainer:
    """Minibatch training loop with early stopping on validation AUC."""

    def __init__(self, config: GNNConfig | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for the GNN layer. Install it, or run the tabular "
                "experiments (EXP01-EXP03) which do not need it."
            )
        self.config = config or GNNConfig()
        self.device = resolve_device(self.config.device)
        self.model: GraphSAGE | None = None
        self.result: GNNTrainResult | None = None
        self._tensors: dict[str, Any] = {}

    def _to_device(
        self, features: np.ndarray, neighbour_index: np.ndarray, neighbour_mask: np.ndarray
    ) -> None:
        self._tensors = {
            "features": torch.as_tensor(features, dtype=torch.float32, device=self.device),
            "index": torch.as_tensor(
                np.where(neighbour_index < 0, 0, neighbour_index).astype(np.int64),
                dtype=torch.long,
                device=self.device,
            ),
            "mask": torch.as_tensor(neighbour_mask, dtype=torch.bool, device=self.device),
        }

    def fit(
        self,
        features: np.ndarray,
        neighbour_index: np.ndarray,
        neighbour_mask: np.ndarray,
        train_nodes: np.ndarray,
        train_labels: np.ndarray,
        valid_nodes: np.ndarray,
        valid_labels: np.ndarray,
    ) -> GraphSAGETrainer:
        from sklearn.metrics import roc_auc_score

        cfg = self.config
        torch.manual_seed(cfg.seed)
        self._to_device(features, neighbour_index, neighbour_mask)

        n_relations = neighbour_index.shape[1]
        self.model = GraphSAGE(features.shape[1], n_relations, cfg).to(self.device)
        optimiser = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        # Plain BCE by default. Weighting the positive class would distort the base rate just
        # as resampling does, and this model's scores feed the decision layer, which needs
        # probabilities rather than a ranking. Calibration is applied downstream instead.
        pos_weight = None
        if cfg.balance_classes:
            pos_weight = torch.tensor(
                [(len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1)],
                dtype=torch.float32,
                device=self.device,
            )
            LOG.warning(
                "balance_classes=True: scores will be mis-calibrated by construction and must "
                "be passed through models.calibration before any decision uses them."
            )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_nodes_t = torch.as_tensor(train_nodes, dtype=torch.long, device=self.device)
        train_labels_t = torch.as_tensor(train_labels, dtype=torch.float32, device=self.device)

        best_auc, best_epoch, best_state, stale = -np.inf, -1, None, 0
        history: list[dict[str, float]] = []
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)

        LOG.info(
            "training GraphSAGE on %s | %s train nodes, %d features, %d relations",
            self.device,
            f"{len(train_nodes):,}",
            features.shape[1],
            n_relations,
        )

        for epoch in range(cfg.max_epochs):
            self.model.train()
            perm = torch.randperm(len(train_nodes), generator=generator).to(self.device)
            total_loss, n_batches = 0.0, 0
            for start in range(0, len(perm), cfg.batch_size):
                sel = perm[start : start + cfg.batch_size]
                optimiser.zero_grad(set_to_none=True)
                logits = self.model(
                    self._tensors["features"],
                    train_nodes_t[sel],
                    self._tensors["index"],
                    self._tensors["mask"],
                )
                loss = criterion(logits, train_labels_t[sel])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimiser.step()
                total_loss += float(loss.item())
                n_batches += 1

            valid_scores = self.predict(valid_nodes)
            auc = float(roc_auc_score(valid_labels, valid_scores))
            history.append(
                {"epoch": epoch, "train_loss": total_loss / max(n_batches, 1), "valid_auc": auc}
            )
            LOG.info(
                "epoch %2d | loss %.5f | valid AUC %.5f%s",
                epoch,
                total_loss / max(n_batches, 1),
                auc,
                "  *" if auc > best_auc else "",
            )

            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    LOG.info("early stopping at epoch %d (best %d)", epoch, best_epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.result = GNNTrainResult(
            best_epoch=best_epoch,
            best_valid_auc=best_auc,
            history=history,
            device=self.device,
        )
        return self

    def predict(self, nodes: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Predicted default probability for the given node indices."""
        if self.model is None:
            raise RuntimeError("GraphSAGETrainer.predict called before fit")
        self.model.eval()
        batch_size = batch_size or (self.config.batch_size * 4)
        nodes_t = torch.as_tensor(np.asarray(nodes), dtype=torch.long, device=self.device)
        out = []
        with torch.no_grad():
            for start in range(0, len(nodes_t), batch_size):
                logits = self.model(
                    self._tensors["features"],
                    nodes_t[start : start + batch_size],
                    self._tensors["index"],
                    self._tensors["mask"],
                )
                out.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def embeddings(self, nodes: np.ndarray, batch_size: int | None = None) -> np.ndarray:
        """Learned node embeddings, for inspection or as features to another model."""
        if self.model is None:
            raise RuntimeError("embeddings requested before fit")
        self.model.eval()
        batch_size = batch_size or (self.config.batch_size * 4)
        nodes_t = torch.as_tensor(np.asarray(nodes), dtype=torch.long, device=self.device)
        out = []
        with torch.no_grad():
            for start in range(0, len(nodes_t), batch_size):
                emb = self.model.embed(
                    self._tensors["features"],
                    nodes_t[start : start + batch_size],
                    self._tensors["index"],
                    self._tensors["mask"],
                )
                out.append(emb.detach().cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, self.config.embed_dim))

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("save called before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.model.state_dict(), "config": self.config},
            path,
        )
        return path
