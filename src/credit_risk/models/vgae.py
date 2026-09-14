"""Variational graph autoencoder over the time-respecting cohort graph.

What it is for
--------------
The GraphSAGE model in :mod:`credit_risk.models.gnn` asks "does the neighbourhood help predict
default?". This asks a different, unsupervised question: **how well is this borrower's
neighbourhood explained at all?** A VGAE reconstructs the graph's edges from latent node codes and
carries a per-node posterior variance. A borrower whose neighbourhood the model cannot reconstruct
confidently has a large sigma, and the hypothesis under test is that such borrowers are ones whose
PD is less trustworthy.

The model never sees the target, and every edge it consumes is strictly backward-looking, so it
inherits the leakage guarantees of the graph construction.

Two implementation notes that matter for interpreting a null result
-------------------------------------------------------------------
* **Early stopping on held-out link prediction.** Without it, "the VGAE found no signal" and "the
  VGAE was not trained long enough" are indistinguishable. A validation split of nodes is held out
  and the reconstruction loss on it drives early stopping.
* **One encode per step.** The obvious implementation encodes the batch, then re-encodes every
  positive neighbour, then re-encodes every negative sample -- roughly 40x the necessary work at
  fan-out 10 with two relations. Here the union of required nodes is encoded once and indexed
  into, which is both faster and numerically identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from credit_risk.utils.runtime import get_logger, resolve_device

LOG = get_logger("models.vgae")

try:
    import torch
    from torch import nn

    from credit_risk.models.gnn import GNNConfig, SAGELayer

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]
    GNNConfig = object  # type: ignore[assignment,misc]


if TORCH_AVAILABLE:

    @dataclass
    class VGAEConfig(GNNConfig):
        latent_dim: int = 16
        #: Weight on the KL term. Small by design: the objective here is a usable reconstruction
        #: and a comparable per-node sigma, not a generative model with a calibrated prior.
        kl_weight: float = 1e-4
        n_negative_samples: int = 1
        #: Fraction of training nodes held out to drive early stopping.
        valid_fraction: float = 0.1

    class VGAE(nn.Module):
        """GraphSAGE encoder producing a Gaussian posterior per node."""

        def __init__(self, in_dim: int, n_relations: int, config: VGAEConfig):
            super().__init__()
            self.config = config
            self.n_relations = n_relations

            # `n_layers - 1` shared layers, then a split into mu and log-sigma heads.
            dims = [in_dim] + [config.hidden_dim] * (config.n_layers - 1)
            self.shared_layers = nn.ModuleList(
                [
                    SAGELayer(dims[i], dims[i + 1], n_relations, config.dropout)
                    for i in range(config.n_layers - 1)
                ]
            )
            last = dims[-1]
            self.mu_layer = SAGELayer(last, config.latent_dim, n_relations, config.dropout)
            self.logstd_layer = SAGELayer(last, config.latent_dim, n_relations, config.dropout)

        def encode(
            self,
            features: "torch.Tensor",
            batch_nodes: "torch.Tensor",
            neighbour_index: "torch.Tensor",
            neighbour_mask: "torch.Tensor",
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            fan = self.config.fanout
            k1 = min(fan[0], neighbour_index.shape[2])
            idx1 = neighbour_index[batch_nodes][:, :, :k1]
            msk1 = neighbour_mask[batch_nodes][:, :, :k1]

            if not self.shared_layers:
                h_self = features[batch_nodes]
                h_neighbours = features[idx1]
            else:
                k2 = min(fan[1] if len(fan) > 1 else k1, neighbour_index.shape[2])
                flat = idx1.reshape(-1)
                idx2 = neighbour_index[flat][:, :, :k2]
                msk2 = neighbour_mask[flat][:, :, :k2] & msk1.reshape(-1)[:, None, None]

                # Every shared layer is applied, not just the first. The original implementation
                # built `n_layers - 1` layers and used only `shared_layers[0]`, silently ignoring
                # the rest whenever n_layers > 2.
                h_neighbours = features[flat]
                h_self = features[batch_nodes]
                for layer in self.shared_layers:
                    h_neighbours = torch.relu(layer(h_neighbours, features[idx2], msk2))
                    h_self = torch.relu(layer(h_self, features[idx1], msk1))
                h_neighbours = h_neighbours.reshape(idx1.shape[0], idx1.shape[1], k1, -1)

            mu = self.mu_layer(h_self, h_neighbours, msk1)
            logstd = self.logstd_layer(h_self, h_neighbours, msk1)
            # An unbounded log-sigma lets the posterior collapse or explode; clamping keeps the
            # reported sigma comparable across nodes, which is the quantity the experiment uses.
            return mu, logstd.clamp(-6.0, 2.0)

        def reparameterise(self, mu: "torch.Tensor", logstd: "torch.Tensor") -> "torch.Tensor":
            if not self.training:
                return mu
            return mu + torch.randn_like(mu) * torch.exp(logstd)

        def forward(self, features, batch_nodes, neighbour_index, neighbour_mask):
            mu, logstd = self.encode(features, batch_nodes, neighbour_index, neighbour_mask)
            return self.reparameterise(mu, logstd), mu, logstd


class VGAETrainer:
    """Trains the VGAE by link reconstruction with negative sampling."""

    def __init__(self, config: "VGAEConfig | None" = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the VGAE layer (EXP09).")
        self.config = config or VGAEConfig()
        self.device = resolve_device(self.config.device)
        self.model: VGAE | None = None
        self._tensors: dict[str, Any] = {}
        self.history: list[dict[str, float]] = []

    def _to_device(
        self, features: np.ndarray, neighbour_index: np.ndarray, neighbour_mask: np.ndarray
    ) -> None:
        if self._tensors:
            return  # already resident; re-uploading a 100k x 130 matrix per call is pure waste
        self._tensors = {
            "features": torch.as_tensor(features, dtype=torch.float32, device=self.device),
            "index": torch.as_tensor(
                np.where(neighbour_index < 0, 0, neighbour_index).astype(np.int64),
                dtype=torch.long, device=self.device,
            ),
            "mask": torch.as_tensor(neighbour_mask, dtype=torch.bool, device=self.device),
        }

    # -- loss ---------------------------------------------------------------
    def _batch_loss(self, batch: "torch.Tensor", n_nodes: int) -> tuple["torch.Tensor", float]:
        """Reconstruction + KL for one batch, using a single encode pass.

        The batch node, its sampled positive neighbours and an equal number of uniformly-drawn
        negatives are gathered into one unique node list, encoded once, and indexed back out.
        """
        index, mask = self._tensors["index"], self._tensors["mask"]
        features = self._tensors["features"]
        k1 = min(self.config.fanout[0], index.shape[2])

        pos_idx = index[batch][:, :, :k1].reshape(-1)
        pos_msk = mask[batch][:, :, :k1].reshape(-1)
        valid = pos_msk.nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            zero = torch.zeros((), device=self.device, requires_grad=True)
            return zero, 0.0

        # Source node for each surviving positive edge.
        n_slots = self.n_relations * k1
        source = batch[valid // n_slots]
        target = pos_idx[valid]
        negative = torch.randint(0, n_nodes, (valid.numel(),), device=self.device)

        unique, inverse = torch.unique(
            torch.cat([source, target, negative]), return_inverse=True
        )
        z, mu, logstd = self.model(features, unique, index, mask)

        n = valid.numel()
        z_source = z[inverse[:n]]
        z_target = z[inverse[n : 2 * n]]
        z_negative = z[inverse[2 * n :]]

        pos_loss = -nn.functional.logsigmoid((z_source * z_target).sum(-1)).mean()
        neg_loss = -nn.functional.logsigmoid(-(z_source * z_negative).sum(-1)).mean()

        # KL( N(mu, sigma^2) || N(0, I) ) = -0.5 * sum(1 + log sigma^2 - mu^2 - sigma^2),
        # with logstd = log sigma so log sigma^2 = 2 * logstd and sigma^2 = exp(2 * logstd).
        kl = -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1)
        )
        return pos_loss + neg_loss + self.config.kl_weight * kl, float(kl.item())

    # -- training -----------------------------------------------------------
    def fit(
        self,
        features: np.ndarray,
        neighbour_index: np.ndarray,
        neighbour_mask: np.ndarray,
        train_nodes: np.ndarray,
    ) -> VGAETrainer:
        self._to_device(features, neighbour_index, neighbour_mask)
        self.n_relations = neighbour_index.shape[1]
        n_nodes = features.shape[0]

        torch.manual_seed(self.config.seed)
        self.model = VGAE(features.shape[1], self.n_relations, self.config).to(self.device)
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        rng = np.random.default_rng(self.config.seed)
        nodes = np.asarray(train_nodes)
        shuffled = rng.permutation(len(nodes))
        n_valid = max(int(len(nodes) * self.config.valid_fraction), 1)
        valid_nodes = nodes[shuffled[:n_valid]]
        fit_nodes = nodes[shuffled[n_valid:]]

        LOG.info(
            "training VGAE on %s | %s fit / %s valid nodes | latent dim %d",
            self.device, f"{len(fit_nodes):,}", f"{len(valid_nodes):,}", self.config.latent_dim,
        )

        best_loss, best_state, stale = float("inf"), None, 0
        batch_size = self.config.batch_size

        for epoch in range(self.config.max_epochs):
            self.model.train()
            order = rng.permutation(len(fit_nodes))
            total, seen = 0.0, 0
            for start in range(0, len(order), batch_size):
                batch = torch.as_tensor(
                    fit_nodes[order[start : start + batch_size]],
                    dtype=torch.long, device=self.device,
                )
                optimiser.zero_grad(set_to_none=True)
                loss, _ = self._batch_loss(batch, n_nodes)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimiser.step()
                total += float(loss.item()) * len(batch)
                seen += len(batch)
            train_loss = total / max(seen, 1)

            self.model.eval()
            total, seen = 0.0, 0
            with torch.no_grad():
                for start in range(0, len(valid_nodes), batch_size):
                    batch = torch.as_tensor(
                        valid_nodes[start : start + batch_size],
                        dtype=torch.long, device=self.device,
                    )
                    loss, _ = self._batch_loss(batch, n_nodes)
                    total += float(loss.item()) * len(batch)
                    seen += len(batch)
            valid_loss = total / max(seen, 1)

            improved = valid_loss < best_loss
            if improved:
                best_loss, stale = valid_loss, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:
                stale += 1
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
            )
            LOG.info(
                "epoch %2d | train %.4f | valid %.4f%s", epoch, train_loss, valid_loss,
                "  *" if improved else "",
            )
            if stale >= self.config.patience:
                LOG.info("early stopping at epoch %d (best valid %.4f)", epoch, best_loss)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_valid_loss = best_loss
        return self

    # -- inference ----------------------------------------------------------
    def embed(
        self,
        features: np.ndarray,
        neighbour_index: np.ndarray,
        neighbour_mask: np.ndarray,
        nodes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mu, sigma)`` for the given nodes."""
        if self.model is None:
            raise RuntimeError("VGAETrainer.embed called before fit")
        self._to_device(features, neighbour_index, neighbour_mask)
        self.model.eval()

        batch_size = self.config.batch_size * 2
        mus, sigmas = [], []
        with torch.no_grad():
            for start in range(0, len(nodes), batch_size):
                batch = torch.as_tensor(
                    np.asarray(nodes[start : start + batch_size]),
                    dtype=torch.long, device=self.device,
                )
                mu, logstd = self.model.encode(
                    self._tensors["features"], batch,
                    self._tensors["index"], self._tensors["mask"],
                )
                mus.append(mu.cpu().numpy())
                sigmas.append(torch.exp(logstd).cpu().numpy())
        return np.concatenate(mus), np.concatenate(sigmas)

    def save(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("save called before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config}, path)
        return path
