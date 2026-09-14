"""Tabular autoencoder, used as an unsupervised "is this borrower unusual?" detector.

The idea is deliberately narrow. The autoencoder never sees the target. It learns to reconstruct
the *training* population's feature distribution, and a borrower it reconstructs badly is one that
population does not describe well. Reconstruction error is therefore a measure of **distance from
the data the PD model was fitted on** -- which is a reasonable prior for "the PD model's output
here is less trustworthy", and is a claim that has to be tested rather than assumed.

Scope limits, stated because they bound the interpretation:

* Only **numeric** columns are encoded. Categorical features are one-hot encoded first by the
  caller if they are wanted; passing a frame with pandas ``category`` columns silently drops them.
  A borrower unusual only in their loan *purpose* would not be flagged.
* Standardisation uses **training** means and standard deviations. Without this the MSE is
  dominated by whichever columns have the largest units (``annual_inc``, ``revol_bal``), and the
  "anomaly score" degenerates into "has a big number somewhere".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.utils.runtime import get_logger, resolve_device

LOG = get_logger("models.autoencoder")

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in torch-free installs
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


@dataclass
class TabularAutoencoderConfig:
    """Configuration. A dataclass, like every other config in this project.

    ``hidden_dims`` uses ``field(default_factory=...)`` rather than a bare list literal: a mutable
    default argument is shared across every instance and is a standard Python defect.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    patience: int = 10
    dropout: float = 0.1
    device: str = "auto"
    seed: int = 42


if TORCH_AVAILABLE:

    class _AutoencoderNet(nn.Module):
        def __init__(self, input_dim: int, config: TabularAutoencoderConfig):
            super().__init__()

            encoder: list[nn.Module] = []
            in_dim = input_dim
            for hidden in config.hidden_dims:
                encoder += [
                    nn.Linear(in_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(config.dropout),
                ]
                in_dim = hidden
            encoder.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder)

            decoder: list[nn.Module] = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
                decoder += [
                    nn.Linear(in_dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.LeakyReLU(),
                    nn.Dropout(config.dropout),
                ]
                in_dim = hidden
            decoder.append(nn.Linear(in_dim, input_dim))
            self.decoder = nn.Sequential(*decoder)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            latent = self.encoder(x)
            return latent, self.decoder(latent)


class TabularAutoencoder:
    """Fit on train, score anywhere."""

    def __init__(self, input_dim: int, config: TabularAutoencoderConfig | None = None) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the autoencoder (EXP08).")
        self.config = config or TabularAutoencoderConfig()
        self.device = resolve_device(self.config.device)
        torch.manual_seed(self.config.seed)
        self.model = _AutoencoderNet(input_dim, self.config).to(self.device)
        self.input_dim = input_dim
        self.feature_names: list[str] | None = None
        self.feature_means: pd.Series | None = None
        self.feature_stds: pd.Series | None = None

    # -- preparation -------------------------------------------------------
    def _prepare(self, X: pd.DataFrame, fit_stats: bool = False) -> "torch.Tensor":
        numeric = X.select_dtypes(include=[np.number])

        if fit_stats:
            self.feature_names = list(numeric.columns)
            self.feature_means = numeric.mean()
            # A zero-variance column would divide by zero; it also carries no information, so a
            # standard deviation of 1 leaves it as a constant the network trivially reproduces.
            self.feature_stds = numeric.std().replace(0.0, 1.0).fillna(1.0)
        elif self.feature_names is None:
            raise RuntimeError("TabularAutoencoder scored before fit")
        else:
            # Column order must match training exactly, or the network reconstructs the wrong
            # feature in each position and the anomaly score is meaningless.
            missing = set(self.feature_names) - set(numeric.columns)
            if missing:
                raise ValueError(f"columns missing at scoring time: {sorted(missing)}")
            numeric = numeric[self.feature_names]

        standardised = ((numeric - self.feature_means) / self.feature_stds).fillna(0.0)
        return torch.tensor(standardised.to_numpy(dtype=np.float32), dtype=torch.float32)

    # -- training ----------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, X_valid: pd.DataFrame) -> TabularAutoencoder:
        train_tensor = self._prepare(X_train, fit_stats=True)
        valid_tensor = self._prepare(X_valid, fit_stats=False)

        if train_tensor.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim={self.input_dim} but {train_tensor.shape[1]} numeric columns were "
                "supplied; construct the autoencoder with the numeric column count"
            )

        generator = torch.Generator().manual_seed(self.config.seed)
        # drop_last: BatchNorm1d raises on a batch of size 1 in training mode, which happens
        # whenever len(train) % batch_size == 1. Dropping the remainder is simpler and costs at
        # most one batch per epoch.
        train_loader = DataLoader(
            TensorDataset(train_tensor), batch_size=self.config.batch_size,
            shuffle=True, drop_last=True, generator=generator,
        )
        valid_loader = DataLoader(
            TensorDataset(valid_tensor), batch_size=self.config.batch_size, shuffle=False
        )

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        best_loss, best_state, stale = float("inf"), None, 0
        LOG.info(
            "training autoencoder on %s | %s train / %s valid rows | %d features",
            self.device, f"{len(X_train):,}", f"{len(X_valid):,}", train_tensor.shape[1],
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            total, seen = 0.0, 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                optimiser.zero_grad(set_to_none=True)
                _, reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimiser.step()
                total += float(loss.item()) * batch.size(0)
                seen += batch.size(0)
            train_loss = total / max(seen, 1)

            self.model.eval()
            total, seen = 0.0, 0
            with torch.no_grad():
                for (batch,) in valid_loader:
                    batch = batch.to(self.device)
                    _, reconstructed = self.model(batch)
                    total += float(criterion(reconstructed, batch).item()) * batch.size(0)
                    seen += batch.size(0)
            valid_loss = total / max(seen, 1)

            improved = valid_loss < best_loss
            if improved:
                best_loss, stale = valid_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
            LOG.info(
                "epoch %3d | train %.5f | valid %.5f%s", epoch, train_loss, valid_loss,
                "  *" if improved else "",
            )
            if stale >= self.config.patience:
                LOG.info("early stopping at epoch %d (best valid %.5f)", epoch, best_loss)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    # -- scoring -----------------------------------------------------------
    def score_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """Mean squared reconstruction error per row, in standardised feature units."""
        self.model.eval()
        tensor = self._prepare(X, fit_stats=False)
        loader = DataLoader(
            TensorDataset(tensor), batch_size=self.config.batch_size * 4, shuffle=False
        )

        errors = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                _, reconstructed = self.model(batch)
                errors.append(((reconstructed - batch) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(errors)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "input_dim": self.input_dim,
                "feature_names": self.feature_names,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
            },
            path,
        )
        return path
