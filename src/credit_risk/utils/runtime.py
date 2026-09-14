"""Seeding, logging, device selection and run manifests.

The manifest is the reproducibility contract in code form: every script writes one next to
its output recording the config, the seed, the library versions, the git commit and -- critically --
whether the run used real or synthetic data. A result whose manifest says ``synthetic`` is never
reported as a finding.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "credit_risk"
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent -- safe to call from every module."""
    root = logging.getLogger(LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
        root.setLevel(os.environ.get("CREDIT_RISK_LOGLEVEL", "INFO").upper())
        root.propagate = False
    return root if name is None else root.getChild(name)


def set_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Seed Python, NumPy and (if installed) PyTorch. Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def configure_threads(max_threads: int | None = None, fraction: float | None = None) -> int:
    """Cap the thread pools every numeric library spins up.

    By default these libraries each claim every core, which makes a training run monopolise the
    machine. ``CREDIT_RISK_MAX_THREADS`` (absolute) or ``CREDIT_RISK_THREAD_FRACTION`` (a share of
    the available cores) bound them, so a long run can be left going while the machine stays
    usable.

    Must be called **before** numpy/LightGBM/torch read their environment, which is why
    ``utils.cli.resolve`` calls it first thing. Returns the cap applied.
    """
    total = os.cpu_count() or 1

    if max_threads is None:
        env_absolute = os.environ.get("CREDIT_RISK_MAX_THREADS")
        env_fraction = os.environ.get("CREDIT_RISK_THREAD_FRACTION")
        if env_absolute:
            max_threads = int(env_absolute)
        elif fraction is not None or env_fraction:
            share = fraction if fraction is not None else float(env_fraction)
            max_threads = max(1, int(round(total * share)))
        else:
            return total  # no cap requested; leave the libraries alone

    max_threads = max(1, min(int(max_threads), total))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = str(max_threads)

    try:
        import torch

        torch.set_num_threads(max_threads)
    except ImportError:
        pass

    get_logger("runtime").info(
        "thread cap: %d of %d cores (set CREDIT_RISK_MAX_THREADS to change)", max_threads, total
    )
    return max_threads


def default_n_jobs() -> int:
    """Worker count for libraries that take an explicit ``n_jobs`` (LightGBM, sklearn).

    Environment variables alone do not bound LightGBM, which defaults to all cores regardless.
    """
    capped = os.environ.get("OMP_NUM_THREADS")
    return int(capped) if capped else (os.cpu_count() or 1)


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string.

    ``auto`` uses CUDA when a GPU is actually usable and CPU otherwise. Everything in this
    project is written to run correctly on both -- the GNN is the only component where the
    difference is felt, and even there the graph is small enough for CPU to finish.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if preference == "cuda":
            raise RuntimeError("device='cuda' requested but torch is not installed") from None
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    if preference == "cuda":
        if not cuda_ok:
            raise RuntimeError("device='cuda' requested but torch reports no usable CUDA device")
        return "cuda"
    return "cuda" if cuda_ok else "cpu"


def describe_device(device: str) -> str:
    """Human-readable device description for logs and manifests."""
    if device != "cuda":
        return f"cpu ({platform.processor() or platform.machine()})"
    try:
        import torch

        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"cuda:{idx} ({name}, {total_gb:.1f} GiB)"
    except Exception:  # pragma: no cover - diagnostics only
        return "cuda (unavailable for description)"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # pragma: no cover - git may be absent
        return None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "shap", "torch", "networkx"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


@dataclass
class RunManifest:
    """Everything needed to say what a result actually is."""

    run_name: str
    seed: int
    config: dict[str, Any]
    data_source: str = "unknown"  # "real" | "synthetic" | "unknown"
    device: str = "cpu"
    notes: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform_: str = field(default_factory=platform.platform)
    git_commit: str | None = field(default_factory=_git_commit)
    packages: dict[str, str] = field(default_factory=_package_versions)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        if self.data_source == "synthetic":
            get_logger("manifest").warning(
                "Run %r used SYNTHETIC data. Its numbers are pipeline checks, not findings.",
                self.run_name,
            )
        return path

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as fh:
            return json.load(fh)
