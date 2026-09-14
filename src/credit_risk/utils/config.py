"""Configuration loading and project path resolution.

Deliberately small: YAML in, attribute-accessible mapping out. No Hydra, no plugin system.
Every script in ``scripts/`` takes ``--config path/to.yaml`` and an optional ``--seed``, and
every value a run depended on is written back out beside the results so a run can be
reconstructed from its own output.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root.

    Resolved from this file's location (``src/credit_risk/utils/config.py`` -> up four levels)
    so that scripts work regardless of the working directory they are launched from.
    An explicit ``CREDIT_RISK_ROOT`` environment variable wins if set.
    """
    env = os.environ.get("CREDIT_RISK_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


class Paths:
    """Canonical locations. Directories are created on access, not at import time."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else project_root()

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data_raw(self) -> Path:
        return self._mk(self.root / "data" / "raw")

    @property
    def data_interim(self) -> Path:
        return self._mk(self.root / "data" / "interim")

    @property
    def data_processed(self) -> Path:
        return self._mk(self.root / "data" / "processed")

    @property
    def artifacts(self) -> Path:
        return self._mk(self.root / "artifacts")

    @property
    def results_tables(self) -> Path:
        return self._mk(self.root / "results" / "tables")

    @property
    def results_figures(self) -> Path:
        return self._mk(self.root / "results" / "figures")

    @property
    def results_reports(self) -> Path:
        return self._mk(self.root / "results" / "reports")

    @staticmethod
    def _mk(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Paths(root={self.root!s})"


class Config(Mapping):
    """Read-only nested mapping with attribute access.

    ``cfg.model.learning_rate`` and ``cfg["model"]["learning_rate"]`` are equivalent.
    Immutability is intentional: a config that mutates mid-run cannot be trusted to
    describe the run afterwards. Use :meth:`merged` to derive a variant.
    """

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = {}
        for key, value in dict(data or {}).items():
            self._data[key] = Config(value) if isinstance(value, Mapping) else value

    # -- Mapping protocol -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key {key!r}; available keys: {sorted(self._data)}"
            ) from exc

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self.to_dict()!r})"

    # -- helpers ----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            out[key] = value.to_dict() if isinstance(value, Config) else copy.deepcopy(value)
        return out

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value by dotted path, e.g. ``cfg.get_path("model.params.num_leaves")``."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Config) and part in node:
                node = node[part]
            elif isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def merged(self, overrides: Mapping[str, Any]) -> Config:
        """Return a new Config with ``overrides`` deep-merged over this one."""
        return Config(_deep_merge(self.to_dict(), dict(overrides)))


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config, resolving a single optional ``_base_`` inheritance link.

    ``_base_`` is relative to the including file's directory. One level of inheritance is
    supported deliberately -- deep config hierarchies become impossible to reason about,
    and the point of these files is that a reader can see the whole run at a glance.
    """
    path = Path(path)
    if not path.is_absolute():
        candidate = project_root() / path
        path = candidate if candidate.exists() else path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"config {path} must contain a mapping at the top level, got {type(raw)}")

    raw = dict(raw)
    base_ref = raw.pop("_base_", None)
    if base_ref is not None:
        base_cfg = load_config(path.parent / str(base_ref))
        raw = _deep_merge(base_cfg.to_dict(), raw)

    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw)


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Turn ``["model.num_leaves=64", "seed=7"]`` into a nested override dict.

    Values are parsed as YAML scalars, so ``true``, ``3``, ``0.1`` and ``null`` arrive typed.
    """
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        dotted, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value)
        node = out
        parts = dotted.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out
