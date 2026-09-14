"""Shared command-line plumbing, so every script has the same shape.

Each script accepts ``--config``, ``--seed``, ``--set key.path=value`` and ``--out-name``, and
writes a run manifest beside its output. Keeping this in one place means "how do I reproduce this
number?" has one answer rather than seven.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.utils.config import Config, Paths, load_config, parse_overrides
from credit_risk.utils.runtime import (
    RunManifest,
    configure_threads,
    get_logger,
    set_seed,
)

LOG = get_logger("cli")


def base_parser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="YAML config under configs/. Relative paths resolve from the repository root.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. --set model.num_leaves=64. Repeatable.",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Name used for this run's outputs. Defaults to the config stem.",
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip figure generation (headless / fast runs)."
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Cap numeric-library threads so a long run leaves the machine usable. "
             "Also settable via CREDIT_RISK_MAX_THREADS or CREDIT_RISK_THREAD_FRACTION.",
    )
    return parser


def resolve(args: argparse.Namespace) -> tuple[Config, Paths, str, int]:
    """Load the config with overrides applied and seed everything."""
    # Before anything imports a BLAS-backed array: thread pools are read at import time.
    configure_threads(getattr(args, "max_threads", None))

    if not args.config:
        raise SystemExit("--config is required")
    config = load_config(args.config, parse_overrides(args.overrides))
    if args.seed is not None:
        config = config.merged({"seed": args.seed})
    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_name = args.out_name or Path(args.config).stem
    paths = Paths()
    LOG.info("run %r | config %s | seed %d", run_name, args.config, seed)
    return config, paths, run_name, seed


def write_manifest(
    run_name: str,
    config: Config,
    seed: int,
    paths: Paths,
    data_source: str,
    device: str = "cpu",
    notes: dict[str, Any] | None = None,
) -> Path:
    manifest = RunManifest(
        run_name=run_name,
        seed=seed,
        config=config.to_dict(),
        data_source=data_source,
        device=device,
        notes=notes or {},
    )
    return manifest.save(paths.results_reports / f"{run_name}_manifest.json")


def save_table(frame: pd.DataFrame, paths: Paths, name: str, index: bool = True) -> Path:
    path = paths.results_tables / f"{name}.csv"
    frame.to_csv(path, index=index)
    LOG.info("wrote %s (%d rows)", path.name, len(frame))
    return path


def save_json(payload: dict[str, Any], paths: Paths, name: str) -> Path:
    path = paths.results_tables / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    LOG.info("wrote %s", path.name)
    return path


def write_report(lines: list[str], paths: Paths, name: str, data_source: str) -> Path:
    """Write a short markdown verdict, stamped with the data provenance.

    The stamp is not decoration. A reader must be able to tell at a glance whether a number came
    from real LendingClub data or from the synthetic generator, because the second kind is a
    pipeline check and not a finding.
    """
    banner = (
        "> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing "
        "about consumer credit. Re-run against the real extract before quoting anything here.\n"
        if data_source == "synthetic"
        else f"> Computed on **{data_source}** data.\n"
    )
    path = paths.results_reports / f"{name}.md"
    path.write_text(banner + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s", path.name)
    return path


def load_prepared(paths: Paths, filename: str = "loans.parquet") -> pd.DataFrame:
    """Load the prepared modelling table, with a useful error if it is missing."""
    from credit_risk.data.prepare import read_processed

    path = paths.data_processed / filename
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run:\n"
            f"  python scripts/prepare_data.py --config configs/data/lending_club.yaml"
        )
    return read_processed(path)


def detect_source(paths: Paths) -> str:
    from credit_risk.data.download import detect_data_source

    return detect_data_source(paths.data_raw)
