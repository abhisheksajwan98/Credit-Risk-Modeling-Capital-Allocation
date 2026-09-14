#!/usr/bin/env python
"""Fetch the LendingClub extract from Kaggle, or generate a synthetic stand-in.

    python scripts/download_data.py --config configs/data/lending_club.yaml
    python scripts/download_data.py --synthetic --n-loans 200000

Credentials come from ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``. See
docs/DATASET.md section 2 for how to create the token.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.data.download import (  # noqa: E402
    ACCEPTED_FILE,
    REJECTED_FILE,
    KaggleAuthError,
    download_lending_club,
    verify_raw,
)
from credit_risk.data.synthetic import SyntheticSpec, write_synthetic_raw  # noqa: E402
from credit_risk.utils.cli import base_parser, resolve  # noqa: E402
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.download")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/data/lending_club.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a schema-faithful synthetic extract instead of downloading.",
    )
    parser.add_argument("--n-loans", type=int, default=None, help="Synthetic rows to generate.")
    parser.add_argument("--force", action="store_true", help="Re-download even if checksums match.")
    parser.add_argument(
        "--with-rejected",
        action="store_true",
        help="Also fetch the 27M-row rejected-applications file (selection-bias analysis only).",
    )
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)

    if args.synthetic:
        synth = config.get("synthetic", {})
        spec = SyntheticSpec(
            n_loans=args.n_loans or int(synth.get("n_loans", 60_000)),
            seed=args.seed or int(synth.get("seed", seed)),
            zip_effect_sd=float(synth.get("zip_effect_sd", 0.18)),
            emp_effect_sd=float(synth.get("emp_effect_sd", 0.12)),
            vintage_drift_per_year=float(synth.get("vintage_drift_per_year", 0.10)),
        )
        path = write_synthetic_raw(paths.data_raw, spec)
        LOG.warning(
            "SYNTHETIC data written to %s. Every downstream result will be stamped "
            "'synthetic' and must not be reported as a finding.",
            path.name,
        )
        return 0

    dataset = config.get("dataset", {})
    files = [dataset.get("files", {}).get("accepted", ACCEPTED_FILE)]
    if args.with_rejected or bool(dataset.get("download_rejected", False)):
        files.append(dataset.get("files", {}).get("rejected", REJECTED_FILE))

    try:
        checksums = download_lending_club(
            paths.data_raw,
            files=tuple(files),
            dataset=str(dataset.get("kaggle_ref", "wordsforthewise/lending-club")),
            force=args.force,
        )
    except KaggleAuthError as exc:
        LOG.error("%s", exc)
        return 2

    ok = verify_raw(paths.data_raw, tuple(files))
    LOG.info("checksum verification: %s", "PASSED" if ok else "FAILED")
    for name, digest in checksums.items():
        LOG.info("  %-40s %s", name, digest[:32])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
