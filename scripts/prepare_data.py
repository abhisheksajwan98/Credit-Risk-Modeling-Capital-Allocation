#!/usr/bin/env python
"""Raw CSV -> the split-assigned modelling table, plus a measured data profile.

    python scripts/prepare_data.py --config configs/data/lending_club.yaml

Writes ``data/processed/loans.parquet`` and ``results/reports/data_profile.md``. The profile is
measured from the data rather than copied from documentation, which is the point -- several
widely-repeated "facts" about this dataset do not survive checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from credit_risk.data.prepare import prepare_loans, write_processed  # noqa: E402
from credit_risk.data.splits import SplitWindows, assert_temporal_ordering  # noqa: E402
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    resolve,
    save_json,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.prepare")


def _profile_lines(df: pd.DataFrame, report, windows: SplitWindows, source: str) -> list[str]:
    lines = [
        "# Data profile",
        "",
        f"Rows read from raw file: **{report.rows_read:,}**",
        f"After the {df['term_months'].dropna().iloc[0] if 'term_months' in df else 36}-month "
        f"term filter: **{report.rows_after_term_filter:,}**",
        f"After the issue-window filter: **{report.rows_after_window_filter:,}**",
        f"Final modelling table: **{len(df):,}** rows x {df.shape[1]} columns",
        "",
        "## Rows dropped",
        "",
        f"- malformed / non-numeric id: {report.rows_dropped_bad_id:,}",
        f"- non-terminal status inside a matured window: {report.rows_dropped_non_terminal:,}",
        f"- unmapped loan_status: {report.rows_dropped_unmapped_status:,}",
        "",
        "## Splits",
        "",
        "| split | window | rows | default rate |",
        "|---|---|---|---|",
    ]
    for name in ("history", "train", "valid", "test", "monitor"):
        start, end = getattr(windows, name)
        n = report.split_counts.get(name, 0)
        rate = report.default_rate_by_split.get(name)
        rate_text = f"{rate:.4f}" if rate is not None else "n/a (unlabelled by design)"
        lines.append(f"| {name} | {start} .. {end} | {n:,} | {rate_text} |")

    lines += ["", "## Default rate by vintage", "", "| year | default rate |", "|---|---|"]
    for year, rate in sorted(report.default_rate_by_vintage.items()):
        lines.append(f"| {year} | {rate:.4f} |")

    top_missing = sorted(report.missingness.items(), key=lambda kv: -kv[1])[:25]
    lines += ["", "## Highest missingness", "", "| column | missing |", "|---|---|"]
    lines += [f"| `{c}` | {v:.1%} |" for c, v in top_missing]

    if report.availability_shift:
        lines += [
            "",
            "## Availability shift (train -> test)",
            "",
            "Columns whose *missingness* moves between windows. These partly encode the",
            "origination date, so a model can learn 'is this populated?' as a vintage proxy.",
            "The feature builder's availability guard drops them; see docs/LEAKAGE.md section 5.",
            "",
            "| column | shift in missing rate |",
            "|---|---|",
        ]
        worst = sorted(report.availability_shift.items(), key=lambda kv: -abs(kv[1]))[:25]
        lines += [f"| `{c}` | {v:+.1%} |" for c, v in worst]

    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]

    lines += ["", f"Data source: **{source}**."]
    return lines


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/data/lending_club.yaml")
    parser.add_argument("--raw-file", default=None, help="Override the raw file path.")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)

    source = detect_source(paths)
    if source == "absent":
        raise SystemExit(
            "No raw data found in data/raw/. Run one of:\n"
            "  python scripts/download_data.py --config configs/data/lending_club.yaml\n"
            "  python scripts/download_data.py --synthetic"
        )

    dataset = config.get("dataset", {})
    raw_path = Path(args.raw_file) if args.raw_file else (
        paths.data_raw / dataset.get("files", {}).get("accepted", "accepted_2007_to_2018Q4.csv.gz")
    )
    if not raw_path.exists():
        raise SystemExit(f"raw file not found: {raw_path}")

    windows = SplitWindows.from_config(config.get("splits", {}))
    windows.validate()
    population = config.get("population", {})

    df, report = prepare_loans(
        raw_path,
        windows=windows,
        term_months=int(population.get("term_months", 36)),
        chunksize=int(population.get("chunksize", 250_000)),
    )

    # Fails loudly rather than warning: an overlapping split is not a degraded result, it is
    # an invalid one.
    assert_temporal_ordering(df)
    LOG.info("temporal ordering check passed")

    write_processed(df, paths.data_processed)
    save_json(report.to_dict(), paths, "data_profile")
    write_report(_profile_lines(df, report, windows, source), paths, "data_profile", source)
    write_manifest(
        run_name,
        config,
        seed,
        paths,
        data_source=source,
        notes={"rows": len(df), "raw_file": str(raw_path.name)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
