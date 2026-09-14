#!/usr/bin/env python
"""Reproduce EXP01-EXP02 on Home Credit, to show the pipeline generalises across schemas.

    python scripts/run_home_credit.py --application data/raw/home_credit/application_train.csv
    python scripts/run_home_credit.py --synthetic            # smoke test, no download needed

Layers B and C are **not** available on this dataset and the loader refuses rather than
improvising -- Home Credit has no borrower-to-borrower links, no interest rate and no recovery
amounts. See the module docstring in ``credit_risk/data/home_credit.py``.

Every number this script produces carries a caveat that must travel with it: **the splits are
random, not out-of-time**, because Home Credit contains no absolute calendar date. Metrics are
therefore optimistic relative to deployment and are not comparable with the LendingClub results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from credit_risk.data.home_credit import (  # noqa: E402
    build_matrix,
    generate_synthetic_home_credit,
    home_credit_features,
    prepare_home_credit,
)
from credit_risk.evaluation.metrics import compare_models, compute_metrics  # noqa: E402
from credit_risk.models.baseline import LogisticBaseline, LogisticConfig  # noqa: E402
from credit_risk.models.boosting import BoostingConfig, BoostingModel  # noqa: E402
from credit_risk.models.calibration import compare_calibrations  # noqa: E402
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.home_credit")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/models/boosting.yaml")
    parser.add_argument("--application", default=None, help="Path to application_train.csv")
    parser.add_argument("--bureau", default=None, help="Optional path to bureau.csv")
    parser.add_argument("--synthetic", action="store_true", help="Use a shaped stand-in frame.")
    parser.add_argument("--nrows", type=int, default=None)
    args = parser.parse_args()
    config, paths, _, seed = resolve(args)
    run_name = "home_credit"

    if args.synthetic:
        import tempfile

        frame = generate_synthetic_home_credit(n_rows=args.nrows or 20_000, seed=seed)
        tmp = Path(tempfile.mkdtemp()) / "application_train.csv"
        frame.to_csv(tmp, index=False)
        application, source = tmp, "synthetic"
    elif args.application:
        application, source = Path(args.application), "real"
    else:
        raise SystemExit("pass --application <path to application_train.csv>, or --synthetic")

    df, report = prepare_home_credit(
        application, bureau_path=args.bureau, seed=seed, nrows=args.nrows
    )
    numeric, categorical = home_credit_features(df)
    LOG.info("features: %d numeric, %d categorical", len(numeric), len(categorical))

    splits = {s: df[df["split"] == s] for s in ("train", "valid", "test")}
    y = {s: splits[s]["default"].astype(int).to_numpy() for s in splits}

    X_train, medians, levels = build_matrix(splits["train"], numeric, categorical)
    X_valid, _, _ = build_matrix(splits["valid"], numeric, categorical, medians, levels)
    X_test, _, _ = build_matrix(splits["test"], numeric, categorical, medians, levels)

    metrics = {}
    logistic = LogisticBaseline(numeric, categorical, LogisticConfig()).fit(X_train, y["train"])
    metrics["logistic"] = compute_metrics(y["test"], logistic.predict_proba(X_test))
    LOG.info("logistic | %s", metrics["logistic"].summary_line())

    boosting = BoostingModel(categorical, BoostingConfig(n_estimators=2000)).fit(
        X_train, y["train"], X_valid, y["valid"], seed=seed
    )
    p_valid = boosting.predict_proba(X_valid)
    p_test = boosting.predict_proba(X_test)
    metrics["lightgbm"] = compute_metrics(y["test"], p_test)
    LOG.info("lightgbm | %s", metrics["lightgbm"].summary_line())

    calibration = compare_calibrations(y["valid"], p_valid, y["test"], p_test)

    save_table(compare_models(metrics), paths, f"{run_name}_exp01_models")
    save_table(calibration.table(), paths, f"{run_name}_exp02_calibration")
    save_table(boosting.feature_importance().head(30), paths, f"{run_name}_importance", index=False)

    lines = [
        "# Home Credit: EXP01 and EXP02 (secondary dataset)",
        "",
        "> **Splits here are random, not out-of-time.** Home Credit contains no absolute calendar "
        "date -- every temporal field is expressed in days relative to each application -- so a "
        "temporal split is impossible. These metrics are therefore optimistic relative to "
        "deployment and are **not comparable** with the LendingClub results, which use a strict "
        "out-of-time test window. This is the principal reason the dataset was not selected as "
        "the primary one; see docs/DATASET.md.",
        "",
        f"Rows: {report.rows:,}. Default rate: {report.default_rate:.4f}. "
        f"Features: {len(numeric)} numeric, {len(categorical)} categorical.",
        "",
        "## EXP01",
        "",
        compare_models(metrics).round(5).to_markdown(),
        "",
        "## EXP02",
        "",
        calibration.table().round(5).to_markdown(),
        "",
        f"**Verdict.** {calibration.verdict()}",
        "",
        "## Notes from preparation",
        "",
    ]
    lines += [f"- {n}" for n in report.notes]
    lines += [
        "",
        "## What is deliberately absent",
        "",
        "- **No graph (Layer B).** Home Credit's tables are hierarchical: a borrower and *their "
        "own* bureau records. There are no borrower-to-borrower links, so any graph would be a "
        "k-nearest-neighbour construction in feature space -- a smoothing regulariser, not "
        "relational financial information.",
        "- **No decision layer (Layer C).** No interest rate, no recovery amount, no realised "
        "cashflow. Every term in an expected-profit calculation would have to be invented.",
    ]
    write_report(lines, paths, f"{run_name}_exp01_exp02", source)
    save_json(
        {
            "prepare": report.to_dict(),
            "models": {k: v.to_dict() for k, v in metrics.items()},
            "calibration": {k: v.to_dict() for k, v in calibration.per_method.items()},
        },
        paths, f"{run_name}_exp01_exp02",
    )
    write_manifest(run_name, config, seed, paths, data_source=source,
                   notes={"split_strategy": "random (no calendar date exists in this dataset)"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
