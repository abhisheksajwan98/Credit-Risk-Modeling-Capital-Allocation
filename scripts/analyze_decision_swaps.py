#!/usr/bin/env python
"""EXP11: does acting on blind-spot signals improve the book?

    python scripts/analyze_decision_swaps.py --config configs/experiments/exp11_swaps.yaml

Two capital-constrained books are built from the same applicant pool and the same budget:

* **Policy A** ranks by expected profit per dollar and funds until the budget binds.
* **Policy B** does the same but refuses to fund borrowers flagged by the blind-spot signals.
* **Policy C** funds flagged borrowers at reduced exposure instead of refusing them.

Profit is read from **observed cashflows** (``total_pymnt + recoveries -
collection_recovery_fee - funded_amnt``), never from an assumed loss-given-default, and the
comparison is between the two *portfolios* at equal budget rather than between the two swap sets.
Return on capital is reported alongside profit, because a book that deploys more capital will
usually earn more profit without being better.

An earlier version of this experiment used a hardcoded ``LGD=0.5`` profit formula and compared the
swap sets. That understated book profit by a factor of 2.4 and **reversed the sign** of the
headline result. See docs/AUDIT_REPORT.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.decision.blindspot import (  # noqa: E402
    build_portfolio,
    compare_capital_constrained,
    exposure_capped_policy,
)
from credit_risk.decision.expected_loss import (  # noqa: E402
    EconomicsConfig,
    expected_profit,
    fit_cashflow_model,
    realised_profit,
)
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    load_prepared,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.exp11")


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp11_swaps.yaml")
    parser.add_argument("--flag-quantile", type=float, default=0.90,
                        help="Signals above this quantile are treated as blind spots.")
    parser.add_argument("--budget-fraction", type=float, default=0.80,
                        help="Capital budget as a fraction of profitable demand.")
    parser.add_argument("--exposure-cap", type=float, default=0.5,
                        help="Exposure multiplier applied to flagged borrowers under Policy C.")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].reset_index(drop=True)

    # --- signals ------------------------------------------------------------
    required = {
        "pd": paths.artifacts / "boosting_pd_test.npy",
        "anomaly": paths.artifacts / "anomaly_test.npy",
        "disagreement": paths.artifacts / "disagreement_test.npy",
    }
    missing = [str(p.name) for p in required.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing signal files: " + ", ".join(missing) + "\nRun, in order:\n"
            "  python scripts/train_baseline.py --config configs/models/boosting.yaml\n"
            "  python scripts/analyze_blind_spots.py\n"
            "  python scripts/analyze_disagreement.py"
        )
    signals = {k: np.load(v) for k, v in required.items()}
    for name, values in signals.items():
        if len(values) != len(test):
            raise SystemExit(
                f"{name} has {len(values):,} rows but the test split has {len(test):,}. "
                "Re-run the upstream scripts against the current processed data."
            )

    pd_hat = signals["pd"]
    anomaly, disagreement = signals["anomaly"], signals["disagreement"]

    # --- economics, estimated from observed training cashflows --------------
    economics = EconomicsConfig(
        annual_cost_of_funds=float(config.get_path("economics.annual_cost_of_funds", 0.03)),
        opex_fixed=float(config.get_path("economics.opex_fixed", 150.0)),
        opex_variable_rate=float(config.get_path("economics.opex_variable_rate", 0.010)),
        weighted_average_life_years=float(
            config.get_path("economics.weighted_average_life_years", 1.55)
        ),
    )
    cashflow = fit_cashflow_model(train, economics)

    # --- flags ---------------------------------------------------------------
    q = float(args.flag_quantile)
    flag_anomaly = anomaly > np.quantile(anomaly, q)
    flag_disagree = disagreement > np.quantile(disagreement, q)
    flagged = flag_anomaly | flag_disagree
    LOG.info(
        "flagged %s of %s test borrowers (%.1f%%) at the %.0fth percentile of either signal",
        f"{flagged.sum():,}", f"{len(test):,}", 100 * flagged.mean(), 100 * q,
    )

    analysis = compare_capital_constrained(
        test, pd_hat, flagged, cashflow, economics, budget_fraction=float(args.budget_fraction)
    )

    # --- Policy C: cap exposure rather than refuse ---------------------------
    exposure = pd.to_numeric(test["funded_amnt"], errors="coerce").fillna(0.0)
    apr = pd.to_numeric(test["int_rate"], errors="coerce").fillna(13.0) / 100.0
    profit = expected_profit(pd_hat, exposure.to_numpy(), apr.to_numpy(), cashflow, economics)
    density = pd.Series(
        np.divide(profit, exposure.to_numpy(), out=np.zeros_like(profit),
                  where=exposure.to_numpy() > 0),
        index=test.index,
    )
    approve_c, exposure_c = exposure_capped_policy(
        density, exposure, flagged, analysis.budget, cap_multiplier=float(args.exposure_cap)
    )
    # Realised profit scales with the exposure actually taken, so a half-size loan earns half the
    # realised return. This is the same linear-scaling assumption the simulator makes for
    # counterfactual exposures, and it is an assumption -- see docs/ASSUMPTIONS.md.
    scale = np.where(flagged, float(args.exposure_cap), 1.0)
    capped_profit = float((realised_profit(test) * scale)[approve_c].sum())
    capped_capital = float(exposure_c[approve_c].sum())
    policy_c = build_portfolio(test, approve_c, "C_exposure_capped", analysis.budget)
    policy_c.capital_deployed = capped_capital
    policy_c.observed_profit = capped_profit
    policy_c.return_on_capital = capped_profit / capped_capital if capped_capital else float("nan")
    policy_c.budget_utilisation = capped_capital / analysis.budget
    analysis.portfolios.append(policy_c)

    table = analysis.frame
    save_table(table, paths, f"{run_name}_portfolios", index=False)

    baseline = "A_pd_ranked"
    verdict_b = analysis.verdict("B_blindspot_aware", baseline)
    a = analysis.portfolios[0]
    c_delta_roc = policy_c.return_on_capital - a.return_on_capital

    lines = [
        "# EXP11: does acting on blind-spot signals improve the book?",
        "",
        f"Applicant pool: {len(test):,} out-of-time loans (2015 vintage). "
        f"Capital budget ${analysis.budget:,.0f}, "
        f"{100 * float(args.budget_fraction):.0f}% of profitable demand.",
        f"Flagged as blind spots: {flagged.sum():,} borrowers ({100 * flagged.mean():.1f}%) "
        f"above the {100 * q:.0f}th percentile of the anomaly **or** disagreement signal.",
        "",
        "## Portfolios at equal budget, measured on observed cashflows",
        "",
        table.round(4).to_markdown(index=False),
        "",
        "## Verdict",
        "",
        verdict_b,
        "",
        f"The swap involved {analysis.n_swapped_out:,} loans out "
        f"(realised return {analysis.swapped_out_return:+.2%}, default "
        f"{analysis.swapped_out_default_rate:.2%}) and {analysis.n_swapped_in:,} loans in "
        f"(realised return {analysis.swapped_in_return:+.2%}, default "
        f"{analysis.swapped_in_default_rate:.2%}).",
        "",
        f"Policy C, which caps flagged borrowers at {float(args.exposure_cap):.0%} exposure rather "
        f"than refusing them, changed return on capital by {c_delta_roc:+.2%} against Policy A.",
        "",
        "## What this does and does not show",
        "",
        "- Profit here is **observed**: `total_pymnt + recoveries - collection_recovery_fee - "
        "funded_amnt` for every matured loan. No loss-given-default is assumed.",
        "- The comparison is between **whole portfolios at the same budget**, not between the two "
        "swap sets. Swap sets need not deploy equal capital, so differencing their profits is not "
        "a portfolio result.",
        "- **Return on capital is the comparable quantity.** Profit alone rewards whichever book "
        "happens to deploy more capital.",
        "- This is a *retrospective reallocation* on an already-approved population. It does not "
        "establish what would happen to borrowers LendingClub declined, and it is not evidence "
        "that the policy is safe to deploy.",
        "",
        "> **Correction.** An earlier version of this experiment computed profit as "
        "`amount x apr x 1.55 - amount x 0.5 x default` and compared the two swap sets rather "
        "than the portfolios. That formula understated realised book profit by roughly a factor "
        "of 2.4, and it reported a **$9.8M loss** as a *'net increase in profit'* -- the "
        "conclusion was asserted regardless of the sign of the number printed above it. Measured "
        "on observed cashflows at equal budget, the effect is a fraction of a percentage point of "
        "return on capital in either direction, i.e. not a profit result at all. "
        "See `docs/AUDIT_REPORT.md`.",
    ]
    write_report(lines, paths, f"{run_name}", source)
    save_json(
        {
            "budget": analysis.budget,
            "flag_quantile": q,
            "n_flagged": int(flagged.sum()),
            "portfolios": [p.to_dict() for p in analysis.portfolios],
            "n_swapped_out": analysis.n_swapped_out,
            "n_swapped_in": analysis.n_swapped_in,
            "swapped_out_return": analysis.swapped_out_return,
            "swapped_in_return": analysis.swapped_in_return,
            "cashflow_model": cashflow.to_dict(),
        },
        paths, f"{run_name}",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
