#!/usr/bin/env python
"""Repository validation: every reported number must trace to a current artefact.

    python scripts/validate_repository.py

Run this before submitting or after any pipeline change. It is deliberately separate from the
pytest suite: the tests check that the code is correct, this checks that the *results on disk*
were produced by that code and that the README does not quote a number nothing produces.

Confirms that every number quoted in the README traces to an artefact produced by the current
code, that the pipeline chain is intact, and that the leakage controls still hold on real data.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from credit_risk.data import schema  # noqa: E402
from credit_risk.data.splits import assert_temporal_ordering  # noqa: E402
from credit_risk.decision.expected_loss import realised_profit  # noqa: E402
from credit_risk.features.build import FeatureBuilder  # noqa: E402
from credit_risk.graph.construction import TimeRespectingGraph, assert_no_future_edges  # noqa: E402
from credit_risk.utils.cli import load_prepared  # noqa: E402
from credit_risk.utils.config import Paths  # noqa: E402

paths = Paths()
ok, fail = [], []


def check(name, condition, detail=""):
    (ok if condition else fail).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


print("=" * 78)
print("1. PIPELINE CHAIN")
print("=" * 78)
df = load_prepared(paths)
check("prepared table present", len(df) > 0, f"{len(df):,} rows")
check("graph artefact present", (paths.data_processed / "graph.npz").exists())
check("cohort features present", (paths.data_processed / "cohort_features.parquet").exists())

splits = df["split"].value_counts().to_dict()
check("all five splits present", set(splits) >= {"history", "train", "valid", "test", "monitor"},
      str({k: f"{v:,}" for k, v in splits.items()}))

print()
print("=" * 78)
print("2. LEAKAGE CONTROLS (real data)")
print("=" * 78)
assert_temporal_ordering(df)
check("temporal split ordering", True)

builder = FeatureBuilder.load(paths.artifacts / "boosting_features.pkl")
feature_names = set(builder.feature_names)
check("no post-origination column in features",
      not (feature_names & schema.FORBIDDEN_AS_FEATURES),
      f"{len(feature_names)} features")
check("primary set excludes LC underwriting",
      not (feature_names & {"grade", "sub_grade", "int_rate", "installment"}))

graph = TimeRespectingGraph.load(paths.data_processed / "graph.npz")
assert_no_future_edges(graph, df)
check("graph has no future edges", True, f"{graph.n_edges:,} edges")

resolved = df["outcome_observed_period"].notna()
issue = pd.PeriodIndex(df.loc[resolved, "issue_period"])
observed = pd.PeriodIndex(df.loc[resolved, "outcome_observed_period"])
check("no outcome resolves before issue", bool((observed > issue).all()))

monitor = df[df["split"] == "monitor"]
check("monitoring window unlabelled", bool(monitor["default"].isna().all()),
      f"{len(monitor):,} rows")

print()
print("=" * 78)
print("3. SIGNAL ARTEFACTS ALIGN WITH THE TEST SPLIT")
print("=" * 78)
test = df[df["split"] == "test"]
for name in ("boosting_pd_test", "anomaly_test", "disagreement_test"):
    path = paths.artifacts / f"{name}.npy"
    if path.exists():
        values = np.load(path)
        check(f"{name} aligned", len(values) == len(test), f"{len(values):,} vs {len(test):,}")
    else:
        check(f"{name} present", False)

print()
print("=" * 78)
print("4. README NUMBERS TRACE TO ARTEFACTS")
print("=" * 78)

exp01 = json.loads((paths.results_tables / "boosting_exp01_exp02.json").read_text())
auc_lr = exp01["models"]["logistic"]["roc_auc"]
auc_gb = exp01["models"]["lightgbm"]["roc_auc"]
check("EXP01 AUC 0.6581 -> 0.6705",
      abs(auc_lr - 0.6581) < 5e-4 and abs(auc_gb - 0.6705) < 5e-4,
      f"{auc_lr:.4f} -> {auc_gb:.4f}")

cal = exp01["calibration"]
check("EXP02 calibration leaves AUC unchanged",
      abs(cal["identity"]["roc_auc"] - cal["platt"]["roc_auc"]) < 1e-9,
      f"ECE {cal['identity']['ece']:.4f} -> {cal['platt']['ece']:.4f}")

exp08 = json.loads((paths.results_tables / "exp08_blindspots.json").read_text())
check("EXP08 KEEP, 10/10 strata, p<0.01",
      exp08["decision"] == "KEEP" and exp08["n_strata_positive"] == 10
      and exp08["sign_test_p"] < 0.01,
      f"p={exp08['sign_test_p']:.4f}, OR={exp08['odds_ratio_per_sd']:.3f}")

exp10 = json.loads((paths.results_tables / "exp10_disagreement.json").read_text())
check("EXP10 KEEP on log-odds scale",
      exp10["decision"] == "KEEP" and exp10["scale"] == "logodds",
      f"{exp10['n_strata_positive']}/{exp10['n_strata']} strata, p={exp10['sign_test_p']:.4f}")
check("EXP10 log-odds less PD-confounded than probability scale",
      abs(exp10["spearman_with_pd_logodds"]) < abs(exp10["spearman_with_pd_probability"]),
      f"{exp10['spearman_with_pd_logodds']:+.3f} vs {exp10['spearman_with_pd_probability']:+.3f}")

exp11 = json.loads((paths.results_tables / "exp11_swaps.json").read_text())
a = next(p for p in exp11["portfolios"] if p["policy"] == "A_pd_ranked")
b = next(p for p in exp11["portfolios"] if p["policy"] == "B_blindspot_aware")
check("EXP11 RoC 8.17% vs 8.13%",
      abs(a["return_on_capital"] - 0.0817) < 5e-4 and abs(b["return_on_capital"] - 0.0813) < 5e-4,
      f"{a['return_on_capital']:.4f} vs {b['return_on_capital']:.4f}")
check("EXP11 default rate falls 17.30% -> 16.32%",
      abs(a["default_rate"] - 0.1730) < 5e-4 and abs(b["default_rate"] - 0.1632) < 5e-4,
      f"{a['default_rate']:.4f} -> {b['default_rate']:.4f}")
check("EXP11 return-on-capital change is immaterial (<50bp)",
      abs(b["return_on_capital"] - a["return_on_capital"]) < 0.005,
      f"{(b['return_on_capital'] - a['return_on_capital']):+.4f}")

exp06 = json.loads((paths.results_tables / "exp06_rl_exp03_exp06.json").read_text())
conditions = exp06["exp06"]
check("EXP06 trained on a disjoint pool",
      exp06.get("training_pool") == "validation window", str(exp06.get("training_pool")))
breaches = [c for c in conditions.values() if c.get("control_breach")]
check("EXP06 control not breached", len(breaches) == 0,
      f"{len(conditions)} conditions checked")
nulls = [c for c in conditions.values() if "No policy beat" in c["verdict"]]
check("EXP06 null in every condition", len(nulls) == len(conditions),
      f"{len(nulls)}/{len(conditions)}")

print()
print("=" * 78)
print("5. ECONOMICS ARE OBSERVED, NOT ASSUMED")
print("=" * 78)
profit = realised_profit(test)
funded = pd.to_numeric(test["funded_amnt"], errors="coerce")
check("observed book profit computable", np.isfinite(profit.sum()),
      f"${profit.sum():,.0f} on ${funded.sum():,.0f}")

defaulted = pd.to_numeric(test["default"], errors="coerce") > 0.5
back = (pd.to_numeric(test["total_rec_prncp"], errors="coerce").fillna(0)
        + pd.to_numeric(test["recoveries"], errors="coerce").fillna(0)
        - pd.to_numeric(test["collection_recovery_fee"], errors="coerce").fillna(0))
lgd = (1 - back / funded).clip(0, 1)[defaulted].mean()
check("empirical LGD ~0.518 (measured, not assumed)", abs(lgd - 0.518) < 0.01, f"{lgd:.4f}")

cashflow = exp11["cashflow_model"]
check("cashflow model fitted from data",
      cashflow["n_good"] > 1000 and cashflow["n_bad"] > 1000,
      f"n_good={cashflow['n_good']:,} n_bad={cashflow['n_bad']:,}")

print()
print("=" * 78)
print("6. NO STALE OUTPUT PRESENTED AS CURRENT")
print("=" * 78)
stale = ROOT / "results/reports/cohort_graph_exp09_uncertainty.md"
check("EXP09 stale output is banner-marked",
      "STALE OUTPUT" in stale.read_text(encoding="utf-8") if stale.exists() else False)
for removed in ("boosting_exp11_swaps.md", "boosting_exp08_blindspots.md",
                "boosting_exp10_disagreement.md"):
    check(f"superseded {removed} removed",
          not (ROOT / "results/reports" / removed).exists())

manifests = list((ROOT / "results/reports").glob("*_manifest.json"))
sources = {m.stem: json.loads(m.read_text())["data_source"] for m in manifests}
synthetic = [k for k, v in sources.items() if v == "synthetic"]
check("only the Home Credit adapter is synthetic",
      set(synthetic) <= {"home_credit_manifest"}, str(synthetic))

print()
print("=" * 78)
print(f"RESULT: {len(ok)} passed, {len(fail)} failed")
if fail:
    print("FAILURES:")
    for f in fail:
        print("  -", f)
print("=" * 78)
sys.exit(1 if fail else 0)
