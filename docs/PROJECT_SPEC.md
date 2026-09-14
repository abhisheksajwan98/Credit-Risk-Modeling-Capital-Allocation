# Project Specification

**Graph-Based Credit Risk & Reinforcement Learning Lending Decision System**

Status: finalised 2026-08-26. This document is the contract the rest of the repository implements.
Where an implementation choice below is later revised, this file is updated first.

---

## 1. Thesis

> Can borrower-level credit-risk modelling be improved by incorporating relational financial
> information, and can calibrated risk predictions subsequently be used by a decision policy to
> optimise lending outcomes rather than merely predict default?

The project is deliberately organised as three layers that each answer one half of that question,
and a fourth cross-cutting layer that keeps the result honest.

| Layer | Question | Primary output |
|-------|----------|----------------|
| **A — Risk** | How likely is this borrower to default? | A *calibrated* probability of default (PD) |
| **B — Graph** | Does relational information add signal beyond borrower-level features? | Graph features + GraphSAGE, benchmarked against A |
| **C — Decision** | Given PD, what is the *right action*? | A lending policy, benchmarked against static credit policy |
| **D — Trust** | Should anyone believe this? | Leakage tests, explanations, fairness slice, drift monitors |

The deliverable should read as *an ML-driven credit decision system*, not as a survey of techniques.

---

## 2. Dataset decision (summary)

**Selected: LendingClub accepted-loan data, 2007-2018Q4** (`wordsforthewise/lending-club`, Kaggle,
licence **CC0: Public Domain**, verified 2026-08-26).

Home Credit Default Risk was evaluated as the nominated primary candidate and **rejected as the
principal dataset** for three specific reasons:

1. **No absolute calendar date exists anywhere in Home Credit.** Every temporal field (`DAYS_BIRTH`,
   `DAYS_CREDIT`, `DAYS_DECISION`, ...) is expressed relative to each application. Out-of-time
   validation, vintage analysis and drift measurement — three things this project treats as
   mandatory — are therefore impossible.
2. **No borrower-to-borrower relations.** Home Credit's tables are strictly hierarchical
   (a borrower and *their own* bureau records / prior applications). Any graph would have to be
   fabricated from feature-space similarity, which does not test the project's central question.
3. **No loan pricing or recovery amounts.** Layer C's reward would be entirely invented.

Home Credit is retained as a **secondary adapter** so that Layer A (EXP01-EXP03) can be reproduced
on it, which is worth having: it demonstrates the pipeline generalises across schemas.

Full evaluation, licensing and download procedure: [`DATASET.md`](DATASET.md).

---

## 3. Modelling population

The single most consequential decision in a credit-risk project is *which rows are modelled and what
counts as default*. This project fixes it as follows.

| Rule | Value | Reason |
|------|-------|--------|
| Product | 36-month term only | Uniform maturity means one fixed outcome horizon, no censoring, and no term-mix confound between vintages |
| Issue window (modelled) | `2010-01` ... `2015-12` | Every loan has at least 36 months of observation before the 2018-12 data cutoff, so **every outcome is fully realised** |
| Issue window (graph history only) | `2007-06` ... `2009-12` | Provides past neighbours for early modelled loans; never used as training rows |
| Excluded from modelling | `2016-01` onward | 36-month loans issued from 2016 have not matured by the data cutoff. Used as *unlabelled production traffic* for drift monitoring only |

**Target.** `default = 1` where terminal `loan_status` is `Charged Off` or `Default`;
`default = 0` where `loan_status` is `Fully Paid`. The legacy
`Does not meet the credit policy. Status:*` variants map to the same two classes. Any row that is
still non-terminal inside the matured window is a data anomaly: it is dropped and the count reported.

**Splits — strictly out-of-time, never random.**

```
train   issue_d in [2010-01, 2014-06]     (~244k loans)
valid   issue_d in [2014-07, 2014-12]     (~ 82k loans)
test    issue_d in [2015-01, 2015-12]     (~292k loans)   <- out-of-time, touched once
```

A random split would be indefensible here: it lets the model see the future of the same credit cycle.
The out-of-time test set is the only honest estimate of deployment performance.

---

## 4. What is *not* in scope

Deliberately excluded, because none of them are required to answer the thesis:
graph transformers, tabular transformers, causal inference, survival analysis, federated learning,
offline RL beyond simple fitted-Q, multi-agent RL, RLHF, Bayesian deep learning, graph contrastive
pre-training, distributed training, and large hyper-parameter sweeps.

Two exclusions worth stating explicitly because they are *tempting*:

- **No `torch-geometric`.** GraphSAGE and neighbour sampling are implemented directly in PyTorch
  (~200 lines). PyG's Windows sampling wheels are unreliable, and — more importantly — a hand-written
  sampler is something that can be explained line-by-line in an interview. A library call is not.
- **No experiment-tracking server, no orchestration framework, no model registry.** Runs write
  versioned YAML/JSON/Parquet under `results/` and `artifacts/`. That is sufficient for reproducibility
  at this scale, and every added service is surface area that has to be defended.

---

## 5. Layer A — credit risk prediction

| Component | Choice | Note |
|-----------|--------|------|
| Baseline | Logistic Regression | One-hot + scaling; the interpretable banking-standard reference |
| Strong model | **LightGBM** | One boosting framework only. XGBoost is not also implemented — the comparison is not informative and doubles the surface area |
| Imbalance | class weights, evaluated *against* doing nothing | ~15% default rate is not extreme; resampling usually harms calibration, so this is tested rather than assumed |
| Calibration | Platt (sigmoid) **and** isotonic, fitted on the validation window | Mandatory — Layer C consumes probabilities, not scores |
| Metrics | ROC-AUC, PR-AUC, Brier, log-loss, reliability curve, ECE, KS | Ranking quality and probability quality are reported separately, always |

---

## 6. Layer B — graph

Relational structure is a **shared-attribute cohort graph**, not an observed transaction network —
LendingClub publishes no counterparty links. This is stated plainly rather than dressed up.

- **R1 `zip3`** — shared 3-digit ZIP prefix (~900 values). Economic rationale: borrowers in one local
  labour and housing market are exposed to correlated shocks. This is the primary relation.
- **R2 `emp_norm`** — normalised employment string. Carries a known discontinuity: LendingClub's
  `emp_title` held **employer name before 2013-09-23 and job title after**. This is documented,
  and R2's contribution is ablated separately rather than silently blended in.

**Every edge is time-respecting**: an edge from `j` to `i` requires `issue_d[j] < issue_d[i]`. Neighbour
*features* need only that. Neighbour *labels* additionally require the neighbour's outcome to have
been **resolved before** `issue_d[i]`. The two are treated as separate information sets throughout;
conflating them is the classic way graph credit models leak. See [`GRAPH_DESIGN.md`](GRAPH_DESIGN.md).

The comparison is three-way and the GNN is not assumed to win:

- **A** tabular features only
- **B** tabular + graph-derived aggregate features (time-windowed cohort statistics)
- **C** GraphSAGE over sampled time-respecting neighbourhoods

---

## 7. Layer C — decision

```
features -> PD (calibrated) -> expected loss / expected profit -> action
```

Economics are derived from **observed cashflows** wherever possible, and labelled as assumptions
wherever not (see [`ASSUMPTIONS.md`](ASSUMPTIONS.md)). LGD is estimated empirically from
`total_rec_prncp`, `recoveries` and `collection_recovery_fee` rather than assumed as a round number.

Policy ladder, each evaluated in the identical simulator under identical seeds:

1. `FixedThresholdPolicy` — approve if PD below tau
2. `RiskBandPolicy` — grade-style bands mapped to exposure
3. `ExpectedProfitPolicy` — per-applicant argmax of expected profit (the strong myopic baseline)
4. `ContextualBanditPolicy` — LinUCB / Thompson sampling; demonstrates exploration-exploitation
5. `BudgetAwareQPolicy` — finite-horizon MDP under a capital constraint

**Why this is genuinely RL and not supervised learning.** Under an unconstrained balance sheet the
optimal policy *is* the myopic per-applicant argmax, and RL buys nothing — we expect and report that.
The problem becomes sequential only when **capital binds**: funding a marginal applicant today
consumes capital that a better applicant tomorrow cannot then use, so the optimal decision depends on
remaining budget and time left in the funding period. That opportunity cost is the shadow price of
capital, and it is exactly what the learned value function has to recover. The headline experiment is
therefore *RL is approximately equal to myopic when capital is free; RL beats myopic when capital is
scarce* — a falsifiable claim, not a leaderboard number. Full formulation:
[`RL_FORMULATION.md`](RL_FORMULATION.md).

---

## 8. Experiments

| ID | Question | Decides |
|----|----------|---------|
| EXP01 | Does non-linear modelling improve default prediction over logistic regression? | Layer A model choice |
| EXP02 | Does calibration improve PD reliability, and at what cost to ranking? | Which probabilities Layer C consumes |
| EXP03 | Does expected-loss/profit optimisation beat a fixed PD threshold? | Whether decisions differ from predictions |
| EXP04 | Do graph-derived cohort features add signal beyond tabular features? | Whether relational info matters at all |
| EXP05 | Does GraphSAGE add value beyond graph-derived features? | Whether a *GNN* is warranted, or aggregates suffice |
| EXP06 | Can a budget-aware learned policy beat static policies on risk-adjusted return? | The RL claim |
| EXP07 | How do model and policy degrade under vintage drift? | Deployment realism |

Each experiment writes a machine-readable result to `results/tables/` and a short written verdict to
`results/reports/`. A null result is reported as a null result.

---

## 9. Compute budget

Target: one consumer laptop (developed against an RTX 3060 Laptop, 6 GB VRAM; CPU-only must also work).

| Stage | Expected cost |
|-------|---------------|
| Raw download | ~1.4 GB compressed |
| Prepared parquet | ~250 MB |
| Feature build | ~2-4 min, under 6 GB RAM |
| Logistic regression | under 1 min |
| LightGBM (with modest search) | ~5-15 min CPU |
| Graph construction | ~2-5 min; ~620k nodes, ~12M sampled edges |
| GraphSAGE (2 layers, 64-dim) | ~10-20 min GPU / ~40-60 min CPU |
| Simulation and policy sweep | ~5 min per policy |

Anything that exceeds this is simplified rather than scaled up.

---

## 10. Reproducibility contract

- Every script takes a YAML config from `configs/` and a `--seed`; defaults are pinned.
- Data preparation is deterministic given the raw files; feature fitting happens on **train only**.
- `tests/` contains leakage tests that fail loudly, not warnings that scroll past.
- The out-of-time test window is evaluated **once per experiment**, at the end, and that number is
  the one reported. Iterating against it would turn it into a validation set.
