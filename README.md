# Graph-Based Credit Risk & Reinforcement Learning Lending Decision System

An end-to-end ML credit decision system on real LendingClub data, built to answer one question:

> Can borrower-level credit-risk modelling be improved by incorporating relational financial
> information, and can calibrated risk predictions then be used by a decision policy to optimise
> lending outcomes rather than merely predict default?

A second question was added later, and the two are reported side by side:

> Can representation-learning signals identify situations where a conventional credit-risk model is
> unreliable, and can that information improve lending decisions?

The governing philosophy is a **small theoretical surface area, fully explainable**. It is not a
survey of techniques. Several experiments return null or negative results, and those are reported as
findings rather than buried — including one headline conclusion that reversed under audit
(see [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md)).

---

## Data

**LendingClub 2007–2018Q4** (`wordsforthewise/lending-club`, Kaggle, licence **CC0**), restricted to
**36-month loans issued 2010-01 … 2015-12** so every outcome is fully matured — no censoring, no
term-mix confound. Splits are strictly **out-of-time**, never random:

| Split | Window | Loans | Default rate |
|---|---|---|---|
| history (graph only) | 2007-06 … 2009-12 | 8,277 | unlabelled by design |
| **train** | 2010-01 … 2014-06 | 239,104 | 12.67% |
| **valid** | 2014-07 … 2014-12 | 90,615 | 14.13% |
| **test** (out-of-time) | 2015-01 … 2015-12 | 283,026 | 14.89% |
| monitor (production traffic) | 2016-01 … 2018-12 | 988,585 | unlabelled by design |

The rising default rate across windows is real vintage deterioration, and it is the reason the
out-of-time split matters.

---

## Results

All figures are on the **out-of-time 2015 test book (283,026 loans)** and were produced by the
current code. `OBSERVED` = measured from data; `SIMULATED` = produced inside the lending simulator
under the assumptions in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md).

| # | Question | Result | Verdict |
|---|---|---|---|
| **EXP01** | Does non-linear modelling beat logistic regression? | AUC **0.6581 → 0.6705** (+0.0124) `OBSERVED` | Material. Boosting justified |
| **EXP02** | Does calibration improve PD reliability? | AUC **unchanged to 5 d.p.**; ECE **0.0193 → 0.0111**; calibration slope 0.991 → 1.005 `OBSERVED` | The point of the project in one row: ranking metrics are blind to calibration |
| **EXP03** | Does expected-profit optimisation beat a fixed PD threshold? | +12,991/episode, 95% CI **[−17,702, +44,103]** `SIMULATED` | **Null result.** Interval includes zero |
| **EXP04** | Do graph cohort features add signal? | AUC **−0.0010**. But with matched configs, ECE **0.0105 → 0.0032**, O/E **0.930 → 0.999** `OBSERVED` | No discrimination gain; **real calibration gain** |
| **EXP05** | Does GraphSAGE beat the aggregates? | AUC **0.6616** vs 0.6679 (aggregates) vs 0.6690 (tabular) `OBSERVED` | **No.** The GNN loses to both |
| **EXP06** | Can a budget-aware learned policy beat static policies? | **No learned policy wins in any of 4 budget conditions** — every interval includes zero. The analytical shadow-price policy leads each constrained condition `SIMULATED` | **Null result for RL** |
| **EXP07** | How does the model degrade under vintage drift? | O/E **0.860 (2010) → 1.077 (2015)** while AUC moves 0.7475 → 0.6705 `OBSERVED` | Calibration drifts before ranking does |
| **EXP08** | Does autoencoder anomaly flag untrustworthy PDs? | Survives PD control in **10/10** deciles, sign-test **p=0.0010**, odds ratio 1.032/SD `OBSERVED` | **KEEP** |
| **EXP09** | Does VGAE relational uncertainty flag them? | Fails PD control (**5/10** deciles, sign-test **p=0.6230**), odds ratio 1.002/SD `OBSERVED` | **REJECT** |
| **EXP10** | Does model disagreement flag them? | Survives in **9/10** deciles, **p=0.0107**, odds ratio 1.043/SD `OBSERVED` | **KEEP** |
| **EXP11** | Does acting on those signals improve the book? | Return on capital **8.17% → 8.13%**; book default rate **17.30% → 16.32%** `OBSERVED` cashflows | **Risk reduction at constant profitability** — not a profit gain |

### Four results worth reading twice

**Calibration is where the graph pays.** EXP04/EXP05 show the graph adds nothing to *discrimination*
— GraphSAGE is the worst of the three arms. But with matched model configurations the graph features
cut expected calibration error by two-thirds and move observed/expected from 0.930 to 0.999. For a
decision engine that multiplies PD by an exposure, that is the property that matters.

**The blind-spot signals are real but small.** Both survive a conditional test against the PD level
— the test the first version of these experiments did not run — but the odds ratios are 1.03–1.04
per standard deviation. They flag *miscalibration*, not bad borrowers.

**Acting on them buys risk reduction, not profit.** EXP11's earlier claim of a $9.8M effect was an
artefact of an assumed-LGD profit formula and a swap-set comparison; on observed cashflows at equal
capital the profit effect is a fraction of a percentage point. What survives is a genuine
risk/return trade-off: roughly one point of book default rate, for free.

**RL does not earn its place here, and that is the finding.** The RL experiment carries a control:
with unconstrained capital the myopic expected-profit rule is optimal, so a learner beating it
indicates a problem rather than a success. In the pre-audit run LinUCB beat it by 35% — because
learning policies were being trained on the *evaluation pool*. After training them on the validation
window instead, LinUCB fell from 199,117 to **119,252** and now loses to the myopic rule
significantly. The control holds, and across all four budget conditions **no learned policy beats
the myopic baseline significantly**.

What does help under a binding budget is pricing capital *analytically*: the shadow-price policy
`expected_profit(lambda)` leads every constrained condition (153,856 vs 133,529 at 60% budget). The
capital-constrained problem here is a knapsack with a closed-form solution, so the defensible
conclusion is that the analytical policy is preferable to the learned one. The RL layer is retained
as a documented negative result, not as a recommendation.

---

## Architecture

- **Layer A — Risk.** Logistic baseline vs LightGBM, then Platt/isotonic calibration. A hard
  deny-list of 49 post-origination columns, enforced on every feature build.
- **Layer B — Graph.** A time-respecting cohort graph (`zip3`, normalised `emp_title`) and a
  GraphSAGE written directly in PyTorch — no `torch-geometric`, so message passing is explainable
  line by line. Neighbour *features* and neighbour *labels* are gated separately and strictly.
- **Layer C — Decision.** Economics estimated from observed cashflows (empirical LGD **0.518**, not
  assumed), a policy ladder from fixed threshold to budget-aware Q-learning, and a simulator whose
  conditional coupling reproduces real outcomes exactly at the funded exposure.
- **Layer D — Trust.** Leakage tests that fail the build, SHAP reason codes, PSI/calibration drift on
  unlabelled production traffic, and a fairness slice over proxy groups.
- **Blind-spot layer.** Autoencoder anomaly, VGAE relational uncertainty and model disagreement,
  each assessed *conditional on the PD level* rather than marginally.

---

## Quickstart

**Environment notes** (these cost real time to rediscover):

- **Smart App Control** on Windows blocks unsigned `.pyd` files, so PyPI wheels for
  scipy/scikit-learn/LightGBM fail to import. Use a **conda** environment; Anaconda's packages are
  signed.
- **Windows long paths** are disabled by default, so the conda `pytorch` package cannot extract.
  Install torch from **pip**.
- The resulting MKL/torch OpenMP clash is handled by `KMP_DUPLICATE_LIB_OK` in
  `src/credit_risk/__init__.py`. Do not remove it.

```bash
conda create -n credit-risk python=3.13 numpy pandas scipy scikit-learn pyarrow lightgbm shap matplotlib pyyaml pytest numba
conda activate credit-risk
pip install torch --index-url https://download.pytorch.org/whl/cu129
pip install -e . --no-deps
```

Data (needs a Kaggle token at `~/.kaggle/kaggle.json`):

```bash
python scripts/download_data.py --config configs/data/lending_club.yaml
```

No token? Every stage runs on a schema-faithful synthetic generator, and every artefact built from it
is stamped `data_source: synthetic` and refused as a finding:

```bash
python scripts/download_data.py --synthetic --n-loans 200000
```

Full pipeline. `--max-threads` keeps a long run from monopolising the machine:

```bash
python scripts/prepare_data.py          --config configs/data/lending_club.yaml --max-threads 10
python scripts/build_graph.py           --config configs/graph/cohort_graph.yaml
python scripts/train_baseline.py        --config configs/models/boosting.yaml
python scripts/train_gnn.py             --config configs/graph/cohort_graph.yaml
python scripts/analyze_blind_spots.py   --config configs/experiments/exp08_blindspots.yaml
python scripts/train_vgae.py            --config configs/experiments/exp09_vgae.yaml
python scripts/analyze_disagreement.py  --config configs/experiments/exp10_disagreement.yaml
python scripts/analyze_decision_swaps.py --config configs/experiments/exp11_swaps.yaml
python scripts/train_policy.py          --config configs/experiments/exp06_rl.yaml
python scripts/evaluate.py              --config configs/models/boosting.yaml
```

Tests (150, all passing):

```bash
python -m pytest -q
```

---

## What this project does not claim

- **Not deployable.** Every policy result comes from a simulator, on an *already-approved*
  population, from one institution and one credit cycle in a long expansion. Nothing here has been
  tested through a downturn.
- **No causal claims.** SHAP explains the model, not the borrower and not the world.
- **No reject inference.** The model estimates `P(default | approved)`, not `P(default | applied)`.
  The gap is quantified against the rejected-applications file but not corrected — doing it properly
  needs propensity data LendingClub never published.
- **No real-world impact claim.** No money was lent.

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) | The contract the code implements. **Read first** |
| [`docs/AUDIT_REPORT.md`](docs/AUDIT_REPORT.md) | Bugs, mathematical errors and the result that reversed |
| [`docs/DATASET.md`](docs/DATASET.md) | Dataset evaluation, licensing, and why Home Credit was rejected |
| [`docs/LEAKAGE.md`](docs/LEAKAGE.md) | Every leakage rule and the test that enforces it |
| [`docs/GRAPH_DESIGN.md`](docs/GRAPH_DESIGN.md) | Time-respecting graph, the two information sets |
| [`docs/RL_FORMULATION.md`](docs/RL_FORMULATION.md) | Why capital constraints make lending sequential — and when they do not |
| [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) | Observed vs estimated vs assumed, itemised |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Column-by-column decision-time rulings |
| [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) | Q&A companion |
