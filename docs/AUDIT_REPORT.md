# Audit report

A full correctness audit of the repository, carried out before submission. Every claim below is
traceable to a file in `results/` or to a test in `tests/`.

The headline: **one reported result was wrong in sign and had to be withdrawn**, two experiments
rested on a confound that made their conclusions unsupported as stated, and the reinforcement-
learning experiment tripped its own falsification criterion. All are corrected or explicitly
marked below.

---

## A. Bugs found

| # | Where | Defect |
|---|---|---|
| A1 | `scripts/analyze_decision_swaps.py` | Report template asserted *"a net increase in profit"* and *"this proves representation learning improves decision quality"* by string-interpolating a value, **regardless of its sign**. The printed value was `$-9,794,019` — a loss. |
| A2 | `scripts/analyze_decision_swaps.py` | "Realised Profit" was computed from a formula (`amount x apr x 1.55 - amount x 0.5 x default`), not from the realised cashflows the dataset contains. |
| A3 | `src/credit_risk/models/autoencoder.py` | `BatchNorm1d` with no `drop_last`: a final batch of size 1 raises at training time whenever `len(train) % batch_size == 1`. |
| A4 | `src/credit_risk/models/autoencoder.py` | Mutable default argument `hidden_dims: list[int] = [32, 16]`, shared across instances. |
| A5 | `src/credit_risk/models/vgae.py` | `encode()` built `n_layers - 1` shared layers but applied only `shared_layers[0]`, silently ignoring the rest for `n_layers > 2`. |
| A6 | `src/credit_risk/models/vgae.py` | `z_expanded` referenced outside the branch that defines it; safe only by coincidence of an adjacent guard. |
| A7 | `src/credit_risk/evaluation/metrics.py` | Field named `observed_expected_ratio` computed `predicted / observed` — the **reciprocal** of the credit-risk O/E convention. Values were right; the name inverted their meaning. |
| A8 | repo-wide | 8 unused imports; `networkx`, `tqdm` declared as dependencies but never imported. |

## B. Mathematical and statistical issues

| # | Issue | Correction |
|---|---|---|
| B1 | **Swap-set differencing.** EXP11 compared `profit(swapped in) - profit(swapped out)`. The two sets need not deploy equal capital, so the difference conflates selection quality with capital volume. | Compare **whole portfolios at an identical budget**, and report **return on capital** alongside profit. |
| B2 | **Assumed LGD.** EXP11 hardcoded `LGD = 0.5`. The empirical LGD measured from training cashflows is **0.518**, and more importantly the *revenue* term was wrong: the formula produced $112M of book profit against **$270M observed**, a factor of 2.4. | Read profit from `total_pymnt + recoveries - collection_recovery_fee - funded_amnt`. |
| B3 | **Marginal-only evidence.** EXP08/09/10 supported "this signal finds blind spots" from a quantile table alone. Because all three signals correlate with the PD level, and a boosted model is already less calibrated at high PD, such a table cannot separate a real signal from a restatement of PD. | Added `conditional_signal_table`: within each PD decile, split at the signal median and compare calibration gaps, with an exact binomial sign test and an incremental logistic coefficient. |
| B4 | **Scale-dependent disagreement.** EXP10 used `abs(p_a - p_b)`, which is mechanically larger mid-range and so partly restates PD (Spearman **+0.316**). | Measure on log-odds: `abs(logit(p_a) - logit(p_b))`. Correlation with PD falls to **-0.150**. |
| B5 | **Confounded ablation.** EXP10's two models differed in features *and* hyper-parameters *and* calibrator. | Both arms refitted under an identical configuration; only the feature set differs. |
| B6 | **Uninterpretable null.** EXP09 trained the VGAE for 10 epochs with no validation split, making "no signal" and "undertrained" indistinguishable. | Early stopping on held-out link reconstruction; the final run stopped at **19 epochs**. |

## C. Leakage issues

**None found in the pre-existing controls**, which is worth stating plainly. The deny-list, the
out-of-time splits, the availability guard and the two-information-set graph gating all held up,
and the planted-violation tests confirm the guards actually fire.

One new risk was introduced by the EXP08–EXP11 work and has been fixed:

| # | Issue | Correction |
|---|---|---|
| C1 | **Learning policies trained on the evaluation pool.** EXP06 trained LinUCB, Thompson sampling and the Q-learner for 120–400 episodes on the *test* applicants, then scored them against static policies that had no such exposure. | Learning policies now train on the **validation window** and are evaluated on the test window. |

## D. Logic issues

| # | Issue |
|---|---|
| D1 | **Control criterion ignored.** EXP06's own report states that with unconstrained capital the myopic policy is optimal and *"if they appear to [beat it], the simulator has a bug"*. LinUCB beat it by **+35.3%, CI excluding zero**, and the criterion was not acted on. A `control_breach` check now fires in the report. |
| D2 | **Degenerate policy reported as a result.** Thompson sampling collapsed to a **0.06% approval rate** (reward 58 against ~150k) and was tabulated beside working policies. Degeneracy is now detected and flagged. |
| D3 | **Unfilled verdicts.** EXP08/09/10 reports ended with *"Check if Q5 exhibits…"* — an instruction to the reader, not a finding — while `INTERVIEW_GUIDE.md` quoted confident conclusions drawn from them. |
| D4 | **Duplicated economics.** EXP11 re-implemented the cost model that already exists in `decision/expected_loss.py`, with different constants. Consolidated into `decision/blindspot.py`. |

## E. Efficiency

| # | Change |
|---|---|
| E1 | VGAE training did **~40x redundant forward passes** per step (encoding the batch, then every positive neighbour, then every negative separately). Now one encode over the union of required nodes. |
| E2 | `VGAETrainer.embed` re-uploaded the full feature matrix to GPU on every call; now cached. |
| E3 | Added `configure_threads` / `--max-threads`, so a long run can be capped (LightGBM ignores `OMP_NUM_THREADS` unless `n_jobs` is set explicitly, which it now honours). |

## F. Files removed

- Unused imports across 5 modules; `networkx` and `tqdm` from `pyproject.toml`.
- Superseded pre-audit outputs: `boosting_exp08_blindspots.md`, `boosting_exp10_disagreement.md`,
  `boosting_exp11_swaps.md` and their tables, replaced by `exp08_blindspots.*`,
  `exp10_disagreement.*`, `exp11_swaps.*`.

## G. Files modified

`decision/blindspot.py` (new), `models/autoencoder.py`, `models/vgae.py`,
`evaluation/metrics.py`, `models/calibration.py`, `models/boosting.py`, `utils/runtime.py`,
`utils/cli.py`, `scripts/analyze_blind_spots.py`, `scripts/analyze_disagreement.py`,
`scripts/train_vgae.py`, `scripts/analyze_decision_swaps.py`, `scripts/train_policy.py`,
`tests/test_blindspot.py` (new, 14 tests), `tests/test_models.py`,
`configs/experiments/exp08–exp11` (new).

## H. Experiments re-run

EXP08, EXP09, EXP10, EXP11 re-run end-to-end on the real 283,026-loan out-of-time window.
EXP06 re-run with the corrected train/evaluation separation.

## I. Results that changed

### I1. EXP11 — the headline reversed, then dissolved

| | Reported before | After correction |
|---|---|---|
| Method | swap-set difference, assumed LGD | portfolios at equal budget, observed cashflows |
| Result | **-$9,794,019**, described as a *"net increase in profit"* | A: RoC **8.17%**, B: RoC **8.13%** |
| Reading | "representation learning improves decision quality" | **-4bp of return on capital: no material difference** |

What *is* supported: the blind-spot-aware book carries a **lower default rate (17.30% → 16.32%)
at essentially unchanged return**. That is risk reduction at constant profitability — a real,
modest result, and not the one originally claimed.

### I2. EXP08 — conclusion survives, on better evidence

Anomaly is nearly orthogonal to PD (Spearman **+0.022**) and the effect holds **10/10 PD deciles**,
sign-test **p=0.0010**, odds ratio **1.032** per SD. **KEEP.**

### I3. EXP10 — conclusion survives, much smaller than claimed

The marginal table showed AUC falling 0.693 → 0.624 across disagreement quintiles. On the log-odds
scale and conditioned on PD, the effect is **9/10 deciles, p=0.0107, odds ratio 1.043** per SD —
real, but a fraction of what the marginal table implied.

Separately: the graph-augmented arm leaves AUC flat (0.6690 → 0.6679) while cutting **ECE
0.0105 → 0.0032** and correcting a **7% understatement** of the portfolio default rate
(predicted 13.84% vs observed 14.89%, → 14.87%). Graph features help *calibration*, not ranking.

### I4. EXP09 — clean negative, now interpretable

**5/10 deciles, p=0.62, odds ratio 1.002**, sigma-to-PD correlation −0.0006 and sigma-to-degree
+0.006. Trained to early stopping at 19 epochs, so this is a property of the signal, not the
training budget. **REJECT.**

### I5. EXP06 — the RL win was a leakage artifact, and RL is not warranted

Re-run with learning policies trained on the **validation window** (90,615 applicants) and
evaluated on the test window.

| Policy | unconstrained | budget 0.60 | budget 0.35 | budget 0.20 |
|---|---|---|---|---|
| expected_profit (myopic) | 147,180 | 133,529 | 80,766 | 46,934 |
| **expected_profit(lambda)** — analytical shadow price | n/a | **153,856** | **119,169** | **61,771** |
| fixed_threshold@0.15 | 134,189 | 135,812 | 85,998 | 47,703 |
| linucb | 119,252 | 117,262 | 63,891 | 46,219 |
| budget_aware_q | 91,555 | 109,918 | 56,347 | 33,811 |
| thompson | 173,320 | 45,094 | 46,445 | 35,437 |

Three findings:

1. **The control now holds.** With unconstrained capital no learner significantly beats the myopic
   optimum. LinUCB, which previously beat it by **+35.3% with a CI excluding zero**, now
   *significantly loses* (**-27,928**, CI [-53,382, -1,760], winning 2/12 seeds). The original
   result was entirely the train-on-evaluation-pool leak of C1.
2. **Thompson sampling's degeneracy was the same artifact.** It previously collapsed to a **0.06%
   approval rate**; trained on a disjoint pool it approves 59.9%. It remains unstable across
   conditions (173,320 unconstrained, 45,094 at budget 0.60) and should not be relied on.
3. **The analytical shadow price beats every learned policy in every constrained condition.** The
   capital-constrained problem is a knapsack with a closed-form solution, and the learners
   underperform it. The advantage over the myopic baseline is directionally consistent and largest
   where the budget binds hardest in relative terms (**+38,402 at budget 0.35**), but at 12 seeds
   **no condition reaches significance** — every confidence interval includes zero.

**Conclusion: reinforcement learning is not justified by this formulation.** The honest result is
that pricing capital correctly matters and that a Lagrangian shadow price obtains it in closed
form, without learning. This is a negative result for the RL layer and is reported as one.

### I6. EXP05 — GraphSAGE is the weakest arm

On real data: tabular **0.6690**, tabular+cohort **0.6679**, GraphSAGE **0.6616**.

## J. Remaining limitations

1. **The shadow-price advantage is not statistically significant.** It leads every constrained
   condition and the ordering is stable, but all four confidence intervals include zero at 12
   seeds. More seeds would be needed to claim it. The *negative* result — that learned policies do
   not beat it — is on firmer ground, since several learners lose significantly.
2. **Selection bias is unaddressed.** Everything is estimated on already-approved LendingClub
   borrowers, so all of it is `P(default | approved)`. Reject inference is discussed in
   `LEAKAGE.md` but not implemented; the propensities needed to do it properly are not published.
3. **One institution, one credit cycle.** 2010–2015 US unsecured consumer credit in an expansion.
   Nothing here has been tested through a downturn.
4. **Counterfactual exposure remains an assumption.** Policy C's half-exposure book scales realised
   return linearly with exposure. Reasonable for a 2x move, unsupported beyond it.
5. **Repeat borrowers cannot be detected.** `member_id` is scrubbed, so rows may not be
   independent and every confidence interval is slightly too narrow.
6. **Blind-spot effects are small.** Odds ratios of 1.03–1.04 per standard deviation are real and
   statistically clear at n=283k, but they are not large. The honest framing is that these signals
   flag *miscalibration*, not that they materially improve discrimination.
7. **Stored result files predating this audit** use the old `observed_expected_ratio` key with the
   reciprocal meaning. Files regenerated after the audit carry both keys with correct names.
