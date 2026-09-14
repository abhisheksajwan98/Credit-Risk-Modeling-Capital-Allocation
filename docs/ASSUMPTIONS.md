# Assumptions

Every quantity in this project falls into one of three categories, and this file states which is
which. Portfolio simulations go wrong when an invented number quietly acquires the same status as a
measured one.

| Status | Meaning |
|---|---|
| **Observed** | Read directly from the data. Arithmetic, not modelling |
| **Estimated** | Fitted from observed data on the training window |
| **Assumed** | Chosen by us. Not measurable in this dataset. Swept in sensitivity analysis |

---

## 1. Observed

Facts about the data, not choices.

| Quantity | Source |
|---|---|
| Default outcome | `loan_status`, mapped to terminal classes only |
| Exposure | `funded_amnt` |
| Price | `int_rate` |
| Term | `term` (36 months only) |
| Cash received | `total_pymnt` + `recoveries` - `collection_recovery_fee` |
| Principal repaid | `total_rec_prncp` |
| Decision date | `issue_d` |
| Realised profit | `total_pymnt + recoveries - collection_recovery_fee - funded_amnt` |

`recoveries` is reported *separately* from `total_pymnt` in LendingClub's schema, so it must be
added rather than assumed to be included. Getting that wrong understates the return on defaulted
loans.

---

## 2. Estimated

Fitted on the **training window only**, from observed cashflows.

| Quantity | How | Why it is not assumed |
|---|---|---|
| **Good-loan yield** `y_good(apr)` | Linear fit of realised return on APR, over non-defaulted training loans | Directly measurable |
| **Defaulted-loan loss rate** `l_bad(apr)` | Linear fit of realised loss on APR, over defaulted training loans | Directly measurable |
| **LGD** | `1 - (total_rec_prncp + recoveries - collection_recovery_fee) / funded_amnt` on defaulted loans | Most simulations assume a round number copied off a Basel slide. Here it is measured |
| **PD** | Calibrated LightGBM | The whole of Layer A |
| **Cohort prior** | Training-window default rate, used to shrink small cohorts | Empirical Bayes, fitted not chosen |

**Why linear fits.** The relationships genuinely are close to linear over LendingClub's 5-31% APR
range, a two-parameter fit cannot overfit, and a coefficient is something that can be read aloud.

**Why `l_bad` decreases with APR.** Not an error: a higher-rate loan collects more interest before
it defaults, so its net loss *on original exposure* is slightly smaller.
`tests/test_decision.py::test_higher_rate_loans_lose_less_when_they_default` asserts the sign.

**Why loss is expressed on original exposure** rather than on balance-at-default: it folds LGD and
the exposure-at-default profile into one directly measurable quantity. Splitting them would require
assuming an amortisation path for defaulters that the public file does not record.

---

## 3. Assumed

These are policy parameters of a hypothetical lender. **None is a property of LendingClub**, and no
attempt is made to pass them off as measured.

| Parameter | Default | Basis | Sensitivity |
|---|---|---|---|
| `annual_cost_of_funds` | 0.03 | Mid-cycle unsecured funding placeholder | **High** — enters every break-even |
| `opex_fixed` | $150/loan | Order-of-magnitude origination + servicing cost | Medium — matters most on small loans |
| `opex_variable_rate` | 1.0% of exposure | Servicing over the life of the loan | Medium |
| `weighted_average_life_years` | 1.55 | **Derived**, not guessed — from the amortisation schedule of a 36-month loan | Low |
| `manual_review_cost` | $75 | Cost of a human underwriter's time | Low (review excluded from headline runs) |
| `review_information_gain` | 0.15 | **Pure assumption** | **High for any policy using review** |
| `review_signal_noise` | 0.45 | **Pure assumption** | High, as above |
| `pd_exposure_elasticity` | 0.15 | Odds multiplier per unit change in exposure ratio | Medium |
| `smoothing_alpha` (cohort) | 25 | Pseudo-count for empirical-Bayes shrinkage | Low |
| Capital budget | swept | The independent variable of EXP06 | By design |

### Weighted-average life is derived, not assumed

Principal on an amortising loan is repaid gradually, so the average dollar is outstanding for far
less than the full term — about **1.55 years on a 36-month loan**. Using the full 3 years would
overstate funding cost by roughly a factor of two, and that error would flow straight into every
break-even PD. `weighted_average_life()` computes it from the schedule; a test asserts it lands
between 1.3 and 1.8 years.

### The manual-review assumption is the weakest link, and it is fenced off

`review_information_gain` says how much a human underwriter learns. **Nothing in this dataset
measures that.** Set it too high and review becomes an oracle: an early version of this project used
0.35 and any policy that used review won by a mile, with a book default rate of 6.8% against 17.2%
for policies that did not — a result that was entirely an artefact of the parameter.

Two defences:

1. The default was reduced to 0.15 with large signal noise, so review is a marginal action.
2. **The headline comparisons (EXP03, EXP06) use `ActionSpace.without_review()`.** The gap between
   two policies should reflect the decision rules being compared, not this parameter. Review is
   analysed separately, on its own terms.

`ExpectedProfitPolicy` additionally *under*-values review by construction — it prices it as "medium
exposure minus the fee" and does not model the option value of being able to decline after
learning. That inconsistency is documented in the code rather than papered over.

---

## 4. Simulated

Clearly labelled as such wherever it appears.

| Quantity | Nature |
|---|---|
| Counterfactual outcomes at unfunded exposures | Conditional coupling — see [`RL_FORMULATION.md`](RL_FORMULATION.md) §5 |
| Episode reward | Simulator output, not a realised P&L |
| Applicant arrival order | Random permutation of the out-of-time window |
| Refined PD after review | Model of a human underwriter |

**The synthetic data generator** deserves its own line. When `data/raw/SYNTHETIC` exists, every
artefact carries `data_source: synthetic` in its manifest, every report is stamped with a banner,
and no number computed from it is a finding about consumer credit. Its risk structure was written
by hand — including modest per-ZIP and per-employer cohort effects, deliberately kept small so that
the graph layer has structure to find without manufacturing a positive result for Layer B.

---

## 5. Assumptions that would change the conclusions

Stated because an interviewer will ask "what would break this?".

**If the cost of funds were 8% rather than 3%**, break-even PDs fall sharply, the approve-heavy
expected-profit policy becomes much more conservative, and the gap to a fixed threshold narrows.
The ranking should survive; the magnitude will not.

**If LGD were 0.85 rather than the ~0.6 estimated here**, the same applies more strongly. Real
LendingClub LGD estimates in the literature often run higher than what falls out of this
calculation, partly because of how recoveries and fees are attributed.

**If the exposure elasticity were much larger**, lending 1.5x would raise PD enough to make the
large-exposure action unattractive, and the action set would effectively collapse to
{reject, small, medium}.

**If the applicant pool were endogenous** — if declining someone changed who applied next month —
the episode structure would be wrong and the whole MDP would need re-specifying.

**If repeat borrowers are common**, rows are not independent, and every confidence interval in this
project is slightly too narrow. `member_id` is scrubbed, so this cannot be detected or corrected.

---

## 6. Where each assumption lives in code

| File | Assumptions |
|---|---|
| `decision/expected_loss.py` | `EconomicsConfig` — costs, funding, WAL |
| `simulation/environment.py` | `SimulationConfig` — review model, exposure elasticity, budget |
| `graph/features.py` | `CohortFeatureConfig` — lookback, smoothing |
| `data/synthetic.py` | `SyntheticSpec` — the entire synthetic risk structure |
| `configs/simulation/base.yaml` | Every runtime override, in one readable file |

Every one is a dataclass field with a comment giving its basis, so a reader can find what a number
means without tracing it through the call stack.
