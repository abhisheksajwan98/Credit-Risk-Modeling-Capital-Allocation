# Reinforcement learning formulation

## 1. Why this is RL and not supervised learning

Worth being blunt about, because "we applied RL to lending" is usually decoration.

Predicting default **is** supervised learning: there is a label, and the prediction does not change
it. *Deciding* is not, for two distinct reasons.

### Reason 1 — the label you need is counterfactual

You never observe what a rejected applicant would have repaid. There is no supervised target for
"was rejecting correct?". You observe a reward for the action you took and nothing for the others.
That is the **bandit** setting by definition — not an analogy to it.

### Reason 2 — under a capital constraint the problem is genuinely sequential

This is the part that earns "RL" rather than just "bandit", and it is worth stating as a falsifiable
claim rather than a motivation:

> **With an unconstrained balance sheet, the optimal policy is exactly the myopic per-applicant
> argmax of expected profit, and RL cannot beat it.**

Approving a good loan never costs you anything elsewhere, so there is no trade-off across time.
We expect this null result and we report it — see the `unconstrained` column of EXP06.

When capital **binds**, funding a marginal applicant today consumes capital that a better applicant
tomorrow cannot use. The decision now depends on how much budget is left and how many applications
remain. That is a Markov decision process, and the myopic rule is provably suboptimal in it.

---

## 2. The MDP

| Element | Definition |
|---|---|
| **State** | `(applicant context, remaining capital, applications remaining in the period)` |
| **Action** | `{reject, approve_small (0.5x), approve_medium (1.0x), approve_large (1.5x)}`, with `review` available but excluded from headline comparisons |
| **Reward** | realised profit: `exposure x return_per_dollar - funding cost - operating cost` |
| **Transition** | capital decreases by the exposure taken; the next applicant arrives |
| **Horizon** | one funding period — 1,000-2,000 applications. Finite and short |
| **Discount** | `gamma = 1`. Undiscounted: the horizon is a few months and every dollar inside it is worth the same. Discounting would distort the objective for no reason |

Exposure is expressed as a **multiple of the amount requested**, so the same action set applies to a
$2,000 and a $35,000 application. Lending *more* than requested is included because it is a real
product decision — a lender that can only say yes or no leaves money on the table with its best
borrowers.

### State discretisation

Deliberately small and discrete: 8 x 5 x 4 = 160 states.

- **`profit_density`** (8 bins) — expected profit per dollar of exposure at medium exposure.
  Chosen over raw PD because profit density is the *sufficient statistic* for the decision: it
  already folds in the interest rate. A fixed PD threshold's failure to do so is exactly what
  EXP03 exposes.
- **`budget_fraction`** (5 bins) — capital remaining.
- **`time_fraction`** (4 bins) — applications remaining.

Three reasons for keeping it tabular: it trains in seconds, it needs no function approximator for a
problem with no approximation error to remove, and **the resulting Q-table can be printed and
read**. Being able to show an interviewer the learned policy as a grid is worth more than a
marginally better score from a DQN.

---

## 3. The shadow price of capital

The constrained problem has known structure, which is what makes the RL result *checkable* rather
than mysterious. Maximising total profit subject to `sum(exposure) <= B` has the Lagrangian:

```
max  sum_i [ profit_i(a) - lambda * exposure_i(a) ]
```

so the optimal rule is: **approve when profit per dollar of capital exceeds `lambda`**, the shadow
price — the marginal profit of the last dollar. The myopic policy is the special case `lambda = 0`.

A learned value function has to recover exactly this: `V(budget, t)` is the expected future profit
from the remaining budget, and its derivative with respect to budget *is* `lambda`.

This gives the experiment a built-in check that most RL write-ups lack:

- `analytical_shadow_price()` computes the exact `lambda` by continuous knapsack relaxation —
  sort by profit density, fill the budget, and `lambda` is the marginal applicant's density.
  It requires knowing the whole pool in advance, so it is an **upper-bound reference**, not a
  competitor.
- `ExpectedProfitPolicy(capital_shadow_price=lambda)` is that solution as a policy.
- `BudgetAwareQPolicy` must approximately recover it *online*, without seeing the pool in advance.

**If the learned policy substantially exceeds the analytical-`lambda` policy, that is a bug in the
simulator, not a breakthrough.** That sentence is the point of including the analytical reference.

`BudgetAwareQPolicy.implied_threshold()` reads the learned approve/decline boundary out of the
Q-table per (budget, time) cell. A threshold that *rises as budget tightens* is the shadow price
appearing in the learned policy, and that qualitative signature is what EXP06 is really testing for.

---

## 4. Algorithm choice follows the formulation

No PPO, no DQN, no actor-critic. The problem does not call for them and they would be harder to
defend.

| Setting | Algorithm | Why |
|---|---|---|
| One-step (unconstrained) | **LinUCB**, **Thompson sampling** | The canonical contextual-bandit answers. A few dozen lines of linear algebra each |
| Sequential (binding capital) | **Tabular Q-learning** | 160 discrete states solved exactly; the result is a readable grid |

**LinUCB** maintains `A_a = lambda I + sum x x'` and `b_a = sum r x` per action, and picks
`argmax theta_a . x + alpha sqrt(x' A_a^-1 x)` — estimated reward plus the width of its confidence
interval. Actions rarely tried in this region of context space have wide intervals and get chosen
*because* they are uncertain. Optimism in the face of uncertainty. Updates use the
Sherman-Morrison rank-one identity, O(d²) rather than an O(d³) re-inversion.

**Thompson sampling** draws `theta_a` from its posterior and acts greedily against the draw, so an
action is chosen in proportion to the posterior probability that it is best. Usually slightly better
in practice with one fewer parameter to tune.

The context vector is seven interpretable dimensions (intercept, PD as scaled log-odds, APR, log
size, profit density, budget fraction, time fraction). A linear bandit with hundreds of features
needs far more data than one funding period provides, and its exploration behaviour becomes
impossible to reason about.

---

## 5. The simulator, and the assumption at its centre

We observe each borrower's outcome at exactly **one** exposure — the amount LendingClub actually
funded. Evaluating a policy that would have lent a different amount requires an assumption.
Pretending otherwise is the central dishonesty in most "RL for lending" write-ups.

### Conditional coupling

Each applicant gets one uniform draw `u`, held fixed across every action, and defaults under action
`a` iff `u < PD(a)`. The draw is sampled **conditional on what actually happened**:

```
borrower actually defaulted:   u ~ Uniform(0, PD_observed)
borrower actually repaid:      u ~ Uniform(PD_observed, 1)
```

Two properties follow, and both matter:

1. **At the observed exposure the simulator reproduces reality exactly.** No borrower who repaid is
   made to default by the simulation. `tests/test_simulation.py::test_coupling_reproduces_the_observed_outcome`
   asserts this over hundreds of applicants — if it ever fails, every policy comparison is running
   against invented outcomes.
2. **Common random numbers.** Because `u` is shared across actions and seeds are shared across
   policies, comparing two policies is a comparison of *decisions*, not of luck. This makes the
   paired comparison far tighter than independent sampling would.

### Realised return, observed where possible

Where the simulated outcome matches what actually happened, the **observed** realised return is
used — a measured quantity from real cashflows. Only where the coupling has flipped the outcome
(because a different exposure moved PD across `u`) does the fitted cashflow model supply a
counterfactual. Preferring the observation wherever one exists is what keeps the simulator anchored
to reality rather than to a model of it.

### What remains assumed

1. **Exposure affects PD only through affordability.** A counterfactual amount is re-scored through
   the same PD model with `loan_to_income` and `dti_post_loan` updated. Real demand effects and
   adverse selection are not modelled.
2. **Realised return per dollar is roughly invariant to exposure.** Reasonable for moderate
   changes, and the reason the action set tops out at 1.5x rather than 5x.
3. **The applicant pool is exogenous.** Rejecting someone does not change who applies next month.
   Fine over one funding period; false over years.
4. **Manual review reveals something.** `review_information_gain` (default 0.15, with large signal
   noise) is a pure assumption — nothing in the data measures how good a human underwriter is. Set
   it high and review becomes an oracle, and any policy using it wins for the wrong reason. This is
   precisely why **the headline comparisons use `ActionSpace.without_review()`**: the gap between
   two policies should reflect the decision rules, not this parameter.

None of this makes the simulator a truth machine. It makes it a *consistent testbed in which
policies can be compared to each other* — a much weaker and much more defensible claim.

---

## 6. Evaluation discipline

- **Common random numbers** across every policy and seed.
- **Paired bootstrap confidence intervals on the difference**, not on the level. The quantity of
  interest is never "policy A earned X" but "policy A earned X more than policy B, and here is the
  interval". An interval straddling zero is a null result and is reported as one.
- **Worst case alongside the mean.** Minimum and 5th-percentile episode reward are reported,
  because a policy that earns more on average while occasionally destroying the book is not better.
  In lending the left tail is the whole subject.
- **Training seeds are offset far from evaluation seeds**, so a policy cannot be rewarded for having
  memorised the applicant sequence it will be scored on.
- **Sensitivity sweeps** over the assumptions. A policy ranking that flips when the cost of funds
  moves 100bp is not a finding.

---

## 7. What this establishes, and what it does not

**Establishes:** under the stated assumptions, in this simulator, on this population, these policies
rank in this order, with these confidence intervals, and the ranking changes with the capital
constraint in the direction theory predicts.

**Does not establish:** that any of these policies is safe to deploy.

The gap between those two statements is large and worth naming explicitly:

- The population is **already-approved** LendingClub borrowers. A policy that approves more than
  LendingClub did is extrapolating into a region with no data.
- One institution, one credit cycle, a long expansion. Nothing here has been tested through a
  downturn.
- Off-policy evaluation on the logged data (IPS, doubly-robust) is **not** performed, because
  LendingClub's own underwriting propensities are unknown and unpublished. Without them, an
  importance-weighted estimate would be arbitrary.
- Exploration in a simulator is free. In production it means **lending money to someone the model
  says to decline**, on real borrowers, with real losses — and the applicants selected for
  exploration are by construction the marginal ones, which raises fairness questions of its own.
  The exploration here demonstrates the mechanism; it is not a recommendation to explore on live
  applicants without a great deal more thought.
