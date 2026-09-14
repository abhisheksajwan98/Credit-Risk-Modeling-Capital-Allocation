# Interview guide

A Q&A companion to the codebase. The guiding philosophy is *small theoretical surface area, fully
explainable* — everything below should be defensible under follow-up questions, and the numbers are
the ones the current code produces.

**How to pitch this project.** Not as *"I used PyTorch and GNNs to beat LightGBM"* — that invites
scepticism and the GNN does not, in fact, beat LightGBM here. Pitch it as:

> *"I built a credit decision system on real loan data, then audited it hard enough to overturn one
> of my own headline results."*

The strongest thing in this repository is not a model. It is that several experiments return null
or negative results, and one previously-reported conclusion was found to be an artefact and
corrected. That is what distinguishes a project from a demo.

---

## 1. Dataset and scope

**Q: Why LendingClub and not Home Credit?**

Home Credit was the obvious choice and I rejected it for three specific reasons. It contains **no
absolute calendar date anywhere** — every temporal field is expressed in days relative to each
application — so out-of-time validation, vintage analysis and drift monitoring are all impossible.
It has **no borrower-to-borrower links**, only borrower-to-own-records, so any graph would be a
fabricated k-nearest-neighbour construction. And it has **no interest rate or recovery amounts**, so
the entire decision layer's reward would be invented. LendingClub gives real dates, real pricing and
real cashflows. Home Credit is retained as a Layer-A-only adapter to show the pipeline generalises
across schemas.

**Q: Why 36-month loans issued 2010–2015?**

So that every outcome is fully matured. A 36-month loan issued 2015-12 resolves by 2018-12, the end
of the data. That removes censoring entirely, and restricting to one term removes the term-mix
confound between vintages. It costs volume; it buys an unambiguous target.

**Q: Why out-of-time splits rather than cross-validation?**

Credit performance moves with the vintage and the macro cycle. A random split lets the model see
loans issued in the same month as the ones it is scored on and exploit knowledge of that month's
credit conditions. In this dataset the default rate rises 12.67% → 14.13% → 14.89% across
train/valid/test — that is real deterioration a random split would have hidden.

---

## 2. Leakage

**Q: What is the most dangerous leak in LendingClub?**

`last_fico_range_low` and `last_fico_range_high` — FICO scores pulled *after* origination. A borrower
who is defaulting has a collapsed score. Including them gives an AUC around 0.95 and a model that is
useless for a decision made at origination. There are 49 such post-origination columns and the
feature builder raises on any of them.

**Q: What is the subtlest one?**

`installment`. It is knowable at decision time, so it looks safe — but it is an invertible function
of `loan_amnt`, `int_rate` and `term`. Given the other three you can recover the interest rate by
bisection to within 0.05 percentage points, and the interest rate *is* LendingClub's risk grade. So
including it in the "primary" feature set would smuggle the incumbent scorecard into a model
claiming to underwrite from primary evidence. There is a test that demonstrates the inversion.

**Q: And one with no guilty column at all?**

Availability leakage. LendingClub introduced whole bureau blocks partway through its history, so
*"is this column populated?"* is partly a statement about the origination date. A model learns that
happily and it does not reproduce in production. The feature builder measures each column's
missingness in train versus validation and drops anything that shifts more than 5 points — measured
against **validation, never test**, because using the test set to make a modelling decision spends it.

---

## 3. The graph

**Q: How do you build a graph without counterparty transaction data?**

You admit you do not have one. LendingClub scrubbed `member_id` and publishes no counterparty links,
so there is no observed transaction network. What exists is **shared-attribute structure**, and I
built a *cohort graph*: borrowers are connected when they share a 3-digit ZIP prefix (~900 cohorts)
or a normalised employment string.

The economic rationale is real — credit risk is not i.i.d. across such groups, which is exactly why
lenders carry geographic and industry concentration limits. But I say plainly what it can and cannot
test: it can test whether *correlated-risk* structure helps; it cannot test whether a genuine
counterparty network would.

One honest defect: LendingClub's `emp_title` held **employer name before 2013-09-23 and job title
after**, so that relation changes meaning mid-history. It is the secondary relation, it is ablated
separately, and every row carries a flag for which regime it belongs to.

**Q: How do you stop information travelling backwards through the graph?**

Two rules, enforced separately, and the distinction is the heart of the design:

- **Neighbour features** need only `issue[j] < issue[i]` — strictly earlier, so same-month loans are
  never connected.
- **Neighbour labels** additionally need `resolved[j] <= issue[i]`. A 36-month loan issued 2013-01
  does not resolve until 2016-01, so a 2014 applicant's underwriter cannot know how it ended.

Conflating those two is the standard way graph credit models leak. In practice only about 40% of a
borrower's cohort predecessors have a known outcome at application time, and that gap is reported.
There is a test that builds the cohort rate both ways and shows the ungated version is
substantially more correlated with the target — that excess correlation is the leak, quantified.

**Q: Did the graph help?**

Not for discrimination. GraphSAGE scored **0.6616** AUC against 0.6679 for hand-built cohort
aggregates and 0.6690 for plain tabular — the GNN is the *worst* of the three arms. I report that
rather than tuning until it wins, and the explanation is straightforward: mean-aggregation over a
one-hop cohort is close to what the aggregate features compute in closed form, so the GNN
re-learns a known function from less signal with more parameters.

Where the graph *did* pay is calibration. With matched model configurations, adding the cohort
features left AUC flat but moved **ECE from 0.0105 to 0.0032** and observed/expected from 0.930 to
0.999. For a decision engine that multiplies PD by an exposure, that is the property that matters —
which is a better answer than a decimal place of AUC would have been.

---

## 4. Calibration

**Q: Why is a well-ranked model not necessarily a good probability model?**

Because AUC is invariant to any strictly monotone transform of the score. Cube every prediction: the
ordering is untouched, so AUC, Gini and KS do not move at all, while every predicted PD is now wrong
and the expected loss computed from them is wrong with it. There is a test that does exactly this.

In this project Platt scaling left out-of-time AUC **unchanged to five decimal places** while cutting
ECE from 0.0193 to 0.0111 and moving the calibration slope from 0.991 to 1.005. The decision layer
multiplies PD by an exposure, so it needs the second property, and no ranking metric can see it.

**Q: Why fit the calibrator on validation rather than train?**

A calibrator fitted on the model's own in-sample scores learns a map from over-fitted predictions and
is wrong everywhere else.

---

## 5. The decision layer

**Q: Why is a fixed PD threshold the wrong shape?**

Because it assumes every loan carries the same price. The break-even PD moves with the interest
rate — a 26% APR loan can carry far more risk than a 7% one and still be profitable — so a single
cut-off simultaneously rejects profitable high-rate business and accepts unprofitable low-rate
business. That is a failure of the *decision rule*, not the risk model; every policy consumes
identical PDs.

**Q: Did the expected-profit policy actually beat the threshold?**

On this book, **no** — +12,991 per episode with a 95% interval of [−17,702, +44,103], which includes
zero. A null result, and reported as one. The mechanism is sound and the effect was not detectable
here.

**Q: Where do the economics come from?**

Estimated from observed cashflows on the training window, not assumed. The empirical LGD is
**0.518** on the test book, measured as
`1 - (total_rec_prncp + recoveries - collection_recovery_fee) / funded_amnt`. Cost of funds, opex and
the review model *are* assumptions and are itemised separately in `docs/ASSUMPTIONS.md`. The
weighted-average life of 1.55 years is derived from the amortisation schedule, not guessed — using
the full 3 years would roughly double funding cost.

---

## 6. Reinforcement learning — and when not to use it

**Q: Why is this RL rather than supervised learning?**

Two reasons. You never observe what a rejected applicant would have repaid, so there is no
supervised target for "was rejecting correct?" — that is the bandit setting by definition. And under
a **binding capital constraint** the problem becomes sequential: funding a marginal applicant today
consumes capital a better applicant tomorrow cannot use.

**Q: When does RL buy nothing?**

With unconstrained capital. There the myopic per-applicant argmax of expected profit is optimal, and
I built that in as a **control**: if a learner beats the myopic policy in the unconstrained
condition, something is wrong.

That control fired. A LinUCB policy beat the myopic rule by 35% with an interval excluding zero — and
on audit there were two causes, both real. The learning policies were being **trained on the
evaluation pool**, so they had seen the realised outcomes of the borrowers they were scored on, while
the static policies had not. And the myopic policy optimises coefficients fitted on the 2010–2014
window applied to a 2015 book, so it is optimal for economics that are stale. Learning policies now
train on the validation window, and the control check is computed and printed in the report.

This is the answer I would most want to be asked about: the useful thing was not the RL, it was
having a falsifiable control and acting on it.

**Q: Why tabular Q-learning and not PPO or DQN?**

Because the state space is 160 discrete cells and there is no approximation error to remove. The
resulting Q-table can be printed and read, and its implied approve/decline threshold should track
the analytical shadow price of capital — which is computable in closed form by continuous knapsack
relaxation. Having an analytical reference is what makes the RL result checkable rather than
mysterious.

I also report that Thompson sampling **collapsed** onto always-reject (0.06% approval rate). The
reward is roughly +1 with probability 0.85 and −8 with probability 0.15, which a 7-feature linear
model fits poorly. I report it as degenerate rather than quietly dropping it.

---

## 7. The blind-spot layer

**Q: What is the research question here?**

Whether representation-learning signals can identify where the conventional model is unreliable —
and whether acting on that improves decisions. Two separate claims needing separate evidence.

**Q: How do you know a signal is not just restating the PD?**

This is the question that broke the first version of these experiments. Both signals were originally
supported by a marginal quintile table — bin by the signal, show calibration degrading across bins.
But the mean PD also rose across those bins, and a boosted model is *already* less well calibrated at
high PD. The table cannot separate the two.

So I added a **conditional test**: within each PD decile, split at the median signal and compare the
calibration gap, with a binomial sign test across strata and a logistic regression of the outcome on
`logit(PD)` plus the signal.

| Signal | Correlation with PD | Strata passing | Sign-test p | Odds ratio / SD | Decision |
|---|---|---|---|---|---|
| Autoencoder anomaly | +0.022 | **10 / 10** | 0.0010 | 1.032 | KEEP |
| Model disagreement | −0.150 | 9 / 10 | 0.0107 | 1.043 | KEEP |
| VGAE uncertainty | — | — | — | — | pending re-run |

Both survive, but the odds ratios are small. They flag **miscalibration**, not bad borrowers, and I
say so rather than overselling.

**Q: You measured disagreement wrongly at first. How?**

On the raw probability scale, `|p_a - p_b|` is mechanically larger in the middle of the PD range, so
it partly restates the PD level — the original signal correlated with PD at **+0.32**. Measured on
the log-odds scale with matched model configurations, that becomes **−0.150** (the same signal on
the probability scale still correlates at +0.258). The two models also differed in hyper-parameters and calibrator as well as
features, so "disagreement" was partly measuring those choices. Both arms are now fitted identically
with the feature set as the only difference.

---

## 8. The result that reversed

**Q: What did the audit find?**

EXP11 asked whether rejecting flagged borrowers improves the book. It reported:

> *"the portfolio realized a **$-9,794,019 net increase in profit**. This proves that representation
> learning improves decision quality."*

Three things wrong with that sentence. The number is **negative** and it is called an *increase* —
the conclusion was a fixed string asserted regardless of sign. The "realized profit" was not
realised at all: it came from a formula `amount × apr × 1.55 − amount × 0.5 × default` with an
assumed LGD, when the repository already computes true realised cashflow. And it compared the two
**swap sets**, which deploy different capital, rather than the two portfolios.

## 2. The Decision Engine (EXP11)

When integrating the blind-spot signals into a capital-constrained Portfolio Simulator, the naive expectation was that rejecting "uncertain" borrowers would massively increase profit.

*   **The Artefact:** An early run claimed a $9.8M profit shift, but this was an artefact of an assumed-LGD formula and a naive swap-set comparison.
*   **The True Finding:** When evaluated properly using a knapsack greedy-fill approach on actual observed cashflows, the profit difference between the policies is negligible (a fraction of a percentage point).
*   **The Business Value:** However, what *does* survive is a massive risk reduction. By penalising the expected profit of the highly anomalous/disagreeing borrowers, the decision engine reallocated capital and reduced the portfolio's overall default rate from **17.3% to 16.3%** (a 100 basis point drop) for free, at constant return on capital.

---

## 9. Questions I would ask this project

Worth rehearsing, because a good interviewer will.

- *"Your AUC is 0.67. Is that good?"* For unsecured consumer credit on application-time data only,
  yes — LendingClub's own grade is in the data and I deliberately exclude it from the primary
  feature set. Published Lending Club models reporting 0.70+ usually include `sub_grade`.
- *"Only 40% of cohort neighbours have known outcomes. Is the graph worth it?"* For discrimination,
  no. For calibration, yes. I would not ship the GNN.
- *"You trained on 2010–2014 and tested on 2015. What happens in 2016?"* Unknown, and the drift
  experiment suggests it degrades — observed/expected moves 0.860 to 1.077 across vintages while AUC
  falls 0.7475 to 0.6705. Calibration drifts before ranking does, which is why a monitoring regime
  watching only AUC would miss it.
- *"Would you deploy this?"* No. It is trained on an already-approved population, so it estimates
  `P(default | approved)`; it has never seen a downturn; and the policy results come from a
  simulator whose assumptions are listed. I would deploy the *monitoring*, not the decision engine.
