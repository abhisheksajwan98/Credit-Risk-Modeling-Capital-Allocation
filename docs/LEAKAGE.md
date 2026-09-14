# Leakage

Leakage is treated as a first-class concern in this project, ahead of squeezing out another
half-point of AUC. The reason is that leakage does not fail loudly. It produces an excellent
validation score and a worthless model, and the failure only surfaces after deployment, on real
money.

Every rule below has a corresponding assertion in `tests/test_leakage.py` or
`tests/test_graph_leakage.py`. They raise; they do not warn.

---

## 1. The test that decides everything

> A column is a valid feature **if and only if its value was already determined at the moment the
> lender had to decide.**

"Correlates with default" is *not* the test. `last_fico_range_low` correlates enormously with
default and is worthless, because you only learn it afterwards.

LendingClub is a notoriously leaky dataset precisely because roughly a third of its columns are
recorded after origination, and they sit in the same flat CSV as the application fields with no
marker distinguishing them.

---

## 2. Target leakage: post-origination columns

These are **outcome** fields. The project uses them to build the target and the realised cashflows,
and a hard deny-list in `src/credit_risk/data/schema.py` stops them reaching any model.

| Column | Why it leaks |
|---|---|
| `last_fico_range_low`, `last_fico_range_high` | A bureau score pulled *after* origination. It collapses for borrowers who are about to default. This single column takes AUC to roughly 0.95 and predicts nothing knowable at decision time. **The most seductive leak in the dataset.** |
| `recoveries`, `collection_recovery_fee` | Money collected *after* charge-off. Non-zero implies default. |
| `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Cumulative repayment. A loan that repaid its full principal did not default. |
| `out_prncp`, `out_prncp_inv` | Outstanding balance. Zero on a matured loan implies it ran to term. |
| `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Payment timing. An early last payment implies default. |
| `last_credit_pull_d` | Date of a post-origination bureau pull. |
| `loan_status` | The target itself. |
| `pymnt_plan`, `hardship_*`, `debt_settlement_flag`, `settlement_*` | Distress and workout programmes. These *are* the default process. |

Enforcement: `schema.assert_no_leakage()` is called on **every** feature-matrix build and raises
on any offender. `schema.CASHFLOW_COLUMNS` names the narrow subset the economics layer may read,
so that permission is visible rather than implicit.

---

## 3. The subtle one: information the lender created

Three columns are knowable at decision time but still need care.

**`funded_amnt`** — the exposure actually taken. It is knowable, because it *is* the decision. But
using it as a PD feature conflates the action with the risk, so it is routed to the economics layer
only. `loan_amnt` (the amount *requested*) is the feature.

**`grade`, `sub_grade`, `int_rate`** — LendingClub's own risk grade and the price derived from it.
Legitimately available, but a model built on them is partly reverse-engineering an existing
scorecard rather than underwriting from primary evidence. Hence two feature sets are reported
throughout:

- `primary` — excludes them. *"Can we underwrite from primary evidence?"*
- `with_lc_grade` — includes them. *"Can we beat the incumbent scorecard?"*

`int_rate` is always available to the **economics** (it determines revenue) even when excluded from
the **PD features**. Those two roles are kept separate deliberately.

**`installment`** — this one is easy to miss. It is knowable at decision time, but it is an
*invertible function* of `loan_amnt`, `int_rate` and `term`:

```
installment = P * r / (1 - (1+r)^-T)
```

Given the other three you can recover the interest rate by bisection to well inside the granularity
of a risk grade — `tests/test_leakage.py::test_interest_rate_is_recoverable_from_instalment`
demonstrates this to within 0.05 percentage points. So including `installment` in the `primary`
feature set would smuggle the incumbent scorecard into a model that claims not to use it. It is
excluded, along with the derived `est_payment_to_income` and `dti_post_loan`.

---

## 4. Temporal leakage

**No random splits anywhere.** Every split is a contiguous window of `issue_d`:

```
train   2010-01 .. 2014-06
valid   2014-07 .. 2014-12
test    2015-01 .. 2015-12
```

A random split lets a model see loans issued in the same month as the ones it is scored on, and
silently exploit knowledge of that month's credit conditions that it could not have had. Credit
performance moves with the vintage and the macro cycle, so this inflates every metric.

`assert_temporal_ordering()` fails the run if any training row was issued after any test row, and
`SplitWindows.validate()` refuses overlapping windows.

**Everything fitted is fitted on train only** — encoders, imputation values, clip bounds,
categorical vocabularies, the cohort smoothing prior. The one exception is the **calibrator**,
which is fitted on `valid`, because a calibrator fitted on the model's own in-sample scores learns
a map that is wrong everywhere else.

**Derived features are measured as at the application date.** `credit_history_months` is
`issue_period - earliest_cr_line`, not "months to today". Computing it against today's date would
put the vintage directly into the feature.

---

## 5. Availability leakage: the trap that has no offending column

This one has no single guilty column, which is what makes it hard to spot.

LendingClub introduced a large bureau block (`num_*`, `mo_sin_*`, `mths_since_recent_*`, `bc_*`,
`tot_hi_cred_lim`, ...) partway through its history, and a second block (`open_il_*`, `all_util`,
`inq_fi`, ...) around 2015-12. In earlier vintages these are simply absent.

So **"is this column populated?" is partly a statement about the origination date.** A model will
learn that association happily — and it will not reproduce in production, where every field is
populated.

Two defences:

1. `schema.LATE_ADDITION_NUMERIC` and `schema.JOINT_APPLICATION_COLUMNS` are excluded by default:
   in the 2010-2015 window they are near-entirely null, so they carry no signal and only risk
   acting as a vintage indicator.
2. **The availability guard.** `FeatureBuilder` measures each feature's missingness in `train` and
   in `valid`, and drops any feature whose missingness moves by more than
   `availability_shift_tolerance` (default 5 percentage points).

The guard is measured against **validation, never test**. Looking at test missingness to make a
modelling decision would spend the out-of-time set. `prepare_data.py` additionally reports the
train-to-test availability shift in `results/reports/data_profile.md` as a diagnostic, but nothing
acts on it.

---

## 6. Graph leakage

The graph is where a credit model leaks most easily and least visibly, because the leak travels
along an edge rather than sitting in a column. Two rules, enforced separately.

### Rule 1 — every edge points strictly backwards

An edge `j -> i` requires `issue_period[j] < issue_period[i]`. Strictly: same-month loans are
**not** connected, because within a month there is no ordering to appeal to and a mutual edge would
let two simultaneous applications each see the other.

### Rule 2 — neighbour *labels* require the neighbour's outcome to have resolved

This is much stricter than Rule 1, and conflating the two is the standard failure mode.

| Aggregate | Condition on neighbour `j` | Rationale |
|---|---|---|
| Feature aggregates (mean FICO, mean DTI, counts) | `issue[j] < issue[i]` | The application existed and was observable |
| **Label** aggregates (cohort default rate) | `resolved[j] <= issue[i]` | The outcome had actually happened *and been booked* |

A 36-month loan issued 2013-01 does not resolve until 2016-01. A borrower applying in 2014 cannot
know how it ended. Using `issue[j] < issue[i]` for a *label* aggregate silently imports outcomes
from the same period the model is scored on.

`resolved[j]` is derived as: last payment date, plus five months for charged-off loans, because
LendingClub books a charge-off at 150 days delinquent. Loans that have **not** reached a terminal
status have no resolution date at all — leaving them NA is what stops an unresolved loan being
counted as a "good" neighbour merely because it has not defaulted *yet*, which would bias every
cohort rate downward.

**How much does this matter?** `test_ungated_cohort_rate_leaks_measurably` answers it
quantitatively rather than asserting it. On a controlled frame where each cohort's risk is redrawn
every 24 months, the correctly-gated cohort rate and the naively-gated one are compared for
correlation with the target. The naive version is substantially more correlated, and every bit of
that excess is leakage.

The scale of the restriction is reported by `build_graph.py` as the *information gap*: typically
only around 40% of a borrower's cohort predecessors have a known outcome at the time they apply.

---

## 7. Leakage this project does *not* claim to have solved

**Selection bias.** Every row in the accepted file passed LendingClub's underwriting. The model
estimates `P(default | approved)`, not `P(default | applied)`. Applying it to the full applicant
population is extrapolation. This is quantified against the rejected-applications file, but reject
inference is **not** implemented as a fix, because doing it properly requires propensity
information that is not published. Pretending otherwise would be worse than naming the limitation.

**Survivorship in the monitoring window.** Loans issued from 2016 have not matured. Some have
already charged off, and keeping only those would select for fast defaults and bias the rate
upward. So the monitoring window carries **no** modelling label at all — `default` is NA there,
even for rows that happen to have resolved. `default_observed` is populated separately and is used
only for graph label aggregates, where a resolved-early outcome is legitimately knowable.

**Duplicate borrowers.** `member_id` is scrubbed, so a borrower who took two loans appears as two
independent rows. If repeat borrowers are common this understates correlation between rows and
makes confidence intervals slightly too narrow. There is no way to detect or correct it in the
public data; it is stated rather than solved.
