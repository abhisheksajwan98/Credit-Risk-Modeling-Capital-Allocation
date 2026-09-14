# Dataset

Evaluated and selected 2026-08-26. Figures marked *(verify)* are populated by
`scripts/prepare_data.py`, which writes the measured values to
`results/reports/data_profile.md`. Nothing in this file is taken on trust from a blog post.

---

## 1. Selection

**Selected: LendingClub accepted + rejected loan data, 2007 - 2018Q4.**

| Field | Value |
|-------|-------|
| Source | Kaggle dataset `wordsforthewise/lending-club` ("All Lending Club loan data") |
| URL | https://www.kaggle.com/datasets/wordsforthewise/lending-club |
| Uploader | Nathan George (`wordsforthewise`) |
| **Licence** | **CC0: Public Domain** (verified via Kaggle API, 2026-08-26) |
| Version | 3 |
| Last updated | 2019-04-10 |
| Total size | 1,356,507,910 bytes (~1.36 GB compressed) |
| Origin | Mirror of the CSVs LendingClub formerly published on its own site |

### Why this dataset

The project needs one dataset that supports *all three* layers. Ranked by how much each property
mattered to the decision:

1. **Real calendar time.** `issue_d` gives the month and year of origination, 2007-06 to 2018-12.
   This is what makes out-of-time validation, vintage analysis and the drift experiment (EXP07)
   possible at all.
2. **Real loan economics.** `int_rate`, `installment`, `term`, `funded_amnt`, `total_pymnt`,
   `total_rec_prncp`, `total_rec_int`, `recoveries` and `collection_recovery_fee` let the decision
   layer compute realised profit and an **empirically estimated LGD** from observed cashflows.
   Without these, Layer C's reward would be a number invented to make the RL agent look good.
3. **Usable relational axes.** 3-digit ZIP prefix (~900 values) and a free-text employment string
   give genuine borrower-to-borrower cohort structure with a defensible economic story
   (correlated local labour-market and housing shocks).
4. **A rejected-applications file.** 27M+ rejected applications with a thin feature set make the
   selection-bias problem *visible and measurable* rather than a footnote.
5. **Permissive licence.** CC0 means no restriction friction on a public portfolio repository.

### Alternatives evaluated and rejected

| Dataset | Why not |
|---------|---------|
| **Home Credit Default Risk** (Kaggle) | The nominated primary candidate. **Rejected** because it has no absolute calendar date anywhere — every temporal field is `DAYS_*` relative to each application — so a temporal split, vintage analysis and drift measurement are impossible. It also has no borrower-to-borrower relations (only borrower-to-own-records), forcing a fabricated similarity graph, and no interest rate or recovery amounts, forcing an invented RL reward. **Retained as a secondary adapter for Layer A only** (`src/credit_risk/data/home_credit.py`). |
| **Give Me Some Credit** (Kaggle) | 150k rows, 10 features, no dates, no loan amounts, no relations, no cashflows. Supports Layer A only, and thinly. Rejected. |
| **Lending Club (other Kaggle mirrors)** | Several exist; most are undocumented subsets, some mix rejected rows into accepted ones. `wordsforthewise` is the fullest and the only one with a stated CC0 licence and reproducible build code. |
| **Freddie Mac / Fannie Mae single-family loan-level** | Genuinely attractive: real seller and servicer identifiers give a true bipartite institution graph, plus real net-loss amounts. Rejected on friction and fit — requires registration and acceptance of a redistribution-restricted licence, and 30-year mortgages have a default horizon far too long for a laptop-scale decision simulator. Documented here as the natural upgrade if this project were extended. |
| **UCI Taiwan default of credit card clients** | 30k rows, six months of repayment history, no relations, no amounts. Useful as a teaching set, too small here. |

---

## 2. Obtaining the data

Raw data is **not committed** to this repository (size, and it is not ours to redistribute even
under CC0 — the download is one command).

### One-time credential setup

1. Sign in at https://www.kaggle.com and open **Settings -> API -> Create New Token**.
2. That downloads `kaggle.json`. Move it to `C:\Users\<you>\.kaggle\kaggle.json`
   (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS).
3. `kaggle.json` is listed in `.gitignore`. It must never be committed.

### Download

```bash
python scripts/download_data.py --config configs/data/lending_club.yaml
```

The script verifies credentials, downloads into `data/raw/`, records the SHA-256 of each file in
`data/raw/checksums.json`, and refuses to re-download if checksums already match.

### No credentials yet?

The full pipeline runs without the real data:

```bash
python scripts/download_data.py --synthetic --n-loans 60000
```

This writes a **schema-faithful synthetic LendingClub extract** to `data/raw/` — same column names,
dtypes, categorical levels, missingness pattern, vintage growth and a plausible risk structure.
It exists so the pipeline is testable and CI-able before the real download lands. Every artefact
produced from it is stamped `data_source: synthetic` and **no result computed on synthetic data is
reported as a finding.**

---

## 3. Files

| File | Rows | Cols | Role |
|------|------|------|------|
| `accepted_2007_to_2018Q4.csv.gz` | ~2.26M *(verify)* | 151 *(verify)* | Funded loans with outcomes. The modelling table. |
| `rejected_2007_to_2018Q4.csv.gz` | ~27.6M *(verify)* | 9 *(verify)* | Declined applications. Used only for the selection-bias analysis. |

The rejected file shares almost no columns with the accepted file (it has amount requested, a risk
score, DTI, ZIP, state, employment length, policy code and date). It therefore **cannot** be used to
train the PD model; it is used to characterise how the approved population differs from the applicant
population.

---

## 4. Target variable

`loan_status` is the raw outcome field. It is mapped as:

| Raw `loan_status` | Terminal? | `default` |
|-------------------|-----------|-----------|
| `Fully Paid` | yes | 0 |
| `Does not meet the credit policy. Status:Fully Paid` | yes | 0 |
| `Charged Off` | yes | 1 |
| `Default` | yes | 1 |
| `Does not meet the credit policy. Status:Charged Off` | yes | 1 |
| `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` | no | dropped (see below) |

Because the modelling population is restricted to **36-month loans issued 2010-01 through 2015-12**,
every modelled loan has had at least 36 months to resolve before the 2018-12 data cutoff. Non-terminal
statuses inside that window are data anomalies rather than censoring; they are dropped and the count
is reported in `results/reports/data_profile.md`. If that count is material (>0.5%), the assumption
is revisited rather than waved through.

Expected default rate in the modelled window: **~15%** *(verify)*. This is a workable imbalance —
strong enough to matter, mild enough that aggressive resampling is likely to do more harm to
calibration than good to ranking. That is tested in EXP01, not assumed.

---

## 5. Feature groups (application-time only)

Only fields knowable **at the moment of the credit decision** are eligible as features. The full
column-by-column ruling is in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md); the groups are:

| Group | Examples | Why it should predict default |
|-------|----------|-------------------------------|
| Loan request | `loan_amnt`, `term`, `purpose`, `installment` | Size and purpose of borrowing; affordability |
| Pricing (see caveat) | `int_rate`, `grade`, `sub_grade` | LendingClub's own risk assessment |
| Income and employment | `annual_inc`, `emp_length`, `emp_title`, `verification_status`, `home_ownership` | Capacity and stability of repayment |
| Affordability ratios | `dti`, derived loan-to-income, instalment-to-income | Debt burden relative to income |
| Bureau — depth | `earliest_cr_line`, `total_acc`, `open_acc`, `mo_sin_old_rev_tl_op` | Length and breadth of credit history |
| Bureau — delinquency | `delinq_2yrs`, `mths_since_last_delinq`, `pub_rec`, `pub_rec_bankruptcies` | Demonstrated past repayment failure |
| Bureau — utilisation | `revol_bal`, `revol_util`, `bc_util`, `total_bal_ex_mort` | Current stress on existing credit lines |
| Bureau — recent activity | `inq_last_6mths`, `acc_open_past_24mths`, `mths_since_recent_inq` | Credit-seeking intensity, a classic early warning |
| Scorecard | `fico_range_low`, `fico_range_high` | External bureau score at application |
| Geography | `addr_state`, `zip_code` (3-digit) | Local economic conditions; also the primary graph relation |

### Caveat on `grade` / `sub_grade` / `int_rate`

These are **LendingClub's own risk grade and the price derived from it**. They are legitimately
available at decision time, but a model built on them is partly learning to reproduce an existing
underwriting model rather than predicting default from primary evidence. This project therefore
reports **two feature sets** throughout:

- `primary` — excludes `grade`, `sub_grade`, `int_rate`. The honest "can we underwrite?" question.
- `with_lc_grade` — includes them. The "can we beat the incumbent's own score?" question.

`int_rate` is always available to the **economics** (it determines revenue) even when excluded from
the **PD features**. Keeping those two roles separate is deliberate.

---

## 6. Temporal fields

| Field | Meaning | Use |
|-------|---------|-----|
| `issue_d` | Month loan was funded | **The decision timestamp.** Drives splits, graph edge direction, and all as-of logic |
| `earliest_cr_line` | First credit line opened | Feature: credit history length at `issue_d` |
| `last_pymnt_d` | Month of last payment received | **Outcome-side only.** Used to estimate when a default became known. Never a feature |
| `last_credit_pull_d` | LendingClub's last bureau pull | **Post-origination.** Never a feature |
| `next_pymnt_d`, `last_fico_range_*` | Post-origination | Never a feature |

---

## 7. Relational keys

LendingClub scrubbed `member_id` (it is entirely null in the public files), so **there is no borrower
identity across loans and no observed counterparty network**. The graph is therefore built from
shared attributes:

| Relation | Key | Cardinality | Rationale |
|----------|-----|-------------|-----------|
| R1 geography | `zip3` = first 3 chars of `zip_code` | ~900 *(verify)* | Shared local labour and housing market |
| R2 employment | normalised `emp_title` | ~300 buckets after normalisation | Shared occupational or employer exposure |

**Known defect in R2, stated up front:** LendingClub's data dictionary notes that *employer title
replaced employer name for all loans listed after 2013-09-23*. So `emp_title` means **employer name**
for roughly the first half of the modelling window and **job title** for the second. R2 is therefore
kept configurable, ablated separately, and never presented as a clean occupation relation. See
[`GRAPH_DESIGN.md`](GRAPH_DESIGN.md).

---

## 8. Missingness

Measured by `scripts/prepare_data.py` into `results/reports/data_profile.md`. Known structural
patterns to expect *(verify)*:

- `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_recent_bc_dlq` — missing means
  *"no such event on file"*, which is **informative**, not absent. Encoded with an explicit indicator
  plus a sentinel, never mean-imputed.
- The `*_il_*`, `*_bc_*`, `mo_sin_*`, `num_*` bureau block was introduced by LendingClub around
  2012-2015 and is **systematically missing in earlier vintages**. This is a genuine trap: a naive
  model can learn "this block is present" as a proxy for vintage. Handled explicitly in
  [`LEAKAGE.md`](LEAKAGE.md) §5.
- `emp_title`, `emp_length` — missing for unemployed, retired and non-disclosing applicants.
- `revol_util`, `dti` — occasional nulls and, for `dti`, sentinel values such as -1 and extreme
  outliers requiring clipping.

---

## 9. Known limitations

Stated plainly because an interviewer will ask.

1. **Approved-only population.** Every accepted-file row passed LendingClub's underwriting. The PD
   model is trained on survivors, so it estimates *P(default | approved)*, not *P(default | applied)*.
   Applying it to the full applicant population is extrapolation. Quantified against the rejected
   file; reject inference is discussed but **not** implemented as a fix, because doing it properly
   needs propensity information that is not published.
2. **Marketplace, not a bank.** LendingClub's economics (investor-funded notes, servicing fee) are
   not a bank's. The cost model in [`ASSUMPTIONS.md`](ASSUMPTIONS.md) is a simplification and is
   labelled as such.
3. **Counterfactual exposure.** We observe the outcome only at the amount actually funded. Reasoning
   about a different loan size requires a modelling assumption, documented in
   [`RL_FORMULATION.md`](RL_FORMULATION.md) §4.
4. **Cohort graph, not a transaction network.** R1/R2 are shared-attribute edges. This tests whether
   *correlated-risk* structure helps; it does not test whether a true counterparty network would.
5. **One institution, one credit cycle.** 2010-2015 US unsecured consumer credit, in a long
   expansion. Nothing here has been tested through a downturn.
6. **Mirror provenance.** The file is a third-party mirror; LendingClub no longer publishes the
   originals. Checksums are recorded at download so results stay reproducible against a fixed copy.

---

## 10. Potential leakage in this dataset

Summarised here, treated fully in [`LEAKAGE.md`](LEAKAGE.md). LendingClub is a notoriously leaky
dataset because roughly a third of its columns are recorded *after* origination. The high-risk set:

`total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`,
`recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`,
`out_prncp`, `out_prncp_inv`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`,
`debt_settlement_flag`, `settlement_*`, `hardship_*`, `pymnt_plan`, `loan_status`.

These are **outcome fields**. The project uses them to build the target and the realised cashflows,
and the feature builder maintains a hard deny-list so they can never reach a model. A test asserts it.

---

## 11. Split strategy

```
graph history only   issue_d in [2007-06, 2009-12]   not modelled, provides past neighbours
train                issue_d in [2010-01, 2014-06]   fit everything here
valid                issue_d in [2014-07, 2014-12]   early stopping, calibration, threshold choice
test  (out-of-time)  issue_d in [2015-01, 2015-12]   evaluated once
monitoring traffic   issue_d in [2016-01, 2018-12]   features and predictions only; outcomes immature
```

Rationale:

- **Out-of-time, not random.** Credit performance is driven by vintage and macro conditions. A random
  split lets a model see the future of the same cycle and inflates every metric.
- **Calibration is fitted on `valid`, never on `train`.** Fitting a calibrator on the data the model
  was trained on gives an over-confident, useless calibration map.
- **All encoders, imputers, bucket vocabularies and cohort statistics are fitted on `train` only** and
  applied unchanged downstream.
- **The monitoring window has no mature labels — on purpose.** That is exactly the production
  situation: features and predictions arrive immediately, outcomes arrive years later. Drift on
  features and predictions is measurable there; calibration drift is measured on labelled vintages.
