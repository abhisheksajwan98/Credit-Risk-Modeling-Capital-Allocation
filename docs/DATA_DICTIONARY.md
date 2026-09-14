# LendingClub Data Dictionary (Leakage Control)

The following lists the column-by-column decision-time rulings for the ~151 LendingClub columns. It explicitly defines what is a valid feature (knowable at the time of decision) and what is post-origination (knowable only after money has changed hands). 

This is not just documentation; it is the contract enforced by `tests/test_leakage.py` and the `schema.py` module.

## 1. Post-Origination (The Deny-List)
These columns must **never** appear in a feature matrix. They are strictly available only to the outcome and economics code.

**Repayment Performance:**
- `out_prncp`, `out_prncp_inv`
- `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`
- `recoveries`, `collection_recovery_fee`
- `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`

**Post-Origination Bureau Refresh:**
- `last_credit_pull_d`
- `last_fico_range_high`, `last_fico_range_low` *(A massive leak: a FICO pulled after origination collapses for borrowers who are about to default).*

**Distress / Workout Programmes:**
- `pymnt_plan`, `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `deferral_term`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `payment_plan_start_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `orig_projected_additional_accrued_interest`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount`
- `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term`

**Target:**
- `loan_status` (The target definition source)

## 2. Exposure & Pricing (Economics Only)
These columns represent the decision taken by the lender (LendingClub) rather than the borrower's raw state.
- `funded_amnt`, `funded_amnt_inv`: The exposure actually taken. (Feature models use `loan_amnt`, the amount requested).
- `grade`, `sub_grade`, `int_rate`: LendingClub's underwriting output. Excluded from the `primary` feature set, as training on them is just reverse-engineering an existing scorecard. Included only in the `with_lc_grade` benchmark.

## 3. Valid Decision-Time Features (The Allow-List)

**Core Numeric:**
- `loan_amnt`, `installment` *(Note: installment can leak the interest rate, but it is explicitly excluded from the `primary` set for this reason).*
- `annual_inc`, `dti`
- `fico_range_low`, `fico_range_high` (Decision-time FICO, unlike `last_fico_*`)
- `open_acc`, `total_acc`
- Delinquency & Derogatory: `delinq_2yrs`, `delinq_amnt`, `mths_since_last_delinq`, `mths_since_last_record`, `mths_since_last_major_derog`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `acc_now_delinq`, `chargeoff_within_12_mths`, `collections_12_mths_ex_med`, `tot_coll_amt`
- Utilisation: `revol_bal`, `revol_util`, `tot_cur_bal`, `total_rev_hi_lim`
- `inq_last_6mths`

**Categorical:**
- `term`, `emp_length`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `initial_list_status`, `application_type`

**Vintage Sensitive (Bureau Blocks):**
LendingClub added a bureau block mid-history. Models can learn "this block is populated" as a proxy for the origination date. 
- *Included but monitored for availability drift*: `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mort_acc`, `mths_since_recent_bc`, `mths_since_recent_bc_dlq`, `mths_since_recent_inq`, `mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`

**Late Additions & Joint Applications (Excluded by Default):**
- Columns introduced around ~2015-12 are effectively null during our modelling window (2010-2015).
- Examples: `open_acc_6m`, `open_act_il`, `open_il_12m`, `total_bal_il`, `il_util`, `inq_fi`, etc. Joint applications (`*_joint`, `sec_app_*`) are similarly excluded.

## 4. Derived & Dropped

- **Graph/Relational Sources**: `zip_code` and `emp_title` are consumed to build edges, then dropped from the tabular matrix.
- **Dropped**: Free text redundant with `purpose` or discontinued fields like `desc`, `title`, `policy_code`, `disbursement_method`.

## 5. Summary Rule
If it happened *after* the loan was funded, it is a leak. `schema.py` defines these strict bounds.
