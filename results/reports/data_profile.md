> Computed on **real** data.

# Data profile

Rows read from raw file: **2,260,701**
After the 36-month term filter: **1,609,754**
After the issue-window filter: **1,609,754**
Final modelling table: **1,609,607** rows x 162 columns

## Rows dropped

- malformed / non-numeric id: 33
- non-terminal status inside a matured window: 147
- unmapped loan_status: 0

## Splits

| split | window | rows | default rate |
|---|---|---|---|
| history | 2007-06 .. 2009-12 | 8,277 | n/a (unlabelled by design) |
| train | 2010-01 .. 2014-06 | 239,104 | 0.1267 |
| valid | 2014-07 .. 2014-12 | 90,615 | 0.1413 |
| test | 2015-01 .. 2015-12 | 283,026 | 0.1489 |
| monitor | 2016-01 .. 2018-12 | 988,585 | n/a (unlabelled by design) |

## Default rate by vintage

| year | default rate |
|---|---|
| 2010 | 0.1092 |
| 2011 | 0.1063 |
| 2012 | 0.1358 |
| 2013 | 0.1233 |
| 2014 | 0.1373 |
| 2015 | 0.1489 |

## Highest missingness

| column | missing |
|---|---|
| `member_id` | 100.0% |
| `orig_projected_additional_accrued_interest` | 99.7% |
| `hardship_type` | 99.6% |
| `hardship_reason` | 99.6% |
| `hardship_status` | 99.6% |
| `deferral_term` | 99.6% |
| `hardship_amount` | 99.6% |
| `hardship_start_date` | 99.6% |
| `hardship_end_date` | 99.6% |
| `payment_plan_start_date` | 99.6% |
| `hardship_length` | 99.6% |
| `hardship_dpd` | 99.6% |
| `hardship_loan_status` | 99.6% |
| `hardship_payoff_balance_amount` | 99.6% |
| `hardship_last_payment_amount` | 99.6% |
| `debt_settlement_flag_date` | 98.7% |
| `settlement_status` | 98.7% |
| `settlement_date` | 98.7% |
| `settlement_amount` | 98.7% |
| `settlement_percentage` | 98.7% |
| `settlement_term` | 98.7% |
| `sec_app_mths_since_last_major_derog` | 98.6% |
| `sec_app_revol_util` | 96.2% |
| `revol_bal_joint` | 96.1% |
| `sec_app_fico_range_low` | 96.1% |

## Availability shift (train -> test)

Columns whose *missingness* moves between windows. These partly encode the
origination date, so a model can learn 'is this populated?' as a vintage proxy.
The feature builder's availability guard drops them; see docs/LEAKAGE.md section 5.

| column | shift in missing rate |
|---|---|
| `pct_tl_nvr_dlq` | -19.2% |
| `avg_cur_bal` | -19.2% |
| `mo_sin_old_rev_tl_op` | -19.2% |
| `mo_sin_rcnt_rev_tl_op` | -19.2% |
| `tot_coll_amt` | -19.2% |
| `tot_cur_bal` | -19.2% |
| `total_rev_hi_lim` | -19.2% |
| `mo_sin_rcnt_tl` | -19.2% |
| `num_accts_ever_120_pd` | -19.2% |
| `num_actv_bc_tl` | -19.2% |
| `num_actv_rev_tl` | -19.2% |
| `num_bc_tl` | -19.2% |
| `num_il_tl` | -19.2% |
| `num_op_rev_tl` | -19.2% |
| `num_rev_tl_bal_gt_0` | -19.2% |
| `num_tl_30dpd` | -19.2% |
| `num_tl_90g_dpd_24m` | -19.2% |
| `num_tl_op_past_12m` | -19.2% |
| `tot_hi_cred_lim` | -19.2% |
| `total_il_high_credit_limit` | -19.2% |
| `num_rev_accts` | -19.2% |
| `mo_sin_old_il_acct` | -19.0% |
| `num_tl_120dpd_2m` | -15.8% |
| `num_bc_sats` | -15.2% |
| `num_sats` | -15.2% |

## Notes

- dropped 147 labelled rows (0.02%) with a non-terminal or unmapped status inside a window that should be fully matured

Data source: **real**.
