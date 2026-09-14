> Computed on **real** data.

# EXP11: does acting on blind-spot signals improve the book?

Applicant pool: 283,026 out-of-time loans (2015 vintage). Capital budget $1,922,008,880, 80% of profitable demand.
Flagged as blind spots: 53,666 borrowers (19.0%) above the 90th percentile of the anomaly **or** disagreement signal.

## Portfolios at equal budget, measured on observed cashflows

| policy            |   n_loans |   capital_deployed |   observed_profit |   return_on_capital |   default_rate |   mean_apr |   budget_utilisation |
|:------------------|----------:|-------------------:|------------------:|--------------------:|---------------:|-----------:|---------------------:|
| A_pd_ranked       |    131570 |        1.92199e+09 |       1.56971e+08 |              0.0817 |         0.173  |     0.1327 |               1      |
| B_blindspot_aware |    133378 |        1.8906e+09  |       1.53764e+08 |              0.0813 |         0.1632 |     0.1266 |               0.9837 |
| C_exposure_capped |    148186 |        1.922e+09   |       1.56622e+08 |              0.0815 |         0.1679 |     0.1293 |               1      |

## Verdict

B_blindspot_aware changed realised profit by **$-3,207,014** on a $1,921,989,925 book, deploying $-31,393,100 of capital (a -0.03% change in return on capital). That is **below 50bp of return on capital and should be read as no material difference in profitability**. The book default rate moved from 17.30% to 16.32% at essentially unchanged return, so the defensible claim is risk reduction at constant profitability -- not a profit improvement.

The swap involved 26,541 loans out (realised return +7.69%, default 17.23%) and 28,349 loans in (realised return +7.48%, default 12.62%).

Policy C, which caps flagged borrowers at 50% exposure rather than refusing them, changed return on capital by -0.02% against Policy A.

## What this does and does not show

- Profit here is **observed**: `total_pymnt + recoveries - collection_recovery_fee - funded_amnt` for every matured loan. No loss-given-default is assumed.
- The comparison is between **whole portfolios at the same budget**, not between the two swap sets. Swap sets need not deploy equal capital, so differencing their profits is not a portfolio result.
- **Return on capital is the comparable quantity.** Profit alone rewards whichever book happens to deploy more capital.
- This is a *retrospective reallocation* on an already-approved population. It does not establish what would happen to borrowers LendingClub declined, and it is not evidence that the policy is safe to deploy.

> **Correction.** An earlier version of this experiment computed profit as `amount x apr x 1.55 - amount x 0.5 x default` and compared the two swap sets rather than the portfolios. That formula understated realised book profit by roughly a factor of 2.4, and it reported a **$9.8M loss** as a *'net increase in profit'* -- the conclusion was asserted regardless of the sign of the number printed above it. Measured on observed cashflows at equal budget, the effect is a fraction of a percentage point of return on capital in either direction, i.e. not a profit result at all. See `docs/AUDIT_REPORT.md`.
