> **Computed on SYNTHETIC data.** These numbers exercise the pipeline and say nothing about consumer credit. Re-run against the real extract before quoting anything here.

# Home Credit: EXP01 and EXP02 (secondary dataset)

> **Splits here are random, not out-of-time.** Home Credit contains no absolute calendar date -- every temporal field is expressed in days relative to each application -- so a temporal split is impossible. These metrics are therefore optimistic relative to deployment and are **not comparable** with the LendingClub results, which use a strict out-of-time test window. This is the principal reason the dataset was not selected as the primary one; see docs/DATASET.md.

Rows: 25,000. Default rate: 0.1086. Features: 13 numeric, 7 categorical.

## EXP01

|          |    n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|-----:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| logistic | 5009 |          0.1084 |           0.10719 |   0.69966 |  0.23265 | 0.39932 | 0.31219 | 0.09107 |    0.31725 | 0.01611 | 0.0487  |             0.96575 |                -0.05213 |                   0.98879 |
| lightgbm | 5009 |          0.1084 |           0.10742 |   0.69768 |  0.22567 | 0.39536 | 0.30831 | 0.09257 |    0.32391 | 0.03258 | 0.10207 |             1.78021 |                 1.58526 |                   0.99089 |

## EXP02

| method   |    n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|-----:|----------------:|------------------:|----------:|---------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| identity | 5009 |          0.1084 |           0.10742 |   0.69768 |  0.22567 | 0.30831 | 0.09257 |    0.32391 | 0.03258 | 0.10207 |             1.78021 |                 1.58526 |                   0.99089 |
| platt    | 5009 |          0.1084 |           0.11313 |   0.69768 |  0.22567 | 0.30831 | 0.09158 |    0.31951 | 0.01794 | 0.05244 |             1.07084 |                 0.08375 |                   1.04361 |
| isotonic | 5009 |          0.1084 |           0.11359 |   0.6903  |  0.20684 | 0.27659 | 0.0918  |    0.31981 | 0.01138 | 0.05296 |             0.96562 |                -0.11961 |                   1.04779 |

**Verdict.** platt reduced Brier by 1.07% (0.09257 -> 0.09158) and moved ROC-AUC by +0.00000; ECE 0.0326 -> 0.0179, calibration slope 1.780 -> 1.071

## Notes from preparation

- DAYS_EMPLOYED == 365243 on 4,546 rows: replaced with NaN plus an explicit indicator. Left as-is it encodes a 1000-year employment history.
- Splits are RANDOM, not out-of-time. Home Credit contains no absolute calendar date, so a temporal split is impossible. Metrics from this adapter are therefore optimistic relative to deployment and are not comparable with the LendingClub results.

## What is deliberately absent

- **No graph (Layer B).** Home Credit's tables are hierarchical: a borrower and *their own* bureau records. There are no borrower-to-borrower links, so any graph would be a k-nearest-neighbour construction in feature space -- a smoothing regulariser, not relational financial information.
- **No decision layer (Layer C).** No interest rate, no recovery amount, no realised cashflow. Every term in an expected-profit calculation would have to be invented.
