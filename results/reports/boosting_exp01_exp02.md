> Computed on **real** data.

# boosting: EXP01 and EXP02

Feature set: **primary** (40 features, 42 dropped by the availability guard and constancy checks).
Out-of-time test window: 283,026 loans, observed default rate 0.1489.

## EXP01 - does non-linear modelling improve default prediction?

|          |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| lightgbm | 283026 |         0.14886 |           0.12973 |   0.6705  |  0.25019 | 0.34101 | 0.24634 | 0.12124 |    0.39864 | 0.01931 | 0.02822 |             0.99098 |                 0.15153 |                   0.87147 |
| logistic | 283026 |         0.14886 |           0.12857 |   0.65806 |  0.23663 | 0.31613 | 0.22797 | 0.1223  |    0.40263 | 0.02044 | 0.03359 |             0.97228 |                 0.12721 |                   0.86368 |

**Verdict.** LightGBM moves out-of-time ROC-AUC by +0.0124 against logistic regression (0.6581 -> 0.6705). That is a material margin and justifies the non-linear model.

## EXP02 - does calibration improve the reliability of PD estimates?

| method   |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| identity | 283026 |         0.14886 |           0.12973 |   0.6705  |  0.25019 | 0.24634 | 0.12124 |    0.39864 | 0.01931 | 0.02822 |             0.99098 |                 0.15153 |                   0.87147 |
| platt    | 283026 |         0.14886 |           0.13827 |   0.6705  |  0.25019 | 0.24634 | 0.12095 |    0.39749 | 0.01111 | 0.0181  |             1.00487 |                 0.09863 |                   0.92885 |
| isotonic | 283026 |         0.14886 |           0.13829 |   0.67023 |  0.24563 | 0.24496 | 0.12098 |    0.39759 | 0.01057 | 0.03183 |             0.98252 |                 0.06048 |                   0.929   |

**Verdict.** platt reduced Brier by 0.24% (0.12124 -> 0.12095) and moved ROC-AUC by +0.00000; ECE 0.0193 -> 0.0111, calibration slope 0.991 -> 1.005

Note the ROC-AUC column: it barely moves. That is the whole point. Calibration is a monotone transform of the score, so every ranking metric is blind to it, while the probabilities the decision layer multiplies by an exposure become materially more accurate. A model that ranks well but is mis-calibrated will systematically mis-state expected loss.
