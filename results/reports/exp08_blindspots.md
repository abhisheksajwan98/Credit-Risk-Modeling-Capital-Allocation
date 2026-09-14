> Computed on **real** data.

# EXP08: tabular anomaly as a blind-spot signal

An autoencoder (32 numeric features, latent dim 16) was fitted on the training window and never saw the target. Reconstruction error scores the 283,026 out-of-time test borrowers.

## Marginal view (descriptive only)

| quantile     |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   observed_expected_ratio |
|:-------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|--------------------------:|
| Q1_normal    | 56606 |        0.0117 |         0.1376 |    0.1351 |            0.0025 |    0.6472 |  0.1148 | 0.0049 |                    0.9816 |
| Q2           | 56605 |        0.024  |         0.1433 |    0.1366 |            0.0067 |    0.6681 |  0.1175 | 0.0085 |                    0.9532 |
| Q3           | 56605 |        0.0383 |         0.1445 |    0.1359 |            0.0086 |    0.6731 |  0.1181 | 0.0112 |                    0.9402 |
| Q4           | 56605 |        0.0629 |         0.1498 |    0.1376 |            0.0122 |    0.6864 |  0.1205 | 0.0133 |                    0.9185 |
| Q5_anomalous | 56605 |        0.2401 |         0.1691 |    0.1462 |            0.0229 |    0.6706 |  0.1339 | 0.0229 |                    0.8648 |

Read this table with care. If the anomaly score correlates with the PD level, calibration will appear to degrade across quintiles simply because a boosted model is less well calibrated at high PD. That is not a blind spot; it is a known property.

## Conditional view (the evidence)

Within each PD decile the population is split at the median anomaly score, and the calibration gap (observed minus predicted default rate) is compared. A positive `gap_difference` means the model under-states risk more for anomalous borrowers *at the same predicted PD*.

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0392 |    14151 |   14152 |           0.0369 |          0.0371 |     0.0002 |          0.0414 |         0.036  |   -0.0054 |           0.0055 |
|            1 |    0.0646 |    14151 |   14152 |           0.0645 |          0.0711 |     0.0066 |          0.0647 |         0.0631 |   -0.0016 |           0.0082 |
|            2 |    0.0825 |    14151 |   14151 |           0.0825 |          0.0944 |     0.0119 |          0.0826 |         0.0866 |    0.004  |           0.0079 |
|            3 |    0.0994 |    14151 |   14152 |           0.0994 |          0.1182 |     0.0188 |          0.0994 |         0.1025 |    0.003  |           0.0158 |
|            4 |    0.1167 |    14151 |   14151 |           0.1167 |          0.1394 |     0.0228 |          0.1166 |         0.1214 |    0.0048 |           0.018  |
|            5 |    0.1355 |    14151 |   14152 |           0.1355 |          0.1562 |     0.0207 |          0.1356 |         0.1487 |    0.0131 |           0.0076 |
|            6 |    0.1567 |    14151 |   14151 |           0.1569 |          0.1838 |     0.027  |          0.1566 |         0.1633 |    0.0067 |           0.0202 |
|            7 |    0.1825 |    14151 |   14152 |           0.1826 |          0.2061 |     0.0235 |          0.1825 |         0.1933 |    0.0108 |           0.0127 |
|            8 |    0.2174 |    14151 |   14151 |           0.2176 |          0.2324 |     0.0148 |          0.2172 |         0.2248 |    0.0076 |           0.0072 |
|            9 |    0.2882 |    14151 |   14152 |           0.2939 |          0.3155 |     0.0215 |          0.2825 |         0.2835 |    0.001  |           0.0206 |

## Verdict

Anomaly score: correlation with PD itself is +0.022. Within PD strata the high-signal half shows a larger gap in 10/10 strata (mean +0.0124, sign-test p=0.0010), so the effect **survives** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.032.

**Decision: KEEP.** The signal carries information about model reliability that the PD does not. Note that the effect on the outcome itself is modest -- the value is in flagging *miscalibration*, not in improving discrimination.

**Scope limit.** Only numeric features are encoded; the 8 categorical columns are not. A borrower unusual only in loan purpose or state would not be flagged.
