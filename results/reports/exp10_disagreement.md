> Computed on **real** data.

# EXP10: model disagreement as a blind-spot signal

Two LightGBM models were fitted under an **identical** configuration and calibrator, differing only in whether the 18 graph cohort features were available. Disagreement is `|logit(p_tabular) - logit(p_graph)|` on the 283,026 out-of-time test borrowers.

Measured on the **logodds** scale, disagreement correlates with the PD level at Spearman -0.150. On the raw probability scale the same quantity correlates at +0.258 -- which is why the log-odds scale is used: an absolute probability difference is mechanically larger in the middle of the PD range and so partly restates the PD rather than measuring disagreement.

## Marginal view (descriptive only)

| quantile    |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   observed_expected_ratio |
|:------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|--------------------------:|
| Q1_agree    | 57639 |        0.0201 |         0.156  |    0.1462 |            0.0098 |    0.6683 |  0.1255 | 0.0105 |                    0.9372 |
| Q2          | 58522 |        0.0639 |         0.1561 |    0.148  |            0.0081 |    0.6787 |  0.125  | 0.0091 |                    0.9484 |
| Q3          | 53794 |        0.1274 |         0.1546 |    0.1454 |            0.0092 |    0.6568 |  0.1255 | 0.0128 |                    0.9404 |
| Q4          | 56815 |        0.2232 |         0.143  |    0.1346 |            0.0084 |    0.6641 |  0.1174 | 0.009  |                    0.9416 |
| Q5_disagree | 56256 |        0.4258 |         0.1345 |    0.1169 |            0.0176 |    0.6771 |  0.1113 | 0.0179 |                    0.869  |

## Conditional view (the evidence)

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0392 |    13676 |   14627 |           0.0401 |          0.0396 |    -0.0004 |          0.0383 |         0.0337 |   -0.0046 |           0.0042 |
|            1 |    0.0646 |    11902 |   16401 |           0.064  |          0.0655 |     0.0015 |          0.065  |         0.0683 |    0.0033 |          -0.0019 |
|            2 |    0.0825 |    13546 |   14756 |           0.0822 |          0.0916 |     0.0095 |          0.0829 |         0.0895 |    0.0066 |           0.0029 |
|            3 |    0.0994 |    13986 |   14317 |           0.0993 |          0.1113 |     0.012  |          0.0995 |         0.1095 |    0.0099 |           0.002  |
|            4 |    0.1167 |    14147 |   14155 |           0.1168 |          0.1337 |     0.0169 |          0.1165 |         0.1271 |    0.0106 |           0.0063 |
|            5 |    0.1355 |    13634 |   14669 |           0.1351 |          0.1537 |     0.0186 |          0.1359 |         0.1512 |    0.0153 |           0.0033 |
|            6 |    0.1567 |    13618 |   14684 |           0.1566 |          0.177  |     0.0204 |          0.1569 |         0.1704 |    0.0135 |           0.0069 |
|            7 |    0.1825 |    14092 |   14211 |           0.1828 |          0.205  |     0.0222 |          0.1822 |         0.1944 |    0.0121 |           0.0101 |
|            8 |    0.2174 |    13896 |   14406 |           0.2172 |          0.2296 |     0.0124 |          0.2176 |         0.2277 |    0.0101 |           0.0023 |
|            9 |    0.2882 |    14001 |   14302 |           0.2881 |          0.3027 |     0.0146 |          0.2883 |         0.2963 |    0.008  |           0.0065 |

## Verdict

Disagreement (logodds scale): correlation with PD itself is -0.150. Within PD strata the high-signal half shows a larger gap in 9/10 strata (mean +0.0043, sign-test p=0.0107), so the effect **survives** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.043.

**Decision: KEEP.**

> **Correction.** The first version of this experiment compared the production PD against an alternative model that differed in hyper-parameters *and* calibrator *and* features, using an absolute probability difference. The disagreement it measured was therefore partly a measure of those choices and partly a restatement of the PD level. See `docs/AUDIT_REPORT.md`.
