> Computed on **real** data.

# EXP09: relational uncertainty as a blind-spot signal

A VGAE (2 GraphSAGE layers, latent dim 16) was trained to reconstruct the cohort graph, stopping early on held-out link reconstruction after 19 epochs (best validation loss 0.3555). Per-node posterior sigma scores the 283,026 out-of-time test borrowers.

## Marginal view (descriptive only)

| quantile     |     n |   mean_signal |   default_rate |   mean_pd |   calibration_gap |   roc_auc |   brier |    ece |   observed_expected_ratio |
|:-------------|------:|--------------:|---------------:|----------:|------------------:|----------:|--------:|-------:|--------------------------:|
| Q1_certain   | 56606 |        0.0241 |         0.1494 |    0.1366 |            0.0127 |    0.6687 |  0.1215 | 0.0135 |                    0.9147 |
| Q2           | 56605 |        0.038  |         0.1492 |    0.1391 |            0.0101 |    0.6681 |  0.1213 | 0.0104 |                    0.9325 |
| Q3           | 56605 |        0.048  |         0.1478 |    0.1389 |            0.0089 |    0.6673 |  0.1204 | 0.01   |                    0.9399 |
| Q4           | 56605 |        0.0596 |         0.1493 |    0.1389 |            0.0104 |    0.6719 |  0.1211 | 0.012  |                    0.9304 |
| Q5_uncertain | 56605 |        0.085  |         0.1487 |    0.1378 |            0.0109 |    0.6765 |  0.1205 | 0.0122 |                    0.9268 |

## Conditional view (the evidence)

|   pd_stratum |   mean_pd |   n_high |   n_low |   predicted_high |   observed_high |   gap_high |   predicted_low |   observed_low |   gap_low |   gap_difference |
|-------------:|----------:|---------:|--------:|-----------------:|----------------:|-----------:|----------------:|---------------:|----------:|-----------------:|
|            0 |    0.0392 |    14151 |   14152 |           0.0389 |          0.0345 |    -0.0044 |          0.0394 |         0.0387 |   -0.0007 |          -0.0037 |
|            1 |    0.0646 |    14151 |   14152 |           0.0645 |          0.0655 |     0.001  |          0.0646 |         0.0687 |    0.0041 |          -0.0031 |
|            2 |    0.0825 |    14151 |   14151 |           0.0825 |          0.0906 |     0.0081 |          0.0826 |         0.0904 |    0.0078 |           0.0003 |
|            3 |    0.0994 |    14151 |   14152 |           0.0994 |          0.1119 |     0.0126 |          0.0994 |         0.1087 |    0.0093 |           0.0033 |
|            4 |    0.1167 |    14151 |   14151 |           0.1167 |          0.1328 |     0.0161 |          0.1167 |         0.128  |    0.0114 |           0.0047 |
|            5 |    0.1355 |    14151 |   14152 |           0.1355 |          0.1522 |     0.0167 |          0.1355 |         0.1526 |    0.0171 |          -0.0004 |
|            6 |    0.1567 |    14151 |   14151 |           0.1567 |          0.1705 |     0.0139 |          0.1568 |         0.1766 |    0.0198 |          -0.006  |
|            7 |    0.1825 |    14151 |   14152 |           0.1826 |          0.2041 |     0.0215 |          0.1825 |         0.1952 |    0.0127 |           0.0088 |
|            8 |    0.2174 |    14151 |   14151 |           0.2175 |          0.2292 |     0.0117 |          0.2173 |         0.228  |    0.0107 |           0.0009 |
|            9 |    0.2882 |    14151 |   14152 |           0.289  |          0.2984 |     0.0094 |          0.2874 |         0.3005 |    0.0131 |          -0.0037 |

## Verdict

Graph uncertainty (VGAE sigma): correlation with PD itself is -0.001. Within PD strata the high-signal half shows a larger gap in 5/10 strata (mean +0.0001, sign-test p=0.6230), so the effect **does not survive** the PD control -- the model under-states risk for high-signal borrowers at a given PD. Holding PD fixed, a one-SD increase in the signal multiplies default odds by 1.002.

**Decision: REJECT.** Relational uncertainty does **not** predict where the PD model fails. This is a clean negative: the model was trained to convergence with early stopping, so the null is a property of the signal rather than of the training budget. Reporting it is the point -- a complex model that adds nothing should be dropped, and the tabular anomaly signal in EXP08 is doing different work.

Sigma correlates with node degree at Spearman +0.006. It is not simply a proxy for neighbourhood size.

> **Correction.** The first version of this experiment trained for 10 epochs with no validation split and no early stopping, which made 'no signal' and 'undertrained' indistinguishable. See `docs/AUDIT_REPORT.md`.
