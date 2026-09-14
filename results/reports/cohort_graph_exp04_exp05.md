> Computed on **real** data.

# cohort_graph: EXP04 and EXP05

Out-of-time test window: 283,026 loans.

|                      |      n |   positive_rate |   mean_prediction |   roc_auc |   pr_auc |    gini |      ks |   brier |   log_loss |     ece |     mce |   calibration_slope |   calibration_intercept |   observed_expected_ratio |
|:---------------------|-------:|----------------:|------------------:|----------:|---------:|--------:|--------:|--------:|-----------:|--------:|--------:|--------------------:|------------------------:|--------------------------:|
| A_tabular            | 283026 |         0.14886 |           0.13841 |   0.66897 |  0.24403 | 0.33794 | 0.24387 | 0.12108 |    0.39796 | 0.01046 | 0.02656 |             0.97142 |                 0.0407  |                   0.92976 |
| B_tabular_plus_graph | 283026 |         0.14886 |           0.14867 |   0.66793 |  0.24223 | 0.33587 | 0.24109 | 0.12105 |    0.39764 | 0.00324 | 0.01127 |             0.97617 |                -0.03603 |                   0.9987  |
| C_graphsage          | 283026 |         0.14886 |           0.13698 |   0.66158 |  0.23501 | 0.32316 | 0.23418 | 0.1217  |    0.40047 | 0.01188 | 0.02457 |             0.95621 |                 0.02607 |                   0.92016 |

## EXP04 - do graph-derived cohort features add signal beyond tabular features?

Arm A (tabular) ROC-AUC 0.6690; arm B (tabular + cohort aggregates) 0.6679; difference **-0.0010**.

The 18 cohort features account for 10.7% of total LightGBM gain in arm B.
**Verdict.** Relational information adds little or nothing beyond the borrower's own file on this dataset. That is a legitimate result, and the more likely one for a shared-attribute cohort graph: geography and employment are already partly encoded in the tabular features (`addr_state`, `emp_length`), so the cohort aggregates are largely redundant with them.

## EXP05 - does GraphSAGE add value beyond the aggregates?

Arm C (GraphSAGE) ROC-AUC 0.6616, against arm B at 0.6679 (**-0.0064**) and arm A at 0.6690 (**-0.0074**).
Trained on `cuda`, best epoch 3, 125 node features.

**Verdict.** The GNN does **not** beat the hand-built cohort aggregates. This is the expected outcome and worth stating plainly: mean-aggregation over a one-hop cohort is close to what the aggregate features already compute in closed form, so the GNN is re-learning a known function from far less signal, with more parameters. On a genuine transaction network -- where multi-hop structure carries information no aggregate captures -- the comparison could well go the other way. This dataset has no such network.

Practical note: arm B is also the more deployable of the two. Its contribution is legible in a SHAP plot and can be given to an applicant as a reason; the GNN's neighbourhood aggregation cannot, which matters under adverse-action requirements.
