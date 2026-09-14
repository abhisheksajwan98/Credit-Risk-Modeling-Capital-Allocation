> Computed on **real** data.

# cohort_graph: graph construction

Nodes: **1,609,607**. Edges: **26,060,254** across 2 relations.

## Relations

| relation   | key_column   |   cohorts |    edges |   mean_degree |   median_degree |   isolated_share |   at_cap_share |
|:-----------|:-------------|----------:|---------:|--------------:|----------------:|-----------------:|---------------:|
| zip3       | zip3         |       939 | 16037403 |        9.9636 |              10 |           0.0008 |         0.9936 |
| emp        | emp_norm     |    308446 | 10022851 |        6.2269 |              10 |           0.2806 |         0.5742 |

## The two information sets

A neighbour's *application* is visible as soon as it is filed. A neighbour's *outcome*
is not visible until the loan resolves, which for a 36-month loan is up to three years
later. Those are different information sets, and the gap between them is large:

| relation   |   mean_prior_applications |   mean_resolved_outcomes |   share_of_predecessors_with_known_outcome |
|:-----------|--------------------------:|-------------------------:|-------------------------------------------:|
| zip3       |                   1951.66 |                 1014.27  |                                     0.5197 |
| emp        |                   1433.48 |                  568.433 |                                     0.3965 |

Feature aggregates use the first column; label aggregates use the second. Using the
first for a label aggregate is the standard way graph credit models leak, and it would
inflate AUC substantially, because the cohort default rate would then contain outcomes
from the same period the model is being scored on. See docs/GRAPH_DESIGN.md.

## Leakage checks

- every edge satisfies `issue_period[neighbour] < issue_period[node]`: **PASSED**
- no self-loops: **PASSED**
- no loan resolves on or before its own issue month: **PASSED**
