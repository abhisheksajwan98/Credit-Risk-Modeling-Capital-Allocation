# Graph design

## 1. What the graph is, stated plainly

LendingClub scrubbed `member_id` and publishes no counterparty links. **There is no observed
transaction network in this dataset.** What exists is shared-attribute structure, and the graph
built here is a **cohort graph**: borrowers are connected when they share a cohort key.

This is said up front rather than dressed up, because the honest framing is what makes the
experiment worth running. What a cohort graph *can* test is whether **correlated-risk structure**
carries information beyond the borrower's own file. What it *cannot* test is whether a genuine
counterparty network would help — that would need a different dataset (Freddie Mac's seller and
servicer identifiers, for instance).

### Why correlated risk is a real question

Credit risk is not i.i.d. across borrowers. Two applicants in the same local labour market are
exposed to the same plant closure; two applicants in the same occupation are exposed to the same
sectoral shock. This is not a modelling curiosity — it is the reason every lender carries
geographic and industry concentration limits, and the reason a portfolio of 10,000 loans in one
ZIP code is riskier than 10,000 loans spread across the country at the same average PD.

So "does knowing how this borrower's cohort has recently performed improve the estimate of their
own risk?" is a question a credit officer would recognise.

---

## 2. Node and edge design

**Nodes.** One node per loan application. Since `member_id` is scrubbed, a loan is effectively a
borrower — repeat borrowers cannot be identified and appear as separate nodes (see
[`LEAKAGE.md`](LEAKAGE.md) §7).

**Node features.** Decision-time features only, drawn from the same `FeatureBuilder` output the
tabular models use, one-hot encoded and standardised on train statistics. **No label information
whatsoever** enters the node features, so message passing moves *features* between borrowers, never
outcomes.

**Edges.** Two relations, each a shared attribute:

| Relation | Key | Cardinality | Economic rationale |
|---|---|---|---|
| **R1 `zip3`** | first 3 characters of `zip_code` | ~900 | Shared local labour and housing market. Stable semantics across the whole history. **Primary relation.** |
| **R2 `emp`** | normalised `emp_title` | ~300 buckets | Shared occupational or employer exposure. **Carries a known defect — see below.** |

**No edge features.** Two loans sharing a ZIP prefix share exactly one thing, and there is nothing
to attach to the edge beyond that fact. Inventing an edge weight (feature-space distance, say)
would smuggle a similarity graph in through the back door.

### The `emp_title` defect, stated because it matters

LendingClub's data dictionary records that **employer title replaced employer name for all loans
listed after 2013-09-23**. So `emp_title` means *employer name* for roughly the first half of the
modelling window and *job title* for the second.

The consequences are handled rather than hidden:

- R2 is the **secondary** relation and is configurable, so it can be switched off.
- Its contribution is **ablated separately** rather than blended into a single "graph helps" claim.
- `prepare_data.py` sets `emp_title_is_employer_name` on every row so any analysis can condition
  on the regime.

An honest reading: R2 is a *cohort of people who wrote the same thing in the employment box*. That
is a weaker construct than either a clean employer relation or a clean occupation relation, and
results resting on it deserve less weight than results resting on R1.

---

## 3. Time-respecting construction

Every edge points strictly backwards:

```
edge j -> i  requires  issue_period[j] < issue_period[i]
```

**Strictly**, not "not later than". Same-month loans are not connected: within a month there is no
ordering to appeal to, and a mutual edge between two simultaneous applications would let each see
the other.

Each node keeps at most `max_neighbours` (default 10) of the **most recent** qualifying
predecessors per relation, optionally within a `lookback_months` window (default 36).

- The cap keeps a 20,000-loan ZIP from swamping a 30-loan one, and bounds memory.
- *Most recent* rather than random: freshness is what carries signal about current local conditions.
- The lookback exists because a ZIP's behaviour in 2008 says little about the same ZIP in 2015.

### Why it is implemented with `searchsorted` rather than a join

The naive construction — self-join on the cohort key, filter by date — generates billions of pairs
on the full dataset before any cap is applied. Instead, within each cohort sorted by issue month,
the number of strictly-earlier members is exactly `searchsorted(months, month_i, side="left")`, so
the eligible window for node `i` is the slice ending there. That is O(m log m) per cohort with no
edge table ever materialised, and `side="left"` is precisely what excludes same-month loans.

### Storage

Neighbourhoods are stored densely as `neighbour_index[n_nodes, n_relations, K]` (int32) plus a
matching boolean mask. Every node has a fixed neighbour budget, so a dense layout is natural and
indexes directly in the GNN forward pass — no sampling library, no scatter kernels, no CSR
bookkeeping. At 620k nodes, 2 relations and K=10 this is about 50 MB.

---

## 4. The two information sets

This is the heart of the design, and the thing most graph-credit write-ups get wrong.

| Aggregate | Condition on neighbour `j` | Available when |
|---|---|---|
| Neighbour **features** (mean FICO, mean DTI, cohort activity) | `issue[j] < issue[i]` | The application existed |
| Neighbour **labels** (cohort default rate) | `resolved[j] <= issue[i]` | The outcome had happened *and been booked* |

The second is far stricter. A 36-month loan issued 2013-01 does not resolve until 2016-01, so a
2014 applicant's underwriter cannot know how it ended.

`resolved[j]` is the last payment month, plus five months for charged-off loans (LendingClub books
a charge-off at 150 days delinquent). A loan that has not reached a terminal status has **no**
resolution date, which stops an unresolved loan being counted as a good neighbour merely because it
has not defaulted yet.

**The information gap is large and is reported.** `build_graph.py` writes a table showing mean
predecessors versus mean *resolved* predecessors — typically only around 40% of a borrower's cohort
predecessors have a known outcome at the time they apply. That gap is the cost of doing this
correctly, and it is exactly the cost most implementations avoid paying by accident.

A note on the two counts, because their relationship is not the obvious one: `n_prior` windows on
the **issue** date (cohort *activity*) while `n_resolved` windows on the **resolution** date
(cohort *experience*). These are different windows on different axes, so `n_resolved <= n_prior`
does **not** hold in general — a neighbour issued forty months ago is outside the issue window but,
having resolved ten months ago, inside the resolution window.

---

## 5. Splitting a graph without leaking

Standard graph-learning practice offers transductive splits (mask node labels, keep all edges) and
inductive splits (hold out subgraphs). **Neither is used here, because the temporal constraint
subsumes both.**

The graph spans *all* nodes — history, train, valid, test and monitoring — and that is safe by
construction, not by partitioning:

- A test node's neighbours are, by the edge rule, always earlier loans. Those may well be training
  loans. **That is correct**: a lender scoring a 2015 application legitimately knows about 2012
  applications. Withholding them would model a lender with amnesia.
- A training node can never see a test node, because test loans are issued later and edges point
  strictly backwards.
- Node features carry no labels, so message passing cannot transport an outcome.
- Cohort label aggregates are gated on resolution, so a 2015 test loan's cohort rate contains only
  outcomes booked before it applied.

The `history` window (2007-06 to 2009-12) exists for exactly one purpose: to give loans at the very
start of the training window real predecessors instead of empty neighbourhoods. History rows are
never modelled and carry no training label.

**Verification is by assertion, not by argument.** `assert_no_future_edges()` checks the property
directly on the built structure — because the construction code is exactly what a future edit might
break — and `tests/test_graph_leakage.py` plants a deliberate violation to confirm the check
actually fires.

---

## 6. The GNN

GraphSAGE, implemented directly in PyTorch (~200 lines, no `torch-geometric`). One layer:

```
h_i' = sigma( W . [ h_i || mean_{j in N_zip3(i)} h_j || mean_{j in N_emp(i)} h_j ] )
```

**Concatenation rather than averaging** the self-representation with the neighbourhood means. That
lets the model weigh "what this borrower looks like" against "what this borrower's cohort looks
like" instead of blurring the two, and gives each relation its own slice of the weight matrix so
geography and employment are not forced to share a coefficient.

**Two layers**, giving each node a two-hop receptive field: the borrower, their cohort, and their
cohort's cohort. Not three — on a cohort graph the three-hop neighbourhood approaches the whole
portfolio and every embedding converges to the same vector, which is over-smoothing.

**Fan-out `[10, 5]`.** The two-hop neighbourhood grows as the product of the fan-outs, and distant
neighbours contribute progressively less, so the second hop is sampled more thinly.

**`balance_classes` defaults to off.** Up-weighting the positive class distorts the effective base
rate exactly as resampling does; in testing it produced an observed/expected ratio near 3. Since
these scores feed the decision layer, the default preserves the base rate and calibration is
applied downstream, the same as for LightGBM.

**Why not `torch-geometric`?** Two reasons, in order. A hand-written sampler and aggregator can be
explained line by line, and "the library did it" is not an answer to "how does message passing
work?". Second, PyG's neighbour sampling needs `pyg-lib` or `torch-sparse`, whose Windows wheels
are unreliable.

---

## 7. The three-arm comparison, and the expected result

| Arm | Model | Question |
|---|---|---|
| **A** | LightGBM on tabular features | Control |
| **B** | LightGBM + time-respecting cohort aggregates | EXP04: does relational information help at all? |
| **C** | GraphSAGE over sampled cohort neighbourhoods | EXP05: does a *GNN* beat the aggregates? |

**Arm B is the honest competitor to the GNN, and it is deliberately strong.** Before claiming a
graph neural network adds value, it has to beat what you get from simply aggregating the cohort's
recent history into columns and handing them to the same model.

The GNN is **not** assumed to win, and the plausible outcome is that it does not. Mean-aggregation
over a one-hop cohort is close to what the aggregate features compute in closed form, so the GNN is
re-learning a known function from less signal with more parameters. On a genuine transaction
network — where multi-hop structure carries information no fixed aggregate captures — the
comparison could easily go the other way. This dataset has no such network, and saying so is more
useful than a marginal AUC gain would be.

There is also a deployment argument for arm B independent of accuracy: a cohort feature's
contribution is legible in a SHAP plot and can be given to an applicant as a reason. A GNN's
neighbourhood aggregation cannot, which matters under adverse-action requirements.

**If arm B beats arm A only slightly or not at all**, that is a legitimate finding too, and the
likely explanation is redundancy: `addr_state` and `emp_length` are already tabular features, so
the cohort aggregates are partly re-encoding information the model already had.

---

## 8. Computational profile

| Quantity | Full dataset |
|---|---|
| Nodes | ~620k modelled + ~100k history |
| Edges | ~12M (2 relations x K=10) |
| Neighbour arrays | ~50 MB |
| Construction time | 2-5 minutes |
| GraphSAGE training | 10-20 min GPU / 40-60 min CPU |
| Peak GPU memory | well under 6 GB at batch 1024 |
