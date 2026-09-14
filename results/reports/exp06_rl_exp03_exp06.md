> Computed on **real** data.

# exp06_rl: EXP03 and EXP06

Applicant pool: 283,026 out-of-time loans; 1,200 processed per episode; 12 seeds under common random numbers.
Learning policies are trained on 90,615 **validation-window** applicants and evaluated on the test window, so they never see the borrowers they are scored on. An earlier version trained them on the evaluation pool itself, which gave them an advantage the static policies did not have.
Estimated from training-window cashflows: good-loan yield -0.001+1.351*APR, defaulted-loan loss rate 0.353-0.372*APR, **empirical LGD 0.518** (mean) / 0.532 (median).

## EXP03 - does expected-profit optimisation beat a fixed PD threshold?

|                       |   n_episodes |   mean_reward |   std_reward |   min_reward |   p05_reward |   approval_rate |   default_rate_of_book |   capital_deployed |   return_on_capital |     revenue |   losses |            costs |   review_rate |
|:----------------------|-------------:|--------------:|-------------:|-------------:|-------------:|----------------:|-----------------------:|-------------------:|--------------------:|------------:|---------:|-----------------:|--------------:|
| expected_profit       |           12 |      147180   |      82293.9 |     16565.1  |    26735.4   |          0.6592 |                 0.1698 |        1.57855e+07 |              0.0094 | 2.03504e+06 |   877329 |      1.01053e+06 |             0 |
| fixed_threshold@0.150 |           12 |      134189   |      65597.5 |      8176.87 |    43059.3   |          0.6136 |                 0.1    |        9.91585e+06 |              0.0136 | 1.11865e+06 |   313764 | 670696           |             0 |
| risk_band             |           12 |       95036.8 |      58387.4 |     -7946.37 |     -787.312 |          0.9498 |                 0.1328 |        1.1068e+07  |              0.0086 | 1.28091e+06 |   389565 | 796304           |             0 |
| approve_all           |           12 |       64455.4 |      66209.7 |    -68255.5  |   -27585.7   |          1      |                 0.1508 |        1.54395e+07 |              0.0042 | 1.89238e+06 |   775595 |      1.05233e+06 |             0 |

| treatment       | baseline              |   n_pairs |   mean_difference |   ci_low |   ci_high |   win_rate | significant   |   relative_improvement |
|:----------------|:----------------------|----------:|------------------:|---------:|----------:|-----------:|:--------------|-----------------------:|
| expected_profit | fixed_threshold@0.150 |        12 |           12991.1 | -17701.6 |   44103.2 |        0.6 | False         |                    0.1 |
| risk_band       | fixed_threshold@0.150 |        12 |          -39152.3 | -64956.5 |  -13675.9 |        0.2 | True          |                   -0.3 |
| approve_all     | fixed_threshold@0.150 |        12 |          -69733.7 | -94804.2 |  -44851.5 |        0.1 | True          |                   -0.5 |

**Verdict.** No policy beat fixed_threshold@0.150 significantly. Best was expected_profit at +12,991 per episode, 95% CI [-17,702, +44,103] -- the interval includes zero, so this is a null result.

The mechanism is worth stating: the expected-profit policy typically runs a *higher* book default rate than the threshold policy and still earns more, because it approves high-rate loans whose price covers their risk and declines low-rate loans whose price does not. A single PD cut-off cannot express that, since the break-even PD moves with the interest rate. This is a failure of the decision rule, not of the risk model -- every policy here consumes identical PDs.

## EXP06 - can a learned policy beat static policies?

| condition | shadow price lambda | verdict |
|---|---|---|
| unconstrained | 0.0000 | No policy beat expected_profit significantly. Best was thompson at +26,139 per episode, 95% CI [-5,131, +57,164] -- the interval includes zero, so this is a null result. |
| budget=0.60 | 0.0067 | No policy beat expected_profit significantly. Best was expected_profit(lambda=0.0067) at +20,328 per episode, 95% CI [-4,524, +47,971] -- the interval includes zero, so this is a null result. |
| budget=0.35 | 0.0317 | No policy beat expected_profit significantly. Best was expected_profit(lambda=0.0317) at +38,402 per episode, 95% CI [-215, +75,003] -- the interval includes zero, so this is a null result. |
| budget=0.20 | 0.0500 | No policy beat expected_profit significantly. Best was expected_profit(lambda=0.0500) at +14,837 per episode, 95% CI [-12,503, +42,205] -- the interval includes zero, so this is a null result. |

| policy                         |   budget=0.20 |   budget=0.35 |   budget=0.60 |   unconstrained |
|:-------------------------------|--------------:|--------------:|--------------:|----------------:|
| approve_all                    |         20618 |         43450 |         74926 |           64455 |
| budget_aware_q(greedy)         |         33811 |         56347 |        109918 |           91555 |
| expected_profit                |         46934 |         80766 |        133529 |          147180 |
| expected_profit(lambda=0.0067) |           nan |           nan |        153856 |             nan |
| expected_profit(lambda=0.0317) |           nan |        119169 |           nan |             nan |
| expected_profit(lambda=0.0500) |         61771 |           nan |           nan |             nan |
| fixed_threshold@0.150          |         47703 |         85998 |        135812 |          134189 |
| linucb@alpha=1                 |         46219 |         63891 |        117262 |          119252 |
| risk_band                      |         37363 |         63935 |         94775 |           95037 |
| thompson                       |         35437 |         46445 |         45094 |          173320 |

**Reading this table.** The `unconstrained` column is the control. There, the myopic expected-profit rule is optimal by construction and the learned policies should not beat it -- if they appear to, the simulator has a bug. As the budget tightens, the shadow price of capital rises and the ranking should change: pricing capital correctly (either analytically via `expected_profit(lambda=...)` or by learning it via `budget_aware_q`) starts to matter more than per-applicant accuracy.

**What this does not establish.** That any of these policies is safe to deploy. Every number is generated inside a simulator whose assumptions are listed in docs/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from a single credit cycle. The defensible claim is a ranking under stated assumptions.
