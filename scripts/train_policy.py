#!/usr/bin/env python
"""EXP03 + EXP06: decision policies, and whether a learned policy beats them.

    python scripts/train_baseline.py --config configs/models/boosting.yaml
    python scripts/train_policy.py   --config configs/experiments/exp06_rl.yaml

EXP03 runs the static policy ladder under an **unconstrained** balance sheet. EXP06 repeats it
across a sweep of capital budgets and adds the learning policies.

The headline claim is a pair, not a single number:

* **Unconstrained**, the myopic expected-profit rule is provably optimal, so RL should show *no*
  gain. Reporting only the constrained case would be misleading.
* **Constrained**, capital has an opportunity cost, the problem becomes sequential, and a
  budget-aware policy should win.

The analytical shadow price is computed alongside as a reference. If the learned policy wildly
exceeds an expected-profit policy given the correct shadow price, that is a bug in the simulator,
not a breakthrough -- see docs/RL_FORMULATION.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_risk.decision.expected_loss import EconomicsConfig, fit_cashflow_model  # noqa: E402
from credit_risk.decision.policies import (  # noqa: E402
    ActionSpace,
    ApproveAllPolicy,
    DecisionContext,
    ExpectedProfitPolicy,
    FixedThresholdPolicy,
    RiskBandPolicy,
)
from credit_risk.rl.formulation import StateEncoder, analytical_shadow_price  # noqa: E402
from credit_risk.rl.policy import (  # noqa: E402
    BanditConfig,
    BudgetAwareQPolicy,
    LinUCBPolicy,
    QLearningConfig,
    ThompsonSamplingPolicy,
    train_policy,
)
from credit_risk.simulation.environment import (  # noqa: E402
    LendingEnvironment,
    SimulationConfig,
    build_applicant_pool,
)
from credit_risk.simulation.evaluation import (  # noqa: E402
    compare_policies,
    difference_table,
    summary_table,
    verdict,
)
from credit_risk.utils.cli import (  # noqa: E402
    base_parser,
    detect_source,
    load_prepared,
    resolve,
    save_json,
    save_table,
    write_manifest,
    write_report,
)
from credit_risk.utils.runtime import get_logger  # noqa: E402

LOG = get_logger("scripts.policy")


def _score_split(paths, pd_run: str, frame: pd.DataFrame) -> np.ndarray | None:
    """Score an arbitrary split with the saved model, or return None if artefacts are absent.

    Needed so that learning policies can be trained on a *disjoint* applicant pool. Without this
    they train and are evaluated on the same borrowers, which lets them memorise realised outcomes
    that the static policies they are compared against never see.
    """
    from credit_risk.features.build import FeatureBuilder
    from credit_risk.models.boosting import BoostingModel
    from credit_risk.models.calibration import ProbabilityCalibrator

    builder_path = paths.artifacts / f"{pd_run}_features.pkl"
    model_path = paths.artifacts / f"{pd_run}_boosting.pkl"
    if not builder_path.exists() or not model_path.exists():
        return None

    builder = FeatureBuilder.load(builder_path)
    model = BoostingModel.load(model_path)
    raw = model.predict_proba(builder.transform(frame))
    for method in ("platt", "isotonic"):
        calibrator_path = paths.artifacts / f"{pd_run}_calibrator_{method}.pkl"
        if calibrator_path.exists():
            return ProbabilityCalibrator.load(calibrator_path).transform(raw)
    return raw


def _load_pd(paths, pd_run: str, test: pd.DataFrame, train: pd.DataFrame,
             valid: pd.DataFrame, seed: int) -> np.ndarray:
    """Load calibrated PDs from a completed baseline run, or fit one if absent."""
    cached = paths.artifacts / f"{pd_run}_pd_test.npy"
    if cached.exists():
        scores = np.load(cached)
        if len(scores) == len(test):
            LOG.info("loaded calibrated PDs from %s", cached.name)
            return scores
        LOG.warning("%s has %d rows but test has %d; refitting", cached.name, len(scores), len(test))

    LOG.info("no cached PDs; fitting a model inline")
    from credit_risk.features.build import FeatureBuilder, FeatureSpec
    from credit_risk.models.boosting import BoostingConfig, BoostingModel
    from credit_risk.models.calibration import ProbabilityCalibrator

    builder = FeatureBuilder(FeatureSpec()).fit(train, valid)
    y_train = train["default"].astype(int).to_numpy()
    y_valid = valid["default"].astype(int).to_numpy()
    model = BoostingModel(
        builder.categorical_features, BoostingConfig(n_estimators=1500)
    ).fit(builder.transform(train), y_train, builder.transform(valid), y_valid, seed=seed)
    calibrator = ProbabilityCalibrator("isotonic").fit(
        y_valid, model.predict_proba(builder.transform(valid))
    )
    return calibrator.transform(model.predict_proba(builder.transform(test)))


def main() -> int:
    parser = base_parser(__doc__ or "", default_config="configs/experiments/exp06_rl.yaml")
    parser.add_argument("--pd-run", default="boosting", help="Baseline run whose PDs to reuse.")
    parser.add_argument("--skip-rl", action="store_true", help="EXP03 only.")
    args = parser.parse_args()
    config, paths, run_name, seed = resolve(args)
    source = detect_source(paths)

    df = load_prepared(paths)
    train, valid, test = (df[df["split"] == s] for s in ("train", "valid", "test"))
    pd_test = _load_pd(paths, args.pd_run, test, train, valid, seed)

    # -- economics, estimated from observed cashflows on the training window ---
    ecfg = config.get("economics", {})
    economics = EconomicsConfig(
        annual_cost_of_funds=float(ecfg.get("annual_cost_of_funds", 0.03)),
        opex_fixed=float(ecfg.get("opex_fixed", 150.0)),
        opex_variable_rate=float(ecfg.get("opex_variable_rate", 0.010)),
        manual_review_cost=float(ecfg.get("manual_review_cost", 75.0)),
        weighted_average_life_years=float(ecfg.get("weighted_average_life_years", 1.55)),
    )
    cashflow = fit_cashflow_model(train, economics)

    action_space = ActionSpace(max_exposure=float(config.get_path("actions.max_exposure", 40000.0)))
    if not bool(config.get_path("actions.include_review", False)):
        action_space = action_space.without_review()
    LOG.info("action space: %s", action_space.names)

    pool = build_applicant_pool(test, pd_test, cashflow)
    scfg = config.get("simulation", {})
    n_steps = int(scfg.get("n_steps", 2000))
    seeds = list(config.get_path("evaluation.seeds", list(range(12))))

    # Learning policies are trained on the VALIDATION window and evaluated on the test window.
    # Training them on the evaluation pool would let them memorise the realised outcomes of the
    # very borrowers they are scored on -- an advantage the static policies do not have, which
    # makes the comparison meaningless. If the validation PDs cannot be produced, the learning
    # policies are skipped rather than trained unfairly.
    pd_valid = _score_split(paths, args.pd_run, valid)
    train_pool = build_applicant_pool(valid, pd_valid, cashflow) if pd_valid is not None else None
    if train_pool is None:
        LOG.warning(
            "could not score the validation window; learning policies will be SKIPPED rather "
            "than trained on the evaluation pool"
        )
    else:
        LOG.info(
            "learning policies train on %s validation applicants, evaluate on %s test applicants",
            f"{len(train_pool):,}", f"{len(pool):,}",
        )

    # Capital that approving every applicant at medium exposure would consume.
    medium = action_space.names.index("approve_medium")
    exposures = action_space.exposures(pool.requested_amount)[:, medium]
    full_demand = float(exposures[:n_steps].sum())

    encoder0 = StateEncoder(cashflow, economics, action_space)
    densities = encoder0.profit_density(
        DecisionContext(pool.pd_estimate, pool.requested_amount, pool.apr)
    )

    def make_env(budget: float | None, on_pool=None) -> LendingEnvironment:
        return LendingEnvironment(
            on_pool if on_pool is not None else pool, cashflow, economics, action_space,
            SimulationConfig(
                n_steps=n_steps,
                capital_budget=budget,
                review_information_gain=float(scfg.get("review_information_gain", 0.15)),
                review_signal_noise=float(scfg.get("review_signal_noise", 0.45)),
                pd_exposure_elasticity=float(scfg.get("pd_exposure_elasticity", 0.15)),
            ),
        )

    def static_policies(space: ActionSpace, shadow: float = 0.0):
        pols = [
            ApproveAllPolicy(space),
            FixedThresholdPolicy(float(config.get_path("policies.fixed_threshold.threshold", 0.15)),
                                 space),
            RiskBandPolicy(
                band_edges=tuple(config.get_path("policies.risk_band.band_edges",
                                                 [0.05, 0.10, 0.18, 0.28])),
                action_space=space,
            ),
            ExpectedProfitPolicy(cashflow, economics, space),
        ]
        if shadow > 0:
            priced = ExpectedProfitPolicy(cashflow, economics, space, capital_shadow_price=shadow)
            priced.name = f"expected_profit(lambda={shadow:.4f})"
            pols.append(priced)
        return pols

    # =====================================================================
    # EXP03 - unconstrained capital
    # =====================================================================
    LOG.info("EXP03: unconstrained capital")
    env_free = make_env(None)
    evaluations = compare_policies(env_free, static_policies(action_space), seeds)
    exp03_summary = summary_table(evaluations)
    exp03_diff = difference_table(evaluations, "fixed_threshold@0.150",
                                  n_bootstrap=int(config.get_path("evaluation.n_bootstrap", 10000)))
    save_table(exp03_summary, paths, f"{run_name}_exp03_summary")
    save_table(exp03_diff, paths, f"{run_name}_exp03_differences")

    # =====================================================================
    # EXP06 - capital sweep, with learning policies
    # =====================================================================
    sweep_rows: list[dict] = []
    budget_reports: dict[str, dict] = {}
    degenerate: list[dict] = []
    fractions = list(config.get_path("budget_sweep.fractions", [1.0, 0.6, 0.35, 0.2]))
    training_curves: dict[str, list[float]] = {}

    if not args.skip_rl:
        for fraction in fractions:
            budget = None if fraction >= 0.999 else fraction * full_demand
            label = "unconstrained" if budget is None else f"budget={fraction:.2f}"
            shadow = (
                0.0 if budget is None
                else analytical_shadow_price(densities[:n_steps], exposures[:n_steps], budget)
            )
            LOG.info("EXP06 | %s | analytical shadow price lambda=%.4f", label, shadow)

            env = make_env(budget)
            # Learning happens here; scoring happens in `env`. Different borrowers, same rules.
            train_env = (
                make_env(budget, on_pool=train_pool) if train_pool is not None else None
            )
            encoder = StateEncoder(
                cashflow, economics, action_space,
                total_budget=budget or full_demand, total_steps=n_steps,
            )
            policies = static_policies(action_space, shadow)

            q_policy = BudgetAwareQPolicy(
                encoder, action_space,
                QLearningConfig(
                    learning_rate=float(config.get_path("rl.q_learning.learning_rate", 0.15)),
                    gamma=float(config.get_path("rl.q_learning.gamma", 1.0)),
                    epsilon_start=float(config.get_path("rl.q_learning.epsilon_start", 0.30)),
                    epsilon_end=float(config.get_path("rl.q_learning.epsilon_end", 0.02)),
                    epsilon_decay_episodes=int(
                        config.get_path("rl.q_learning.epsilon_decay_episodes", 200)
                    ),
                    seed=seed,
                ),
            )
            if train_env is not None:
                report = train_policy(
                    train_env, q_policy,
                    int(config.get_path("rl.train_episodes", 400)),
                    seed_offset=int(config.get_path("rl.seed_offset", 10000)),
                )
                training_curves[label] = report.episode_rewards
                policies.append(q_policy.greedy())

            bandit_cfg = BanditConfig(
                alpha=float(config.get_path("rl.bandit.alpha", 1.0)),
                ridge_lambda=float(config.get_path("rl.bandit.ridge_lambda", 1.0)),
                reward_scale=float(config.get_path("rl.bandit.reward_scale", 1000.0)),
            )
            if train_env is not None:
                for bandit in (
                    LinUCBPolicy(encoder, action_space, bandit_cfg),
                    ThompsonSamplingPolicy(encoder, action_space, bandit_cfg, seed=seed),
                ):
                    train_policy(train_env, bandit,
                                 int(config.get_path("rl.bandit_episodes", 120)),
                                 seed_offset=int(config.get_path("rl.seed_offset", 10000)))
                    # A bandit that has collapsed onto a single action is not a policy result.
                    # Report it as degenerate rather than quoting its reward alongside the others.
                    chosen = int((bandit.counts > 0).sum())
                    if chosen <= 1:
                        LOG.warning(
                            "%s collapsed onto %d action(s) during training and is reported as "
                            "degenerate rather than compared", bandit.name, chosen,
                        )
                        degenerate.append({"condition": label, "policy": bandit.name,
                                           "actions_used": chosen,
                                           "action_counts": bandit.counts.tolist()})
                    policies.append(bandit)

            evals = compare_policies(env, policies, seeds)
            summary = summary_table(evals)
            diffs = difference_table(evals, "expected_profit")
            # The control: with unconstrained capital the myopic expected-profit rule is
            # optimal for the objective it optimises, so a learner beating it does not
            # demonstrate that RL helps -- it demonstrates that the myopic policy's *inputs*
            # are wrong (its cashflow coefficients are fitted on the training vintage and
            # applied to a later one). This check states that plainly instead of leaving the
            # reader to notice.
            control_breach = None
            if budget is None and len(diffs):
                beaten = diffs[(diffs["mean_difference"] > 0) & diffs["significant"]]
                learners = [n for n in beaten.index if "q(" in n or "linucb" in n or "thompson" in n]
                if learners:
                    control_breach = (
                        f"CONTROL BREACHED: {', '.join(learners)} significantly beat the myopic "
                        f"expected-profit policy with unconstrained capital, where it should be "
                        f"optimal. This does not support an RL claim. It indicates the myopic "
                        f"policy is optimising a misspecified objective -- its good-loan yield "
                        f"and loss-rate coefficients are fitted on the 2010-2014 training window "
                        f"and applied to the 2015 test vintage."
                    )
                    LOG.warning(control_breach)

            budget_reports[label] = {
                "shadow_price": shadow,
                "budget": budget,
                "verdict": verdict(diffs, "expected_profit"),
                "control_breach": control_breach,
                "q_implied_threshold": q_policy.implied_threshold(),
            }
            for name, row in summary.iterrows():
                sweep_rows.append({"condition": label, "policy": name, **row.to_dict()})
            save_table(summary, paths, f"{run_name}_exp06_{label.replace('=', '').replace('.', '')}")
            save_table(diffs, paths,
                       f"{run_name}_exp06_diff_{label.replace('=', '').replace('.', '')}")

    sweep = pd.DataFrame(sweep_rows)
    if len(sweep):
        save_table(sweep, paths, f"{run_name}_exp06_sweep", index=False)

    # -- figures ---------------------------------------------------------------
    if not args.no_figures:
        from credit_risk.evaluation.plots import learning_curve, policy_comparison, save
        save(policy_comparison(exp03_summary, title="EXP03: unconstrained capital"),
             paths.results_figures / f"{run_name}_exp03.png")
        for label, rewards in training_curves.items():
            save(learning_curve(rewards, title=f"Q-learning ({label})"),
                 paths.results_figures / f"{run_name}_learning_{label.replace('=', '')}.png")

    # -- verdicts --------------------------------------------------------------
    lines = [
        f"# {run_name}: EXP03 and EXP06",
        "",
        f"Applicant pool: {len(pool):,} out-of-time loans; {n_steps:,} processed per episode; "
        f"{len(seeds)} seeds under common random numbers.",
        (
            f"Learning policies are trained on {len(train_pool):,} **validation-window** "
            f"applicants and evaluated on the test window, so they never see the borrowers they "
            f"are scored on. An earlier version trained them on the evaluation pool itself, which "
            f"gave them an advantage the static policies did not have."
            if train_pool is not None
            else "Learning policies were SKIPPED: validation-window PDs were unavailable, and "
            "training them on the evaluation pool would have made the comparison invalid."
        ),
        f"Estimated from training-window cashflows: good-loan yield "
        f"{cashflow.good_intercept:.3f}{cashflow.good_slope:+.3f}*APR, defaulted-loan loss rate "
        f"{cashflow.bad_intercept:.3f}{cashflow.bad_slope:+.3f}*APR, "
        f"**empirical LGD {cashflow.observed_lgd_mean:.3f}** (mean) / "
        f"{cashflow.observed_lgd_median:.3f} (median).",
        "",
        "## EXP03 - does expected-profit optimisation beat a fixed PD threshold?",
        "",
        exp03_summary.round(4).to_markdown(),
        "",
        exp03_diff.round(1).to_markdown() if len(exp03_diff) else "_no comparisons_",
        "",
        f"**Verdict.** {verdict(exp03_diff, 'fixed_threshold@0.150')}",
        "",
        "The mechanism is worth stating: the expected-profit policy typically runs a *higher* "
        "book default rate than the threshold policy and still earns more, because it approves "
        "high-rate loans whose price covers their risk and declines low-rate loans whose price "
        "does not. A single PD cut-off cannot express that, since the break-even PD moves with "
        "the interest rate. This is a failure of the decision rule, not of the risk model -- "
        "every policy here consumes identical PDs.",
    ]

    if budget_reports:
        lines += [
            "",
            "## EXP06 - can a learned policy beat static policies?",
            "",
            "| condition | shadow price lambda | verdict |",
            "|---|---|---|",
        ]
        for label, payload in budget_reports.items():
            lines.append(f"| {label} | {payload['shadow_price']:.4f} | {payload['verdict']} |")

        breaches = [p["control_breach"] for p in budget_reports.values() if p.get("control_breach")]
        if breaches:
            lines += ["", "### Control check", ""]
            lines += [f"> {b}" for b in breaches]
            lines += [
                "",
                "Learning-policy results in the unconstrained condition should therefore **not** "
                "be read as evidence for reinforcement learning. The correct reading is that the "
                "myopic policy's economic inputs are stale relative to the test vintage.",
            ]
        if degenerate:
            lines += [
                "",
                "### Degenerate policies",
                "",
                "The following collapsed onto a single action during training. Their rewards are "
                "shown in the table for completeness but are **not** policy results:",
                "",
            ]
            lines += [
                f"- `{d['policy']}` under {d['condition']}: used {d['actions_used']} of "
                f"{len(action_space)} actions (counts {d['action_counts']})"
                for d in degenerate
            ]
        lines += [
            "",
            sweep.pivot_table(index="policy", columns="condition", values="mean_reward")
            .round(0).to_markdown(),
            "",
            "**Reading this table.** The `unconstrained` column is the control. There, the myopic "
            "expected-profit rule is optimal by construction and the learned policies should not "
            "beat it -- if they appear to, the simulator has a bug. As the budget tightens, the "
            "shadow price of capital rises and the ranking should change: pricing capital "
            "correctly (either analytically via `expected_profit(lambda=...)` or by learning it "
            "via `budget_aware_q`) starts to matter more than per-applicant accuracy.",
            "",
            "**What this does not establish.** That any of these policies is safe to deploy. "
            "Every number is generated inside a simulator whose assumptions are listed in "
            "docs/ASSUMPTIONS.md, on a population of already-approved LendingClub borrowers from "
            "a single credit cycle. The defensible claim is a ranking under stated assumptions.",
        ]

    write_report(lines, paths, f"{run_name}_exp03_exp06", source)
    save_json(
        {
            "economics": economics.to_dict(),
            "cashflow_model": cashflow.to_dict(),
            "full_demand": full_demand,
            "exp03": exp03_summary.to_dict(),
            "exp06": budget_reports,
            "degenerate_policies": degenerate,
            "training_pool": "validation window" if train_pool is not None else None,
        },
        paths, f"{run_name}_exp03_exp06",
    )
    write_manifest(run_name, config, seed, paths, data_source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
