"""Policy evaluation: comparing decision rules without fooling yourself.

Three disciplines are enforced here, because a simulated reward number on its own is worth very
little.

**Common random numbers.** Every policy sees the same applicants in the same order with the same
coupled outcome draws, seed by seed. Differences between policies are then differences in
decisions, not in luck. This turns the comparison into a *paired* one, which typically shrinks the
confidence interval on the difference by an order of magnitude relative to independent runs.

**Paired intervals, on the difference.** The quantity of interest is never "policy A earned X". It
is "policy A earned X more than policy B, and here is the interval on that difference". A bootstrap
over the paired per-seed differences gives that directly.

**Worst case, not just the mean.** A policy that earns more on average while occasionally
destroying the book is not better. Minimum episode reward and the 5th percentile are reported
alongside the mean, because in lending the left tail is the whole subject.

What this cannot tell you
-------------------------
That a policy is safe to deploy. Every number here is generated inside a simulator whose
assumptions are listed in ``docs/ASSUMPTIONS.md``, on a population of *already-approved*
LendingClub borrowers from one credit cycle. The correct claim is "policy A dominates policy B
under these assumptions", and ``docs/RL_FORMULATION.md`` makes no larger one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from credit_risk.decision.policies import Policy
from credit_risk.simulation.environment import EpisodeResult, LendingEnvironment
from credit_risk.utils.runtime import get_logger

LOG = get_logger("simulation.evaluation")


@dataclass
class PolicyEvaluation:
    """Per-episode results for one policy across seeds."""

    name: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    description: dict[str, Any] = field(default_factory=dict)

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.episodes])

    def rewards(self) -> np.ndarray:
        return np.array([e.total_reward for e in self.episodes], dtype=float)

    def summary(self) -> dict[str, float]:
        frame = self.frame
        rewards = self.rewards()
        return {
            "n_episodes": float(len(rewards)),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "min_reward": float(rewards.min()),
            "p05_reward": float(np.percentile(rewards, 5)),
            "approval_rate": float(frame["approval_rate"].mean()),
            "default_rate_of_book": float(frame["default_rate_of_book"].mean()),
            "capital_deployed": float(frame["capital_deployed"].mean()),
            "return_on_capital": float(frame["return_on_capital"].mean()),
            "revenue": float(frame["revenue"].mean()),
            "losses": float(frame["losses"].mean()),
            "costs": float(frame["costs"].mean()),
            "review_rate": float(frame["n_reviewed"].mean() / max(frame["n_applications"].mean(), 1)),
        }


def evaluate_policy(
    env: LendingEnvironment,
    policy: Policy,
    seeds: Sequence[int],
    keep_logs: bool = False,
) -> PolicyEvaluation:
    """Run ``policy`` once per seed in ``env``."""
    episodes = []
    for seed in seeds:
        episodes.append(env.run(policy, seed=seed, keep_log=keep_logs))
    evaluation = PolicyEvaluation(
        name=policy.name, episodes=episodes, description=policy.describe()
    )
    summary = evaluation.summary()
    LOG.info(
        "%-28s reward %10.0f +/- %8.0f | approve %5.1f%% | book default %5.2f%% | RoC %6.3f",
        policy.name,
        summary["mean_reward"],
        summary["std_reward"],
        100 * summary["approval_rate"],
        100 * summary["default_rate_of_book"],
        summary["return_on_capital"],
    )
    return evaluation


def compare_policies(
    env: LendingEnvironment,
    policies: Sequence[Policy],
    seeds: Sequence[int],
    keep_logs: bool = False,
) -> dict[str, PolicyEvaluation]:
    """Evaluate several policies over the identical set of seeds."""
    seeds = list(seeds)
    LOG.info("evaluating %d policies over %d seeds", len(policies), len(seeds))
    return {p.name: evaluate_policy(env, p, seeds, keep_logs) for p in policies}


def summary_table(evaluations: dict[str, PolicyEvaluation]) -> pd.DataFrame:
    """One row per policy, sorted by mean episode reward."""
    rows = {name: ev.summary() for name, ev in evaluations.items()}
    return pd.DataFrame(rows).T.sort_values("mean_reward", ascending=False)


def paired_difference(
    evaluations: dict[str, PolicyEvaluation],
    treatment: str,
    baseline: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap confidence interval on the *paired* per-seed reward difference.

    Because both policies ran on identical seeds, ``diff[i]`` is a like-for-like comparison on the
    same applicants with the same outcome draws. Resampling those differences gives an interval on
    the improvement itself.

    An interval that straddles zero is a null result, and this project reports null results.
    """
    if treatment not in evaluations or baseline not in evaluations:
        raise KeyError(f"need both {treatment!r} and {baseline!r} in evaluations")

    a = evaluations[treatment].rewards()
    b = evaluations[baseline].rewards()
    if len(a) != len(b):
        raise ValueError("paired comparison requires the same number of episodes per policy")

    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_bootstrap, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "treatment": treatment,
        "baseline": baseline,
        "n_pairs": float(len(diff)),
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "win_rate": float((diff > 0).mean()),
        "significant": bool(lo > 0 or hi < 0),
        "relative_improvement": (
            float(diff.mean() / abs(b.mean())) if b.mean() != 0 else float("nan")
        ),
    }


def difference_table(
    evaluations: dict[str, PolicyEvaluation],
    baseline: str,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired comparison of every policy against one baseline."""
    rows = [
        paired_difference(evaluations, name, baseline, n_bootstrap=n_bootstrap, seed=seed)
        for name in evaluations
        if name != baseline
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("treatment")
    return frame.sort_values("mean_difference", ascending=False)


def verdict(differences: pd.DataFrame, baseline: str) -> str:
    """A sentence stating what the comparison actually showed, including when it showed nothing."""
    if differences.empty:
        return f"No policies were compared against {baseline}."
    best = differences.iloc[0]
    name = str(differences.index[0])
    if not bool(best["significant"]):
        return (
            f"No policy beat {baseline} significantly. Best was {name} at "
            f"{best['mean_difference']:+,.0f} per episode, 95% CI "
            f"[{best['ci_low']:+,.0f}, {best['ci_high']:+,.0f}] -- the interval includes zero, "
            f"so this is a null result."
        )
    return (
        f"{name} beat {baseline} by {best['mean_difference']:+,.0f} per episode "
        f"({100 * best['relative_improvement']:+.1f}%), 95% CI "
        f"[{best['ci_low']:+,.0f}, {best['ci_high']:+,.0f}], winning on "
        f"{100 * best['win_rate']:.0f}% of seeds."
    )


def sensitivity_sweep(
    env_factory,
    policies: Sequence[Policy],
    seeds: Sequence[int],
    parameter: str,
    values: Sequence[float],
) -> pd.DataFrame:
    """Re-run the comparison across values of one assumption.

    The point is to find out which conclusions survive the assumptions being wrong. A policy
    ranking that flips when the cost of funds moves 100bp is not a finding worth reporting; one
    that holds across the sweep is.
    """
    rows = []
    for value in values:
        env = env_factory(value)
        evaluations = compare_policies(env, policies, seeds)
        for name, evaluation in evaluations.items():
            summary = evaluation.summary()
            summary.update({parameter: value, "policy": name})
            rows.append(summary)
    return pd.DataFrame(rows).set_index([parameter, "policy"])
