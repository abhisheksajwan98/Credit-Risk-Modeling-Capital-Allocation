"""Learning policies: contextual bandits and a budget-aware Q-learner.

Algorithm choice follows the formulation, not fashion. No PPO, no DQN, no actor-critic --
the problem does not call for them and they would be far harder to defend:

* The **one-step** problem (unconstrained capital) is a contextual bandit. LinUCB and Thompson
  sampling are the two canonical answers and both are a few dozen lines of linear algebra.
* The **sequential** problem (binding capital) is a finite-horizon MDP with 160 discrete states.
  Tabular Q-learning solves it exactly and the result can be printed as a grid. A DQN here would
  add a function approximator to a problem with no approximation error to remove.

Exploration versus exploitation, concretely
-------------------------------------------
This is where a lending bandit differs from an advertising one, and it is the honest caveat.
Exploring means *lending money to someone the current estimate says you should decline*, to find
out whether the estimate is wrong. In a simulator that is free. In production it is a real loss on
a real borrower, and it raises fairness questions -- the applicants selected for exploration are by
construction the marginal ones. So the exploration here demonstrates the mechanism; it is not a
recommendation to explore on live applicants without a great deal more thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from credit_risk.decision.policies import ActionSpace, DecisionContext, Policy
from credit_risk.rl.formulation import StateEncoder
from credit_risk.utils.runtime import get_logger

LOG = get_logger("rl.policy")


@dataclass
class BanditConfig:
    #: UCB exploration width. Larger explores more. 1.0 is the standard starting point.
    alpha: float = 1.0
    #: Ridge prior precision. Also the initial uncertainty: higher means more confident at zero.
    ridge_lambda: float = 1.0
    #: Reward scaling. Episode rewards are in dollars and can be thousands; the linear algebra
    #: is far better conditioned on a scale near unity.
    reward_scale: float = 1000.0


class LinUCBPolicy(Policy):
    """LinUCB: one ridge regression per action, chosen by upper confidence bound.

    For each action ``a`` it maintains ``A_a = lambda I + sum x x'`` and ``b_a = sum r x``, giving
    ``theta_a = A_a^-1 b_a``. It then picks::

        argmax_a  theta_a . x  +  alpha * sqrt(x' A_a^-1 x)

    The first term is the estimated reward; the second is the width of its confidence interval.
    Actions that have rarely been tried in this region of context space have a wide interval and
    get chosen occasionally *because* they are uncertain -- optimism in the face of uncertainty.
    That is what makes it explore where exploration is informative, rather than at random.
    """

    is_learning = True

    def __init__(
        self,
        encoder: StateEncoder,
        action_space: ActionSpace | None = None,
        config: BanditConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.action_space = action_space or ActionSpace()
        self.config = config or BanditConfig()
        self.d = encoder.context_dim
        self.n_actions = len(self.action_space)
        self.name = f"linucb@alpha={self.config.alpha:g}"
        self._init_state()

    def _init_state(self) -> None:
        lam = self.config.ridge_lambda
        self.A = np.stack([lam * np.eye(self.d) for _ in range(self.n_actions)])
        self.A_inv = np.stack([np.eye(self.d) / lam for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.d))
        self.counts = np.zeros(self.n_actions, dtype=int)

    def reset(self) -> None:
        # Deliberately *not* reset between episodes: the agent is meant to accumulate knowledge
        # across the funding periods it has seen. Use `reset_learning()` to start over.
        return None

    def reset_learning(self) -> None:
        self._init_state()

    def theta(self) -> np.ndarray:
        return np.einsum("aij,aj->ai", self.A_inv, self.b)

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        x = self.encoder.context_vector(context)  # (n, d)
        theta = self.theta()  # (A, d)
        mean = x @ theta.T  # (n, A)
        # sqrt(x' A^-1 x) per action, without forming the full quadratic form for every pair.
        widths = np.sqrt(
            np.maximum(np.einsum("nd,ade,ne->na", x, self.A_inv, x), 0.0)
        )
        scores = mean + self.config.alpha * widths
        return np.argmax(scores, axis=1)

    def update(self, context: DecisionContext, actions: np.ndarray, rewards: np.ndarray) -> None:
        x = self.encoder.context_vector(context)
        scaled = np.asarray(rewards, dtype=float) / self.config.reward_scale
        for row, action, reward in zip(x, np.asarray(actions).ravel(), scaled, strict=True):
            a = int(action)
            self.A[a] += np.outer(row, row)
            self.b[a] += reward * row
            self.counts[a] += 1
            # Sherman-Morrison rank-one update: O(d^2) instead of an O(d^3) re-inversion.
            Ai = self.A_inv[a]
            Ax = Ai @ row
            self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + row @ Ax)

    def describe(self) -> dict[str, Any]:
        return super().describe() | {
            "alpha": self.config.alpha,
            "action_counts": self.counts.tolist(),
        }


class ThompsonSamplingPolicy(Policy):
    """Bayesian linear bandit: sample a coefficient vector, act greedily against the sample.

    Where LinUCB explores by being deterministically optimistic, Thompson sampling explores by
    being *randomly* optimistic -- it draws ``theta_a`` from its posterior and takes the best
    action under that draw. An action is chosen in proportion to the posterior probability that it
    is the best one, which is a more natural notion of "explore what might be good" and usually
    performs slightly better in practice with one fewer parameter to tune.
    """

    is_learning = True

    def __init__(
        self,
        encoder: StateEncoder,
        action_space: ActionSpace | None = None,
        config: BanditConfig | None = None,
        posterior_scale: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.action_space = action_space or ActionSpace()
        self.config = config or BanditConfig()
        self.posterior_scale = float(posterior_scale)
        self.d = encoder.context_dim
        self.n_actions = len(self.action_space)
        self._rng = np.random.default_rng(seed)
        self.name = "thompson"
        self._init_state()

    def _init_state(self) -> None:
        lam = self.config.ridge_lambda
        self.A_inv = np.stack([np.eye(self.d) / lam for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.d))
        self.counts = np.zeros(self.n_actions, dtype=int)

    def reset_learning(self) -> None:
        self._init_state()

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or self._rng
        x = self.encoder.context_vector(context)
        mean = np.einsum("aij,aj->ai", self.A_inv, self.b)
        scores = np.empty((len(x), self.n_actions))
        for a in range(self.n_actions):
            cov = self.posterior_scale**2 * self.A_inv[a]
            sample = generator.multivariate_normal(mean[a], cov, method="cholesky")
            scores[:, a] = x @ sample
        return np.argmax(scores, axis=1)

    def update(self, context: DecisionContext, actions: np.ndarray, rewards: np.ndarray) -> None:
        x = self.encoder.context_vector(context)
        scaled = np.asarray(rewards, dtype=float) / self.config.reward_scale
        for row, action, reward in zip(x, np.asarray(actions).ravel(), scaled, strict=True):
            a = int(action)
            self.b[a] += reward * row
            self.counts[a] += 1
            Ai = self.A_inv[a]
            Ax = Ai @ row
            self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + row @ Ax)

    def describe(self) -> dict[str, Any]:
        return super().describe() | {
            "posterior_scale": self.posterior_scale,
            "action_counts": self.counts.tolist(),
        }


@dataclass
class QLearningConfig:
    learning_rate: float = 0.15
    #: Undiscounted. The horizon is finite and short (one funding period), and every dollar within
    #: it is worth the same, so discounting would only distort the objective.
    gamma: float = 1.0
    epsilon_start: float = 0.30
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 200
    reward_scale: float = 1000.0
    optimistic_init: float = 0.0
    seed: int = 0


class BudgetAwareQPolicy(Policy):
    """Tabular Q-learning over ``(profit density, budget left, time left) x action``.

    This is the policy that is supposed to beat the myopic rule *only when capital binds*. What it
    has to learn is opportunity cost: that approving a thin-margin applicant early is bad not
    because the loan loses money, but because it spends capital a better applicant would have used.

    The learned table is inspectable. :meth:`implied_threshold` reads the profit-density cut-off
    out of it, which should track the analytical shadow price from
    :func:`credit_risk.rl.formulation.analytical_shadow_price`. That correspondence is the check
    that the learner found the right structure rather than an artefact of the simulator.
    """

    is_learning = True

    def __init__(
        self,
        encoder: StateEncoder,
        action_space: ActionSpace | None = None,
        config: QLearningConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.action_space = action_space or ActionSpace()
        self.config = config or QLearningConfig()
        self.n_states = encoder.spec.n_states
        self.n_actions = len(self.action_space)
        self.Q = np.full((self.n_states, self.n_actions), self.config.optimistic_init, dtype=float)
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=int)
        self._rng = np.random.default_rng(self.config.seed)
        self._episodes_seen = 0
        self._pending: tuple[int, int] | None = None
        self.training = True
        self.name = "budget_aware_q"

    # -- exploration schedule ---------------------------------------------
    @property
    def epsilon(self) -> float:
        if not self.training:
            return 0.0
        cfg = self.config
        frac = min(self._episodes_seen / max(cfg.epsilon_decay_episodes, 1), 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def reset(self) -> None:
        self._pending = None
        self._episodes_seen += 1

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or self._rng
        states = self.encoder.encode(context)
        actions = np.argmax(self.Q[states], axis=1)
        if self.training:
            explore = generator.random(len(states)) < self.epsilon
            if explore.any():
                actions = actions.copy()
                actions[explore] = generator.integers(0, self.n_actions, size=int(explore.sum()))
        self._pending = (int(states[0]), int(actions[0])) if len(states) else None
        return actions

    def update(self, context: DecisionContext, actions: np.ndarray, rewards: np.ndarray) -> None:
        """One-step Q update.

        ``context`` here is the state the agent has *arrived in* after acting, which the simulator
        passes on the following call. The bootstrap target uses the best value available from that
        successor state, which is what propagates the value of saved capital backward in time.
        """
        if self._pending is None or not self.training:
            return
        state, action = self._pending
        reward = float(np.asarray(rewards, dtype=float).ravel()[0]) / self.config.reward_scale

        next_states = self.encoder.encode(context)
        next_value = float(np.max(self.Q[int(next_states[0])])) if len(next_states) else 0.0

        target = reward + self.config.gamma * next_value
        lr = self.config.learning_rate
        self.Q[state, action] += lr * (target - self.Q[state, action])
        self.visits[state, action] += 1

    # -- inspection --------------------------------------------------------
    def implied_threshold(self) -> dict[str, Any]:
        """Read the learned approve/decline boundary out of the table.

        For each (budget, time) cell, find the lowest profit-density bin at which the greedy action
        is not "reject". A rising threshold as budget tightens is the shadow price appearing, and
        is the qualitative result EXP06 is really testing for.
        """
        spec = self.encoder.spec
        reject = self.action_space.reject_index
        rows = []
        for b in range(spec.n_budget_bins):
            for t in range(spec.n_time_bins):
                cutoff = None
                for p in range(spec.n_profit_bins):
                    state = (p * spec.n_budget_bins + b) * spec.n_time_bins + t
                    if self.visits[state].sum() == 0:
                        continue
                    if int(np.argmax(self.Q[state])) != reject:
                        cutoff = p
                        break
                rows.append({"budget_bin": b, "time_bin": t, "approve_from_profit_bin": cutoff})
        return {"table": rows, "n_visited_states": int((self.visits.sum(axis=1) > 0).sum())}

    def greedy(self) -> BudgetAwareQPolicy:
        """Return this policy with exploration switched off, for evaluation."""
        self.training = False
        self.name = "budget_aware_q(greedy)"
        return self

    def describe(self) -> dict[str, Any]:
        return super().describe() | {
            "n_states": self.n_states,
            "epsilon": self.epsilon,
            "states_visited": int((self.visits.sum(axis=1) > 0).sum()),
            "episodes_seen": self._episodes_seen,
        }


@dataclass
class TrainingReport:
    episode_rewards: list[float] = field(default_factory=list)
    epsilon: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_episodes": len(self.episode_rewards),
            "first_10_mean": float(np.mean(self.episode_rewards[:10])) if self.episode_rewards else 0.0,
            "last_10_mean": float(np.mean(self.episode_rewards[-10:])) if self.episode_rewards else 0.0,
            "best": float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
        }


def train_policy(env, policy: Policy, n_episodes: int, seed_offset: int = 10_000) -> TrainingReport:
    """Train a learning policy over repeated simulated funding periods.

    Training seeds are offset well away from the evaluation seeds so that a policy cannot be
    rewarded for having memorised the exact applicant sequence it will later be scored on.
    """
    report = TrainingReport()
    for episode in range(n_episodes):
        result = env.run(policy, seed=seed_offset + episode)
        report.episode_rewards.append(result.total_reward)
        if isinstance(policy, BudgetAwareQPolicy):
            report.epsilon.append(policy.epsilon)
        if (episode + 1) % max(n_episodes // 10, 1) == 0:
            recent = np.mean(report.episode_rewards[-20:])
            LOG.info(
                "episode %4d/%d | recent mean reward %10.0f%s",
                episode + 1,
                n_episodes,
                recent,
                f" | eps {policy.epsilon:.3f}" if isinstance(policy, BudgetAwareQPolicy) else "",
            )
    return report
