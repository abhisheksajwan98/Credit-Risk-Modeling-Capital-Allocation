"""The decision problem, stated formally.

Why this is reinforcement learning and not supervised learning
--------------------------------------------------------------
It is worth being blunt, because "we applied RL to lending" is usually a decoration.

Predicting default is supervised learning: there is a label, and the prediction does not change it.
*Deciding* is not, for two reasons.

1. **The label you need is counterfactual.** You never observe what a rejected applicant would have
   repaid, so there is no supervised target for "was rejecting correct?". You observe a reward for
   the action you took and nothing for the others. That is the bandit setting by definition.

2. **Under a capital constraint the problem is genuinely sequential.** This is the part that
   earns the "RL" rather than just "bandit". With an unconstrained balance sheet, the optimal
   policy is exactly the myopic per-applicant argmax of expected profit -- approving a good loan
   never costs you anything elsewhere, so there is no trade-off across time and *RL cannot beat
   ExpectedProfitPolicy*. We expect and report that null result.

   When capital binds, funding a marginal applicant today consumes capital a better applicant
   tomorrow cannot use. The decision now depends on how much budget is left and how many
   applications remain. That is a Markov decision process, and the myopic rule is provably
   suboptimal in it.

The shadow price of capital
---------------------------
The constrained problem has known structure, which is what makes the RL result checkable rather
than mysterious. Maximising total profit subject to ``sum(exposure) <= B`` has the Lagrangian::

    max  sum_i [ profit_i(a) - lambda * exposure_i(a) ]

so the optimal rule is: **approve when profit per dollar of capital exceeds lambda**, where
``lambda`` is the shadow price of capital -- the marginal profit of the last dollar. The myopic
policy is the special case ``lambda = 0``.

A learned value function has to recover exactly this: ``V(budget, t)`` is the expected future
profit from the remaining budget, and its derivative with respect to budget *is* lambda. So the
experiment has a built-in check. If the Q-learner is working, the implied threshold should track
the analytical ``lambda``, and :class:`ExpectedProfitPolicy` with ``capital_shadow_price=lambda``
should match it. If the learned policy beats a well-tuned analytical lambda by a lot, that is a bug,
not a breakthrough.

State design
------------
Deliberately small and discrete, for three reasons: it trains in seconds, it needs no neural
network, and the resulting Q-table can be **printed and read**. Being able to show an interviewer
the learned policy as a grid is worth more than a marginally better score from a DQN.

* ``profit_density`` -- expected profit per dollar at medium exposure, in 8 bins. This rather than
  raw PD, because profit density is the sufficient statistic for the decision: it already folds in
  the interest rate, and a fixed PD threshold's failure to do so is precisely what EXP03 exposes.
* ``budget_fraction`` -- capital remaining, in 5 bins.
* ``time_fraction`` -- applications remaining in the period, in 4 bins.

160 states by 5 actions is 800 values, which a few thousand simulated episodes estimate comfortably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from credit_risk.decision.expected_loss import CashflowModel, EconomicsConfig, expected_profit
from credit_risk.decision.policies import ActionSpace, DecisionContext


@dataclass(frozen=True)
class StateSpec:
    """Discretisation of the MDP state."""

    #: Edges for expected profit per dollar of exposure. Spans clearly-unprofitable through
    #: clearly-profitable; the interesting decisions live in the middle bins.
    profit_density_edges: tuple[float, ...] = (-0.10, -0.03, 0.0, 0.02, 0.05, 0.09, 0.14)
    budget_edges: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75)
    time_edges: tuple[float, ...] = (0.25, 0.50, 0.75)

    @property
    def n_profit_bins(self) -> int:
        return len(self.profit_density_edges) + 1

    @property
    def n_budget_bins(self) -> int:
        return len(self.budget_edges) + 1

    @property
    def n_time_bins(self) -> int:
        return len(self.time_edges) + 1

    @property
    def n_states(self) -> int:
        return self.n_profit_bins * self.n_budget_bins * self.n_time_bins

    def encode(
        self, profit_density: float, budget_fraction: float, time_fraction: float
    ) -> int:
        p = int(np.digitize(profit_density, self.profit_density_edges))
        b = int(np.digitize(budget_fraction, self.budget_edges))
        t = int(np.digitize(time_fraction, self.time_edges))
        return (p * self.n_budget_bins + b) * self.n_time_bins + t

    def decode(self, state: int) -> tuple[int, int, int]:
        t = state % self.n_time_bins
        rest = state // self.n_time_bins
        b = rest % self.n_budget_bins
        p = rest // self.n_budget_bins
        return p, b, t

    def describe(self, state: int) -> str:
        p, b, t = self.decode(state)
        return f"profit_bin={p} budget_bin={b} time_bin={t}"


class StateEncoder:
    """Turns a :class:`DecisionContext` into a discrete state and a continuous context vector.

    The discrete state feeds the tabular Q-learner; the continuous vector feeds the contextual
    bandits. Both are produced here so the two families of policy are guaranteed to see exactly
    the same information -- otherwise a comparison between them measures feature engineering
    rather than algorithm.
    """

    def __init__(
        self,
        cashflow_model: CashflowModel,
        economics: EconomicsConfig | None = None,
        action_space: ActionSpace | None = None,
        spec: StateSpec | None = None,
        total_budget: float | None = None,
        total_steps: int | None = None,
    ) -> None:
        self.cashflow_model = cashflow_model
        self.economics = economics or EconomicsConfig()
        self.action_space = action_space or ActionSpace()
        self.spec = spec or StateSpec()
        self.total_budget = total_budget
        self.total_steps = total_steps
        self._medium = self.action_space.names.index("approve_medium")

    def profit_density(self, context: DecisionContext) -> np.ndarray:
        """Expected profit per dollar of exposure at medium exposure.

        The sufficient statistic for the myopic decision, and the primary axis of the state.
        """
        exposures = self.action_space.exposures(context.requested_amount)[:, self._medium]
        profit = expected_profit(
            context.pd_estimate, exposures, context.apr, self.cashflow_model, self.economics
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            density = np.where(exposures > 0, profit / exposures, 0.0)
        return np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)

    def budget_fraction(self, context: DecisionContext) -> float:
        if context.remaining_budget is None or not self.total_budget:
            return 1.0
        return float(np.clip(context.remaining_budget / self.total_budget, 0.0, 1.0))

    def time_fraction(self, context: DecisionContext) -> float:
        if context.remaining_steps is None or not self.total_steps:
            return 1.0
        return float(np.clip(context.remaining_steps / self.total_steps, 0.0, 1.0))

    def encode(self, context: DecisionContext) -> np.ndarray:
        """Discrete state index per applicant in the context."""
        density = self.profit_density(context)
        budget = self.budget_fraction(context)
        time_left = self.time_fraction(context)
        return np.array(
            [self.spec.encode(float(d), budget, time_left) for d in density], dtype=int
        )

    def context_vector(self, context: DecisionContext) -> np.ndarray:
        """Continuous feature vector for the linear bandits, shape ``(n, d)``.

        Kept to six interpretable dimensions plus an intercept. A linear bandit with hundreds of
        features needs far more data than one funding period provides, and the exploration
        behaviour becomes impossible to reason about.
        """
        pd_hat = np.clip(np.asarray(context.pd_estimate, dtype=float), 1e-6, 1 - 1e-6)
        requested = np.asarray(context.requested_amount, dtype=float)
        apr = np.asarray(context.apr, dtype=float)
        density = self.profit_density(context)

        return np.column_stack(
            [
                np.ones_like(pd_hat),                       # intercept
                np.log(pd_hat / (1.0 - pd_hat)) / 5.0,      # PD as scaled log-odds
                apr,                                        # price
                np.log1p(requested) / 12.0,                 # size, log-scaled
                density,                                    # expected profit per dollar
                np.full_like(pd_hat, self.budget_fraction(context)),
                np.full_like(pd_hat, self.time_fraction(context)),
            ]
        )

    @property
    def context_dim(self) -> int:
        return 7

    def describe(self) -> dict[str, Any]:
        return {
            "n_states": self.spec.n_states,
            "n_actions": len(self.action_space),
            "context_dim": self.context_dim,
            "profit_density_edges": list(self.spec.profit_density_edges),
            "budget_edges": list(self.spec.budget_edges),
            "time_edges": list(self.spec.time_edges),
        }


def analytical_shadow_price(
    profit_densities: np.ndarray,
    exposures: np.ndarray,
    budget: float,
) -> float:
    """The exact shadow price of capital for a known applicant pool.

    Solves the continuous relaxation of the knapsack: sort applicants by profit per dollar, fill
    the budget from the top, and lambda is the profit density of the marginal applicant -- the
    first one that does not fit.

    This is the answer the Q-learner should approximately recover, and having it in closed form is
    what turns "the RL policy scored higher" into a checkable claim. It requires knowing the whole
    pool in advance, which the online policy does not, so it is an upper bound rather than a
    competitor.
    """
    order = np.argsort(-profit_densities)
    cumulative = np.cumsum(exposures[order])
    fits = cumulative <= budget
    if fits.all():
        return 0.0  # budget does not bind: lambda is zero and myopic is optimal
    marginal = int(np.argmax(~fits))
    return float(max(profit_densities[order][marginal], 0.0))
