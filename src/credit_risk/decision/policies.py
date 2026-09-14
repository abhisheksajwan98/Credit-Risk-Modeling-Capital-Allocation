"""Lending policies: turning a probability into an action.

The ladder, in increasing order of sophistication. Each rung is evaluated in the identical
simulator under identical seeds, so the comparison is about the decision rule and nothing else.

1. :class:`FixedThresholdPolicy` -- approve below a PD cut-off. What most people mean by
   "deploying the model", and the thing EXP03 is designed to beat.
2. :class:`RiskBandPolicy` -- grade-style bands mapped to exposure. How consumer credit policy is
   actually written, because a committee can read it.
3. :class:`ExpectedProfitPolicy` -- per-applicant argmax of expected profit. The economically
   correct myopic rule, and a genuinely strong baseline.

The learning policies (contextual bandit, budget-aware Q) live in :mod:`credit_risk.rl.policy`,
because they need to see outcomes and update. Everything here is a fixed function of the context.

Why a threshold policy is the wrong shape
-----------------------------------------
A single PD cut-off implicitly assumes every loan carries the same price and the same size. It does
not: a 26% APR loan can support far more risk than a 7% one and still be profitable, and the
break-even PD moves with the rate. So a fixed threshold rejects profitable high-rate business and
accepts unprofitable low-rate business simultaneously. That is the failure EXP03 measures, and it
is a failure of the *decision rule*, not of the risk model -- the PDs feeding all three policies
here are identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    breakeven_pd,
    expected_profit,
)


@dataclass(frozen=True)
class Action:
    """One available decision."""

    name: str
    #: Exposure as a multiple of the amount the borrower requested. 0 means no money moves.
    exposure_multiplier: float
    #: True for the manual-review action, which incurs a cost and refines the PD estimate
    #: before an automatic decision is taken at medium exposure.
    is_review: bool = False


@dataclass(frozen=True)
class ActionSpace:
    """The discrete action set shared by every policy and the simulator.

    Exposure is expressed as a multiple of the requested amount rather than as a dollar figure, so
    the same action set applies to a $2,000 and a $35,000 application. Lending *more* than
    requested (1.5x) is included because it is a real product decision -- a lender that can only
    say yes or no leaves money on the table with its best borrowers.
    """

    actions: tuple[Action, ...] = (
        Action("reject", 0.0),
        Action("review", 1.0, is_review=True),
        Action("approve_small", 0.5),
        Action("approve_medium", 1.0),
        Action("approve_large", 1.5),
    )
    #: Hard cap on any single exposure, in dollars. A concentration limit, as any real lender has.
    max_exposure: float = 40_000.0

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.actions)

    @property
    def multipliers(self) -> np.ndarray:
        return np.array([a.exposure_multiplier for a in self.actions], dtype=float)

    @property
    def review_index(self) -> int | None:
        for i, a in enumerate(self.actions):
            if a.is_review:
                return i
        return None

    @property
    def reject_index(self) -> int:
        for i, a in enumerate(self.actions):
            if a.exposure_multiplier == 0.0 and not a.is_review:
                return i
        return 0

    def exposures(self, requested: np.ndarray) -> np.ndarray:
        """Dollar exposure for every action, shape ``(n_applicants, n_actions)``."""
        requested = np.asarray(requested, dtype=float)[:, None]
        return np.minimum(requested * self.multipliers[None, :], self.max_exposure)

    def without_review(self) -> "ActionSpace":
        """The action set with manual review removed.

        Used for the headline policy comparisons (EXP03, EXP06). The value of a manual review
        depends entirely on how much a human underwriter learns, which is an *assumption* rather
        than anything measurable in this dataset -- see ``docs/ASSUMPTIONS.md``. Leaving review in
        would mean the gap between two policies partly reflects that assumption rather than the
        decision rules being compared. Review is analysed separately, on its own terms.
        """
        return ActionSpace(
            actions=tuple(a for a in self.actions if not a.is_review),
            max_exposure=self.max_exposure,
        )


@dataclass
class DecisionContext:
    """Everything a policy is allowed to see when deciding.

    Deliberately explicit. If a quantity is not on this object, a policy cannot use it, which makes
    "what does this policy know?" answerable by reading one dataclass rather than tracing code.
    """

    pd_estimate: np.ndarray  # calibrated probability of default, one per applicant
    requested_amount: np.ndarray
    apr: np.ndarray  # annual rate as a fraction, e.g. 0.134
    features: np.ndarray | None = None  # optional context vector for learning policies
    #: Remaining capital in the funding period. ``None`` means an unconstrained balance sheet.
    remaining_budget: float | None = None
    #: Applications left in the period, used by budget-aware policies to price opportunity cost.
    remaining_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pd_estimate)


class Policy(ABC):
    """Base class. ``act`` returns one action index per applicant."""

    name: str = "policy"
    #: Learning policies override this; static ones ignore ``update`` entirely.
    is_learning: bool = False

    @abstractmethod
    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        ...

    def update(self, context: DecisionContext, actions: np.ndarray, rewards: np.ndarray) -> None:
        """Observe outcomes. A no-op for static policies."""
        return None

    def reset(self) -> None:
        """Clear any per-episode state."""
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__, "is_learning": self.is_learning}


class FixedThresholdPolicy(Policy):
    """Approve at a fixed exposure whenever PD is below ``threshold``.

    The strawman, and an honest one -- this is what a great many deployed models amount to. Its
    weakness is structural rather than a matter of tuning: one number cannot express a cut-off that
    should depend on price.
    """

    is_learning = False

    def __init__(
        self,
        threshold: float = 0.15,
        action_space: ActionSpace | None = None,
        approve_action: str = "approve_medium",
    ) -> None:
        self.threshold = float(threshold)
        self.action_space = action_space or ActionSpace()
        self.approve_index = self.action_space.names.index(approve_action)
        self.name = f"fixed_threshold@{threshold:.3f}"

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        approve = np.asarray(context.pd_estimate) < self.threshold
        return np.where(approve, self.approve_index, self.action_space.reject_index)

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"threshold": self.threshold}


class RiskBandPolicy(Policy):
    """Map PD bands to exposure, the way a written credit policy does.

    More expressive than a single threshold because it varies exposure with risk -- lend less to
    riskier borrowers rather than refusing them outright. Still blind to price, which is the gap
    :class:`ExpectedProfitPolicy` closes.
    """

    is_learning = False

    def __init__(
        self,
        band_edges: tuple[float, ...] = (0.05, 0.10, 0.18, 0.28),
        band_actions: tuple[str, ...] = (
            "approve_large",
            "approve_medium",
            "approve_small",
            "approve_small",
            "reject",
        ),
        action_space: ActionSpace | None = None,
    ) -> None:
        if len(band_actions) != len(band_edges) + 1:
            raise ValueError("band_actions must have exactly one more entry than band_edges")
        self.band_edges = np.asarray(band_edges, dtype=float)
        self.action_space = action_space or ActionSpace()
        self.band_indices = np.array(
            [self.action_space.names.index(a) for a in band_actions], dtype=int
        )
        self.name = "risk_band"

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        band = np.digitize(np.asarray(context.pd_estimate), self.band_edges, right=False)
        return self.band_indices[band]

    def describe(self) -> dict[str, Any]:
        return super().describe() | {"band_edges": self.band_edges.tolist()}


class ExpectedProfitPolicy(Policy):
    """Choose the action with the highest expected profit for each applicant.

    The economically correct decision under an unconstrained balance sheet, and therefore the
    baseline the RL policy has to beat. It is important to make this baseline strong: an RL agent
    that only beats a badly-tuned threshold rule has demonstrated nothing.

    ``min_profit`` exists because expected profit near zero is dominated by estimation error in PD.
    Requiring a positive margin before committing capital is what a risk appetite *is*.
    """

    is_learning = False

    def __init__(
        self,
        cashflow_model: CashflowModel,
        economics: EconomicsConfig | None = None,
        action_space: ActionSpace | None = None,
        min_profit: float = 0.0,
        #: Multiplier on expected profit per dollar of capital. Set above zero to make the policy
        #: budget-aware analytically -- this is the Lagrange multiplier on a capital constraint,
        #: and is what the learned value function in the RL layer recovers empirically.
        capital_shadow_price: float = 0.0,
    ) -> None:
        self.cashflow_model = cashflow_model
        self.economics = economics or EconomicsConfig()
        self.action_space = action_space or ActionSpace()
        self.min_profit = float(min_profit)
        self.capital_shadow_price = float(capital_shadow_price)
        self.name = "expected_profit"

    def action_values(self, context: DecisionContext) -> np.ndarray:
        """Expected profit for every (applicant, action) pair, shape ``(n, n_actions)``."""
        exposures = self.action_space.exposures(context.requested_amount)
        n, n_actions = exposures.shape
        pd_hat = np.asarray(context.pd_estimate, dtype=float)[:, None]
        apr = np.asarray(context.apr, dtype=float)[:, None]

        values = expected_profit(
            np.broadcast_to(pd_hat, (n, n_actions)),
            exposures,
            np.broadcast_to(apr, (n, n_actions)),
            self.cashflow_model,
            self.economics,
        ).astype(float)

        # Rejecting is free and earns nothing. Overwrite rather than compute, because the
        # fixed opex term would otherwise make "reject" look like a loss.
        values[:, self.action_space.reject_index] = 0.0

        review_idx = self.action_space.review_index
        if review_idx is not None:
            # Review costs money now in exchange for a better estimate later. This policy
            # values it as "medium exposure, minus the fee" and therefore **systematically
            # under-values it**: properly pricing an option to decline after learning requires
            # modelling how much the reviewer learns, which is an assumption, not a measurement.
            # This is why the headline comparisons use `ActionSpace.without_review()`.
            medium = self.action_space.names.index("approve_medium")
            values[:, review_idx] = values[:, medium] - self.economics.manual_review_cost

        if self.capital_shadow_price > 0:
            # Charge each action for the capital it consumes. With lambda > 0 this becomes the
            # constrained-optimal rule: approve only if profit per dollar exceeds lambda.
            values = values - self.capital_shadow_price * exposures

        return values

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        values = self.action_values(context)
        best = np.argmax(values, axis=1)
        best_value = values[np.arange(len(best)), best]
        # If nothing clears the margin, decline.
        return np.where(best_value > self.min_profit, best, self.action_space.reject_index)

    def breakeven(self, context: DecisionContext) -> np.ndarray:
        """The PD at which this applicant's medium-exposure loan breaks even."""
        medium = self.action_space.names.index("approve_medium")
        exposures = self.action_space.exposures(context.requested_amount)[:, medium]
        return breakeven_pd(context.apr, exposures, self.cashflow_model, self.economics)

    def describe(self) -> dict[str, Any]:
        return super().describe() | {
            "min_profit": self.min_profit,
            "capital_shadow_price": self.capital_shadow_price,
            "economics": self.economics.to_dict(),
            "cashflow_model": self.cashflow_model.to_dict(),
        }


class ApproveAllPolicy(Policy):
    """Approve everything at medium exposure. The lower bound, included so results have a floor."""

    is_learning = False

    def __init__(self, action_space: ActionSpace | None = None) -> None:
        self.action_space = action_space or ActionSpace()
        self.index = self.action_space.names.index("approve_medium")
        self.name = "approve_all"

    def act(self, context: DecisionContext, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.full(len(context), self.index, dtype=int)
