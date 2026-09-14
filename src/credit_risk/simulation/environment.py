"""The lending simulator.

The counterfactual problem, stated honestly
-------------------------------------------
We observe each borrower's outcome at exactly one exposure -- the amount LendingClub actually
funded. To evaluate a policy that would have lent a *different* amount, something has to be
assumed. Pretending otherwise is the central dishonesty in most "RL for lending" write-ups, so the
assumption is made explicit here and swept in sensitivity analysis.

**Conditional coupling.** Each applicant is assigned one uniform draw ``u`` that is held fixed
across every action, and the borrower defaults under action ``a`` iff ``u < PD(a)``. The draw is
sampled *conditional on what actually happened* at the observed exposure:

.. code-block:: text

    if the borrower actually defaulted:      u ~ Uniform(0, PD_observed)
    if the borrower actually repaid:         u ~ Uniform(PD_observed, 1)

This has two properties that matter. At the observed exposure the simulator **reproduces the real
outcome exactly** -- no borrower who repaid is made to default by the simulation. And because ``u``
is shared across actions, comparing two policies on the same applicant is a comparison of
decisions, not of random noise: it is common random numbers, which is what makes the paired
comparison in ``evaluation.py`` far tighter than independent sampling would.

What is still assumed
---------------------
1. That lending more raises PD only through the applicant's own affordability features
   (``loan_to_income``, ``dti_post_loan``), re-scored through the same PD model. Real demand
   effects and adverse selection are not modelled.
2. That realised return per dollar is roughly invariant to exposure, so the observed return can be
   scaled. Reasonable for moderate changes, and the reason the action set tops out at 1.5x rather
   than 5x.
3. That the applicant pool is exogenous -- rejecting someone does not change who applies next
   month. Over a single funding period this is fine; over years it is not.

None of this makes the simulator a truth machine. It makes it a *consistent* testbed in which
policies can be compared to each other, which is a much weaker and much more defensible claim, and
it is the claim made in ``docs/RL_FORMULATION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.decision.expected_loss import (
    CashflowModel,
    EconomicsConfig,
    breakeven_pd,
    funding_cost,
    operating_cost,
)
from credit_risk.decision.policies import ActionSpace, DecisionContext, Policy
from credit_risk.utils.runtime import get_logger

LOG = get_logger("simulation.environment")


@dataclass
class ApplicantPool:
    """The population a policy draws from. All arrays are aligned and the same length."""

    pd_estimate: np.ndarray  # calibrated PD at the *requested* exposure
    requested_amount: np.ndarray
    apr: np.ndarray  # fraction, not percent
    observed_default: np.ndarray  # realised outcome, 0/1
    observed_return: np.ndarray  # realised profit per dollar lent
    observed_exposure: np.ndarray  # the amount actually funded
    #: PD re-scored at each action's exposure, shape ``(n, n_actions)``. Supplied by
    #: ``build_applicant_pool``; if absent, PD is treated as exposure-invariant.
    pd_by_action: np.ndarray | None = None
    features: np.ndarray | None = None
    index: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.pd_estimate)

    def subset(self, idx: np.ndarray) -> ApplicantPool:
        return ApplicantPool(
            pd_estimate=self.pd_estimate[idx],
            requested_amount=self.requested_amount[idx],
            apr=self.apr[idx],
            observed_default=self.observed_default[idx],
            observed_return=self.observed_return[idx],
            observed_exposure=self.observed_exposure[idx],
            pd_by_action=None if self.pd_by_action is None else self.pd_by_action[idx],
            features=None if self.features is None else self.features[idx],
            index=None if self.index is None else self.index[idx],
        )


@dataclass
class SimulationConfig:
    """Episode structure and the assumptions that cannot be measured."""

    #: Applications processed in one funding period.
    n_steps: int = 2000
    #: Capital available for the period. ``None`` means unconstrained -- the condition under which
    #: the myopic expected-profit rule is provably optimal and RL should show *no* gain.
    capital_budget: float | None = None
    #: How much a manual review shrinks the PD estimate toward the truth. Purely an assumption
    #: about how good human underwriters are -- nothing in the data measures it -- so it is kept
    #: deliberately modest and is swept in sensitivity analysis. Set too high, review becomes an
    #: oracle and any policy that uses it wins for the wrong reason.
    review_information_gain: float = 0.15
    #: Noise on the reviewer's signal. Large, because a human reading a file learns something
    #: about the borrower, not the outcome.
    review_signal_noise: float = 0.45
    #: Elasticity of PD to exposure when the re-scored PD is unavailable: a 1x increase in
    #: exposure multiplies the odds of default by this factor.
    pd_exposure_elasticity: float = 0.15
    seed: int = 0


@dataclass
class EpisodeResult:
    """Everything one episode produced, in a form the evaluator can aggregate."""

    total_reward: float
    n_applications: int
    n_approved: int
    n_reviewed: int
    n_defaulted: int
    capital_deployed: float
    revenue: float
    losses: float
    costs: float
    approval_rate: float
    default_rate_of_book: float
    return_on_capital: float
    budget_exhausted_at: int | None
    action_counts: dict[str, int] = field(default_factory=dict)
    per_step: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("per_step", "action_counts")
        }
        out.update({f"action_{k}": v for k, v in self.action_counts.items()})
        return out


class LendingEnvironment:
    """A finite-horizon lending episode.

    One step is one application. The state carries the applicant *and* the portfolio position
    (capital left, applications left), because that is exactly what makes the problem sequential
    rather than a sequence of independent decisions -- see ``docs/RL_FORMULATION.md``.
    """

    def __init__(
        self,
        pool: ApplicantPool,
        cashflow_model: CashflowModel,
        economics: EconomicsConfig | None = None,
        action_space: ActionSpace | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.pool = pool
        self.cashflow_model = cashflow_model
        self.economics = economics or EconomicsConfig()
        self.action_space = action_space or ActionSpace()
        self.config = config or SimulationConfig()

        self._rng: np.random.Generator | None = None
        self._order: np.ndarray | None = None
        self._u: np.ndarray | None = None
        self._step = 0
        self._budget_left = float("inf")
        self._budget_exhausted_at: int | None = None
        self._log: list[dict[str, Any]] = []

    # -- episode lifecycle -------------------------------------------------
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        self._rng = np.random.default_rng(seed)
        n_available = len(self.pool)
        n_steps = min(self.config.n_steps, n_available)
        self._order = self._rng.choice(n_available, size=n_steps, replace=False)

        # Conditional coupling: one uniform per applicant, drawn so that the observed outcome is
        # reproduced exactly at the observed exposure. See the module docstring.
        pd_obs = np.clip(self.pool.pd_estimate[self._order], 1e-6, 1 - 1e-6)
        defaulted = self.pool.observed_default[self._order] > 0.5
        draws = self._rng.random(n_steps)
        self._u = np.where(
            defaulted,
            draws * pd_obs,  # uniform on (0, PD)  -> always defaults at PD_observed
            pd_obs + draws * (1.0 - pd_obs),  # uniform on (PD, 1) -> never defaults there
        )

        self._step = 0
        self._budget_left = (
            float("inf")
            if self.config.capital_budget is None
            else float(self.config.capital_budget)
        )
        self._budget_exhausted_at = None
        self._log = []
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        assert self._order is not None
        if self._step >= len(self._order):
            return {"done": True}
        i = int(self._order[self._step])
        return {
            "done": False,
            "applicant": i,
            "pd_estimate": float(self.pool.pd_estimate[i]),
            "requested_amount": float(self.pool.requested_amount[i]),
            "apr": float(self.pool.apr[i]),
            "remaining_budget": self._budget_left,
            "remaining_steps": len(self._order) - self._step,
        }

    def context(self, applicant: int) -> DecisionContext:
        """Wrap one applicant as a batch of size one, so policies see a uniform interface."""
        return DecisionContext(
            pd_estimate=np.array([self.pool.pd_estimate[applicant]]),
            requested_amount=np.array([self.pool.requested_amount[applicant]]),
            apr=np.array([self.pool.apr[applicant]]),
            features=(
                None if self.pool.features is None else self.pool.features[applicant][None, :]
            ),
            remaining_budget=None if np.isinf(self._budget_left) else self._budget_left,
            remaining_steps=None if self._order is None else len(self._order) - self._step,
        )

    # -- the transition ----------------------------------------------------
    def step(self, action_index: int) -> tuple[float, bool, dict[str, Any]]:
        assert self._order is not None and self._u is not None and self._rng is not None
        if self._step >= len(self._order):
            raise RuntimeError("step called on a finished episode; call reset first")

        applicant = int(self._order[self._step])
        action = self.action_space.actions[action_index]
        u = float(self._u[self._step])

        exposure = min(
            float(self.pool.requested_amount[applicant]) * action.exposure_multiplier,
            self.action_space.max_exposure,
        )
        apr = float(self.pool.apr[applicant])
        pd_here = self._pd_at(applicant, exposure)
        review_cost = 0.0
        declined_for_capital = False

        # -- manual review: pay for a better estimate, then decide automatically -------
        if action.is_review:
            review_cost = self.economics.manual_review_cost
            refined = self._refined_pd(applicant, pd_here, u)
            threshold = float(
                breakeven_pd(apr, exposure, self.cashflow_model, self.economics)
            )
            if refined >= threshold:
                exposure = 0.0
            pd_here = refined

        # -- capital constraint --------------------------------------------------------
        if exposure > 0 and exposure > self._budget_left:
            exposure = 0.0
            declined_for_capital = True
            if self._budget_exhausted_at is None:
                self._budget_exhausted_at = self._step

        # -- realise the outcome -------------------------------------------------------
        if exposure <= 0:
            reward = -review_cost
            defaulted = False
            revenue = loss = 0.0
            cost = review_cost
        else:
            defaulted = u < pd_here
            per_dollar = self._return_per_dollar(applicant, defaulted, apr, exposure)
            gross = exposure * per_dollar
            cost = (
                float(funding_cost(exposure, self.economics, apr))
                + float(operating_cost(exposure, self.economics))
                + review_cost
            )
            reward = gross - cost
            revenue = max(gross, 0.0)
            loss = max(-gross, 0.0)
            self._budget_left -= exposure

        self._log.append(
            {
                "step": self._step,
                "applicant": applicant,
                "action": action.name,
                "exposure": exposure,
                "pd_estimate": pd_here,
                "apr": apr,
                "defaulted": bool(defaulted),
                "reward": reward,
                "revenue": revenue,
                "loss": loss,
                "cost": cost,
                "declined_for_capital": declined_for_capital,
                "remaining_budget": self._budget_left,
            }
        )

        self._step += 1
        done = self._step >= len(self._order)
        return reward, done, self._observation()

    # -- outcome model -----------------------------------------------------
    def _pd_at(self, applicant: int, exposure: float) -> float:
        """PD at a counterfactual exposure.

        Uses the model re-scored at each action's exposure when available. Otherwise falls back to
        an odds-ratio adjustment: lending twice as much multiplies the odds of default by
        ``(1 + elasticity)``. The fallback exists so the simulator still runs when re-scoring was
        not precomputed, and it is flagged in the manifest when used.
        """
        base = float(self.pool.pd_estimate[applicant])
        if exposure <= 0:
            return base
        if self.pool.pd_by_action is not None:
            requested = float(self.pool.requested_amount[applicant])
            multipliers = self.action_space.multipliers
            target = exposure / max(requested, 1e-9)
            slot = int(np.argmin(np.abs(multipliers - target)))
            value = float(self.pool.pd_by_action[applicant, slot])
            if np.isfinite(value):
                return float(np.clip(value, 1e-6, 1 - 1e-6))

        observed = max(float(self.pool.observed_exposure[applicant]), 1e-9)
        ratio = exposure / observed
        odds = base / (1.0 - base)
        odds *= 1.0 + self.config.pd_exposure_elasticity * (ratio - 1.0)
        odds = max(odds, 1e-9)
        return float(np.clip(odds / (1.0 + odds), 1e-6, 1 - 1e-6))

    def _refined_pd(self, applicant: int, pd_here: float, u: float) -> float:
        """PD after a human review.

        Modelled as shrinking the estimate toward the outcome the coupling has already fixed --
        a reviewer sees genuine additional evidence, but imperfectly. ``review_information_gain``
        is the fraction of the remaining uncertainty resolved, and is an assumption, not a
        measurement.
        """
        assert self._rng is not None
        w = float(np.clip(self.config.review_information_gain, 0.0, 1.0))
        latent_outcome = 1.0 if u < pd_here else 0.0
        noisy = np.clip(
            latent_outcome + self._rng.normal(0.0, self.config.review_signal_noise), 0.0, 1.0
        )
        return float(np.clip((1.0 - w) * pd_here + w * noisy, 1e-6, 1 - 1e-6))

    def _return_per_dollar(
        self, applicant: int, defaulted: bool, apr: float, exposure: float
    ) -> float:
        """Profit per dollar lent.

        Where the simulated outcome matches what actually happened, the **observed** realised
        return is used -- a measured quantity. Where the coupling has flipped the outcome (because
        a different exposure moved PD across ``u``), there is nothing observed to use, so the
        fitted cashflow model supplies the counterfactual. Preferring the observation wherever one
        exists is what keeps the simulator anchored to real cashflows.
        """
        actually_defaulted = bool(self.pool.observed_default[applicant] > 0.5)
        if defaulted == actually_defaulted:
            observed = float(self.pool.observed_return[applicant])
            if np.isfinite(observed):
                return observed
        if defaulted:
            return -float(self.cashflow_model.bad_loss_rate(apr))
        return float(self.cashflow_model.good_yield(apr))

    # -- running a whole episode ------------------------------------------
    def run(self, policy: Policy, seed: int | None = None, keep_log: bool = False) -> EpisodeResult:
        """Run one full episode under ``policy`` and summarise it."""
        obs = self.reset(seed)
        policy.reset()
        rng = np.random.default_rng((seed or 0) + 9973)

        while not obs["done"]:
            applicant = obs["applicant"]
            context = self.context(applicant)
            action = int(np.asarray(policy.act(context, rng)).ravel()[0])
            reward, done, obs = self.step(action)
            if policy.is_learning:
                policy.update(context, np.array([action]), np.array([reward]))

        return self._summarise(keep_log)

    def _summarise(self, keep_log: bool) -> EpisodeResult:
        log = pd.DataFrame(self._log)
        approved = log["exposure"] > 0
        deployed = float(log.loc[approved, "exposure"].sum())
        total_reward = float(log["reward"].sum())

        return EpisodeResult(
            total_reward=total_reward,
            n_applications=int(len(log)),
            n_approved=int(approved.sum()),
            n_reviewed=int((log["action"] == "review").sum()),
            n_defaulted=int(log.loc[approved, "defaulted"].sum()),
            capital_deployed=deployed,
            revenue=float(log["revenue"].sum()),
            losses=float(log["loss"].sum()),
            costs=float(log["cost"].sum()),
            approval_rate=float(approved.mean()) if len(log) else 0.0,
            default_rate_of_book=(
                float(log.loc[approved, "defaulted"].mean()) if approved.any() else 0.0
            ),
            return_on_capital=(total_reward / deployed) if deployed > 0 else 0.0,
            budget_exhausted_at=self._budget_exhausted_at,
            action_counts=log["action"].value_counts().to_dict(),
            per_step=log if keep_log else None,
        )


def build_applicant_pool(
    df: pd.DataFrame,
    pd_estimate: np.ndarray,
    cashflow_model: CashflowModel,
    pd_by_action: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> ApplicantPool:
    """Assemble a pool from a scored slice of the loan table."""
    from credit_risk.decision.expected_loss import realised_return

    requested = pd.to_numeric(df["loan_amnt"], errors="coerce").to_numpy(dtype=float)
    funded = pd.to_numeric(df["funded_amnt"], errors="coerce").to_numpy(dtype=float)
    apr = pd.to_numeric(df["int_rate"], errors="coerce").to_numpy(dtype=float) / 100.0
    outcome = pd.to_numeric(df["default"], errors="coerce").to_numpy(dtype=float)
    returns = realised_return(df).to_numpy(dtype=float)

    keep = np.isfinite(requested) & np.isfinite(apr) & np.isfinite(outcome)
    if not keep.all():
        LOG.warning("dropping %d applicants with unusable economics", int((~keep).sum()))

    pool = ApplicantPool(
        pd_estimate=np.clip(np.asarray(pd_estimate, dtype=float), 1e-6, 1 - 1e-6),
        requested_amount=np.nan_to_num(requested, nan=0.0),
        apr=np.nan_to_num(apr, nan=0.13),
        observed_default=np.nan_to_num(outcome, nan=0.0),
        observed_return=returns,
        observed_exposure=np.where(np.isfinite(funded) & (funded > 0), funded, requested),
        pd_by_action=pd_by_action,
        features=features,
        index=df.index.to_numpy(),
    )
    return pool.subset(np.flatnonzero(keep))
