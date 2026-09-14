"""Turning SHAP values into something a person can be told.

A credit decision that affects someone has to come with a reason, and in several jurisdictions
that is a legal requirement rather than a nicety -- US Regulation B obliges a lender to give
specific principal reasons for an adverse action. "The gradient-boosted model output 0.31" is not
one of those.

The pipeline is deliberately mechanical, because the failure mode to avoid is *inventing* a
plausible-sounding explanation:

.. code-block:: text

    SHAP value for this applicant  ->  human-readable phrase  ->  ranked reason list

Every phrase is derived from a SHAP contribution that was actually computed for this specific
applicant. Nothing is generated from the prediction alone, nothing is paraphrased from what
"usually" drives risk, and a feature with no attribution produces no sentence.

The honest caveats, which belong in the interview answer as much as in the code:

* SHAP explains **the model**, not the borrower and not the world. It attributes the model's
  output to its inputs. If the model has learned a proxy, SHAP will faithfully report the proxy.
* Contributions are **local**: they describe this prediction relative to the dataset baseline,
  and do not generalise into "reducing utilisation by 10 points would lower your PD by X".
  That is a causal claim, and nothing here supports causal claims.
* For correlated features, credit is shared between them in a way that depends on the background
  sample. Two runs with different backgrounds can rank two collinear features differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Human-readable phrasing per feature: (increases-risk phrase, decreases-risk phrase).
#: Only features listed here can produce a sentence. That is the point -- an unmapped feature is
#: reported by its raw name rather than given invented prose.
REASON_PHRASES: dict[str, tuple[str, str]] = {
    "fico_mid": ("Low credit bureau score", "Strong credit bureau score"),
    "fico_range_low": ("Low credit bureau score", "Strong credit bureau score"),
    "fico_range_high": ("Low credit bureau score", "Strong credit bureau score"),
    "dti": ("High debt-to-income ratio", "Low debt-to-income ratio"),
    "initial_list_status": ("Listing channel carries higher observed risk",
                            "Listing channel carries lower observed risk"),
    "installment": ("High monthly instalment", "Low monthly instalment"),
    "term_months": ("Longer loan term", "Shorter loan term"),
    "emp_length_years": ("Short employment history", "Long employment history"),
    "total_rev_hi_lim": ("Low total revolving credit limit", "High total revolving credit limit"),
    "tot_cur_bal": ("High total current balances", "Low total current balances"),
    "tot_coll_amt": ("Amounts in collection", "Nothing in collection"),
    "acc_now_delinq": ("Accounts currently delinquent", "No accounts currently delinquent"),
    "delinq_amnt": ("Delinquent balance outstanding", "No delinquent balance"),
    "tax_liens": ("Tax liens on file", "No tax liens"),
    "pub_rec_bankruptcies": ("Bankruptcy on file", "No bankruptcy on file"),
    "collections_12_mths_ex_med": ("Recent collections activity", "No recent collections"),
    "chargeoff_within_12_mths": ("Charge-off in the last twelve months",
                                 "No charge-off in the last twelve months"),
    "mths_since_last_record": ("Recent public record", "No recent public record"),
    "mths_since_last_major_derog": ("Recent major derogatory mark",
                                    "No recent major derogatory mark"),
    "log_annual_inc": ("Low stated income", "Strong stated income"),
    "dti_clean": ("High debt-to-income ratio", "Low debt-to-income ratio"),
    "dti_post_loan": ("High debt burden after this loan", "Manageable debt burden after this loan"),
    "loan_to_income": ("Loan is large relative to income", "Loan is modest relative to income"),
    "annual_inc": ("Low stated income", "Strong stated income"),
    "revol_util_clipped": ("High revolving credit utilisation", "Low revolving credit utilisation"),
    "revol_util": ("High revolving credit utilisation", "Low revolving credit utilisation"),
    "revol_bal_to_income": ("High revolving balances relative to income", "Low revolving balances"),
    "credit_history_months": ("Short credit history", "Long, established credit history"),
    "inq_last_6mths_capped": ("Several recent credit applications", "Few recent credit applications"),
    "inq_last_6mths": ("Several recent credit applications", "Few recent credit applications"),
    "delinq_2yrs": ("Recent delinquencies on file", "No recent delinquencies"),
    "has_delinq_2yrs": ("Delinquency in the last two years", "No delinquency in the last two years"),
    "mths_since_last_delinq": ("Recent delinquency", "No recent delinquency"),
    "pub_rec": ("Public records on file", "No public records"),
    "has_pub_rec": ("Public record on file", "No public records"),
    "has_bankruptcy": ("Bankruptcy on file", "No bankruptcy on file"),
    "has_derogatory": ("Derogatory marks on file", "No derogatory marks"),
    "has_collections": ("Recent collections activity", "No recent collections"),
    "open_acc": ("Many open credit lines", "Few open credit lines"),
    "open_acc_ratio": ("Most accounts currently open", "Few accounts currently open"),
    "total_acc": ("Limited number of credit accounts", "Broad credit file"),
    "loan_amnt": ("Large requested amount", "Modest requested amount"),
    "term": ("Longer loan term", "Shorter loan term"),
    "purpose": ("Loan purpose is higher risk", "Loan purpose is lower risk"),
    "home_ownership": ("Housing status is higher risk", "Housing status is lower risk"),
    "verification_status": ("Income not verified", "Income verified"),
    "addr_state": ("Location carries higher observed risk", "Location carries lower observed risk"),
    "int_rate": ("Priced at a high interest rate", "Priced at a low interest rate"),
    "grade": ("Weak internal credit grade", "Strong internal credit grade"),
    "sub_grade": ("Weak internal credit grade", "Strong internal credit grade"),
    "cohort_zip3_default_rate": (
        "Recent borrowers in this postal area have performed poorly",
        "Recent borrowers in this postal area have performed well",
    ),
    "cohort_emp_default_rate": (
        "Recent borrowers with similar employment have performed poorly",
        "Recent borrowers with similar employment have performed well",
    ),
    "cohort_zip3_default_lift": (
        "Postal area risk above portfolio average",
        "Postal area risk below portfolio average",
    ),
    "cohort_emp_default_lift": (
        "Employment cohort risk above portfolio average",
        "Employment cohort risk below portfolio average",
    ),
}


@dataclass
class Reason:
    """One contributing factor for one applicant."""

    feature: str
    phrase: str
    contribution: float
    value: Any
    direction: str  # "increases_risk" | "decreases_risk"

    def as_line(self) -> str:
        sign = "+" if self.direction == "increases_risk" else "-"
        return f"{sign} {self.phrase}"


@dataclass
class Explanation:
    """A single decision, explained."""

    pd_estimate: float
    risk_band: str
    decision: str
    exposure: float
    reasons_against: list[Reason] = field(default_factory=list)
    reasons_for: list[Reason] = field(default_factory=list)
    baseline_pd: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Format as the adverse-action-style summary in the project README."""
        lines = [
            "Applicant",
            f"  Risk band : {self.risk_band}",
            f"  PD        : {self.pd_estimate:.1%}",
            "",
            "Decision",
            f"  {self.decision}"
            + (f" -- exposure ${self.exposure:,.0f}" if self.exposure > 0 else ""),
            "",
            "Principal reasons",
        ]
        for reason in self.reasons_against:
            lines.append(f"  {reason.as_line()}")
        for reason in self.reasons_for:
            lines.append(f"  {reason.as_line()}")
        if self.notes:
            lines += ["", "Notes"] + [f"  {n}" for n in self.notes]
        return "\n".join(lines)


def risk_band(pd_estimate: float, edges: tuple[float, ...] = (0.05, 0.10, 0.18, 0.28)) -> str:
    labels = ("Very low", "Low", "Medium", "High", "Very high")
    return labels[int(np.digitize(pd_estimate, edges))]


def _phrase_for(feature: str, contribution: float) -> str:
    """Look up a phrase, resolving the naming conventions the feature builder introduces."""
    key = feature
    if key not in REASON_PHRASES:
        # "mths_since_last_delinq_missing" -> the underlying event.
        if key.endswith("_missing") and key[: -len("_missing")] in REASON_PHRASES:
            base = key[: -len("_missing")]
            increases, decreases = REASON_PHRASES[base]
            # A missing "months since" field means the event never happened, so the sense flips.
            return decreases if contribution > 0 else increases
        # One-hot columns arrive as "purpose=small_business".
        if "=" in key:
            column, _, level = key.partition("=")
            if column in REASON_PHRASES:
                increases, decreases = REASON_PHRASES[column]
                pretty = level.replace("_", " ")
                return f"{increases} ({pretty})" if contribution > 0 else f"{decreases} ({pretty})"
        if key.endswith("_clipped") and key[: -len("_clipped")] in REASON_PHRASES:
            key = key[: -len("_clipped")]
        elif key.endswith("_capped") and key[: -len("_capped")] in REASON_PHRASES:
            key = key[: -len("_capped")]
        elif key.endswith("_clean") and key[: -len("_clean")] in REASON_PHRASES:
            key = key[: -len("_clean")]

    increases, decreases = REASON_PHRASES.get(
        key, (f"{feature} raises modelled risk", f"{feature} lowers modelled risk")
    )
    return increases if contribution > 0 else decreases


def explain_row(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    pd_estimate: float,
    decision: str,
    exposure: float = 0.0,
    top_k: int = 4,
    baseline_pd: float = float("nan"),
    min_contribution: float = 1e-4,
) -> Explanation:
    """Build an explanation for one applicant from that applicant's SHAP row.

    ``top_k`` factors in each direction. Regulation B expects a small number of *principal*
    reasons, not a ranked list of forty; and a reason list long enough to include every marginal
    contribution is one nobody reads.
    """
    contributions = np.asarray(shap_values, dtype=float).ravel()
    order = np.argsort(-np.abs(contributions))

    against: list[Reason] = []
    supporting: list[Reason] = []
    seen: set[str] = set()
    for idx in order:
        contribution = float(contributions[idx])
        if abs(contribution) < min_contribution:
            continue
        name = feature_names[idx]
        phrase = _phrase_for(name, contribution)
        if phrase in seen:
            # Collinear features share a phrase; report the strongest one only.
            continue
        seen.add(phrase)
        reason = Reason(
            feature=name,
            phrase=phrase,
            contribution=contribution,
            value=feature_values.get(name, None) if hasattr(feature_values, "get") else None,
            direction="increases_risk" if contribution > 0 else "decreases_risk",
        )
        if contribution > 0 and len(against) < top_k:
            against.append(reason)
        elif contribution < 0 and len(supporting) < top_k:
            supporting.append(reason)
        if len(against) >= top_k and len(supporting) >= top_k:
            break

    return Explanation(
        pd_estimate=float(pd_estimate),
        risk_band=risk_band(float(pd_estimate)),
        decision=decision,
        exposure=float(exposure),
        reasons_against=against,
        reasons_for=supporting,
        baseline_pd=baseline_pd,
    )


def explanation_table(explanations: list[Explanation]) -> pd.DataFrame:
    """Flatten explanations for review or audit sampling."""
    rows = []
    for i, exp in enumerate(explanations):
        rows.append(
            {
                "applicant": i,
                "pd": exp.pd_estimate,
                "band": exp.risk_band,
                "decision": exp.decision,
                "exposure": exp.exposure,
                "top_adverse": "; ".join(r.phrase for r in exp.reasons_against[:3]),
                "top_supporting": "; ".join(r.phrase for r in exp.reasons_for[:3]),
            }
        )
    return pd.DataFrame(rows)
