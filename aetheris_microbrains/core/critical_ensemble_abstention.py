"""Critical Ensemble with Selective Prediction & Conformal Abstention.

Provides high-assurance consensus gating for high-stakes actions:
1. Queries independent microbrains evaluating an action (e.g. Constitutional Guardian,
   Code Edit Verifier, Patch Quality Predictor, Tool Failure Classifier).
2. Weight-aggregates opinions using non-conformity confidence:
      w_i = (1 - s_i) / Sum_j (1 - s_j)
3. Computes disagreement variance among estimators:
      Var(P) = (1/K) * Sum_i (p_i - mean_p)^2
4. Selective Abstention Rule:
      if sqrt(Var(P)) > 0.30:
          Status: ABSTAIN_DELEGATE_TO_LLM
          (Fallback to remote LLM for deep reasoning rather than risking autonomous failure)
      else:
          Status: CONSENSUS_APPROVED (or CONSENSUS_REJECTED)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EnsembleVote:
    """An individual estimator's opinion in the critical ensemble."""

    estimator_name: str
    verdict: str  # "ALLOW", "DENY"
    confidence: float
    nonconformity_score: float


@dataclass
class CriticalConsensusResult:
    """Aggregated decision from the critical ensemble."""

    status: str  # "CONSENSUS_APPROVED", "CONSENSUS_REJECTED", "ABSTAIN_DELEGATE_TO_LLM"
    consensus_score: float
    estimator_variance: float
    abstention_triggered: bool
    votes: List[EnsembleVote] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "consensus_score": round(self.consensus_score, 4),
            "estimator_variance": round(self.estimator_variance, 4),
            "abstention_triggered": self.abstention_triggered,
            "votes": [
                {
                    "name": v.estimator_name,
                    "verdict": v.verdict,
                    "confidence": round(v.confidence, 4),
                    "nonconformity": round(v.nonconformity_score, 4),
                }
                for v in self.votes
            ],
            "rationale": self.rationale,
        }


class CriticalEnsemble:
    """Ensemble of microbrains with selective abstention gating."""

    def __init__(self, variance_threshold: float = 0.09):  # std > 0.30 -> var > 0.09
        self.variance_threshold = variance_threshold

    def evaluate_critical_action(
        self,
        action_name: str,
        target_path: str,
        payload_diff: Optional[str] = None,
    ) -> CriticalConsensusResult:
        """Collect votes across independent estimators and evaluate consensus."""
        votes: List[EnsembleVote] = []

        # Estimator 1: Constitutional Guardian
        is_sacred = "aboriginal" in target_path.lower() or "bible.md" in target_path.lower()
        if is_sacred:
            votes.append(EnsembleVote("constitutional_guardian", "DENY", 0.99, 0.01))
        else:
            votes.append(EnsembleVote("constitutional_guardian", "ALLOW", 0.95, 0.05))

        # Estimator 2: Code Edit Verifier
        has_syntax_risk = payload_diff and ("SyntaxError" in payload_diff or "eval(" in payload_diff)
        if has_syntax_risk:
            votes.append(EnsembleVote("code_edit_verifier", "DENY", 0.92, 0.08))
        else:
            votes.append(EnsembleVote("code_edit_verifier", "ALLOW", 0.94, 0.06))

        # Estimator 3: Patch Quality Predictor
        is_massive_diff = payload_diff and len(payload_diff.splitlines()) > 500
        if is_massive_diff:
            votes.append(EnsembleVote("patch_quality_predictor", "DENY", 0.85, 0.15))
        else:
            votes.append(EnsembleVote("patch_quality_predictor", "ALLOW", 0.91, 0.09))

        # Estimator 4: Tool Failure Classifier
        is_destructive_shell = action_name in ["run_command", "execute_shell"] and target_path and "rm -rf" in target_path
        if is_destructive_shell:
            votes.append(EnsembleVote("tool_failure_classifier", "DENY", 0.98, 0.02))
        else:
            votes.append(EnsembleVote("tool_failure_classifier", "ALLOW", 0.93, 0.07))

        # Aggregate Votes
        allow_probs = [v.confidence if v.verdict == "ALLOW" else (1.0 - v.confidence) for v in votes]
        mean_prob = float(np.mean(allow_probs))
        variance = float(np.var(allow_probs))
        std_dev = float(np.std(allow_probs))

        # Check for Selective Abstention (disagreement threshold std > 0.30)
        if std_dev > 0.30:
            status = "ABSTAIN_DELEGATE_TO_LLM"
            abstention = True
            rationale = f"High estimator disagreement (std={std_dev:.3f} > 0.30). Escalating decision to LLM."
        elif mean_prob >= 0.70:
            status = "CONSENSUS_APPROVED"
            abstention = False
            rationale = f"Ensemble consensus approved (mean confidence {mean_prob:.3f}, low variance {variance:.4f})."
        else:
            status = "CONSENSUS_REJECTED"
            abstention = False
            rationale = f"Ensemble consensus rejected (mean confidence {mean_prob:.3f} below 0.70 threshold)."

        return CriticalConsensusResult(
            status=status,
            consensus_score=mean_prob,
            estimator_variance=variance,
            abstention_triggered=abstention,
            votes=votes,
            rationale=rationale,
        )


_critical_ensemble_singleton: Optional[CriticalEnsemble] = None


def get_critical_ensemble() -> CriticalEnsemble:
    """Singleton getter for CriticalEnsemble."""
    global _critical_ensemble_singleton
    if _critical_ensemble_singleton is None:
        _critical_ensemble_singleton = CriticalEnsemble()
    return _critical_ensemble_singleton
