"""Critical Ensemble with Selective Prediction & Conformal Abstention.

Provides high-assurance consensus gating for high-stakes actions:
1. Queries 4 independent microbrains evaluating an action:
   - `neuro_constitutional_guardian`: Evaluates sovereignty, sacred sites, and boundary violations.
   - `neuro_code_edit_verifier`: Real AST parse tokenization & structural verification.
   - `neuro_patch_quality_predictor`: Diff scale, blast radius, and regression probability estimation.
   - `neuro_tool_failure_classifier`: Action boundary, permission, and Suki-class failure risk.
2. Weight-aggregates opinions using model-predicted conformal nonconformity bounds.
3. Decisions are strictly determined by neural inference outputs (predicted token, confidence, conformal abstention).
4. Computes empirical disagreement variance among estimators:
      Var(P) = (1/K) * Sum_i (p_i - mean_p)^2,   std = sqrt(Var(P))
5. Selective Abstention & Rejection Rules:
      - Direct Rejection (CONSENSUS_REJECTED): Certified hard invariant violations (sacred sites, malformed AST, destructive commands).
      - Selective Abstention (ABSTAIN_DELEGATE_TO_LLM): Ambiguous prediction set (|C_alpha| != 1), missing models, or soft metric variance > 0.25.
      - Consensus Approved (CONSENSUS_APPROVED): Low variance (std <= 0.25), zero abstentions, mean probability >= 0.75.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR
from .layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    is_protected_path,
)

log = logging.getLogger(__name__)

# Semantic Token Categories for Neural Conformal Gating
GUARDIAN_DENY_TOKENS: Set[str] = {
    "violation_severity_blocked",
    "violation_severity_fatal_quarantine",
    "veto_unconditional",
    "action_repo_write_checked",
    "provenance_signature_invalid",
    "sacred_site_aboriginal_core",
    "sacred_site_bible_md",
    "sacred_site_spirit_of_place",
    "sacred_site_songlines",
    "sacred_site_dreamtime",
    "sacred_site_integrity_seal",
}
GUARDIAN_ALLOW_TOKENS: Set[str] = {
    "violation_severity_none",
    "ancestral_law_honored",
    "boundary_containment_active",
    "provenance_signature_valid",
    "kinship_dreamkeeper_guard",
    "kinship_pathfinder_guidance",
    "kinship_explorer_discovery",
}

VERIFIER_DENY_TOKENS: Set[str] = {
    "verifier_verdict_reject_syntax",
    "verifier_verdict_reject_safety",
    "syntax_tree_malformed",
    "secret_leak_pattern_found",
    "side_effect_subprocess_spawn",
    "sacred_path_target_detected",
}
VERIFIER_ALLOW_TOKENS: Set[str] = {
    "verifier_verdict_accept",
    "conformal_safety_guarantee_met",
    "precondition_enforced",
    "syntax_tree_valid",
    "secret_leak_clean",
    "sacred_path_safe",
    "diff_hunk_clean_insertion",
}

PREDICTOR_DENY_TOKENS: Set[str] = {
    "patch_quality_rejected_high_risk",
    "breaking_change_signature_detected",
    "regression_probability_high",
    "ast_syntax_malformed",
}
PREDICTOR_ALLOW_TOKENS: Set[str] = {
    "patch_quality_grade_a",
    "patch_quality_grade_b",
    "patch_quality_grade_c",
    "patch_quality_certified",
    "backward_compatibility_preserved",
    "regression_probability_low",
    "atomic_hunk_isolated",
    "diff_lines_sub_ten",
}

CLASSIFIER_DENY_TOKENS: Set[str] = {
    "failure_fatal_circuit_breaker",
    "failure_severity_system_critical",
    "suki_class_boundary",
    "suki_class_syntax",
    "suki_class_precondition",
    "suki_class_resource",
    "suki_class_sacred_site_mutation",
}
CLASSIFIER_ALLOW_TOKENS: Set[str] = {
    "action_inspect_stack_trace",
    "failure_severity_benign",
    "failure_recoverable_by_retry",
    "clean_success",
}


@dataclass
class EnsembleVote:
    """An individual estimator's opinion in the critical ensemble."""

    estimator_name: str
    verdict: str  # "ALLOW", "DENY", "ABSTAIN"
    confidence: float
    nonconformity_score: float
    details: Dict[str, Any] = field(default_factory=dict)


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
                    "details": v.details,
                }
                for v in self.votes
            ],
            "rationale": self.rationale,
        }


class CriticalEnsemble:
    """Ensemble of trained microbrains with selective abstention gating."""

    def __init__(self, store_dir: Optional[Path] = None, variance_threshold: float = 0.0625):
        self.store_dir = store_dir or STORE_DIR
        self.variance_threshold = variance_threshold
        self._models_cache: Dict[str, AnalyticalMicroTransformer] = {}

    def _get_model(self, model_name: str) -> Optional[AnalyticalMicroTransformer]:
        """Lazy load and cache microbrain model."""
        if model_name in self._models_cache:
            return self._models_cache[model_name]
        m = AnalyticalMicroTransformer.load(model_name, self.store_dir)
        if m is not None:
            self._models_cache[model_name] = m
        return m

    def evaluate_critical_action(
        self,
        action_name: str,
        target_path: Optional[str] = None,
        payload_diff: Optional[str] = None,
    ) -> CriticalConsensusResult:
        """Collect votes across 4 independent neural estimators via conformal inference."""
        votes: List[EnsembleVote] = []
        path_str = str(target_path or "")

        # =========================================================================
        # Estimator 1: Constitutional Guardian
        # =========================================================================
        guardian = self._get_model("neuro_constitutional_guardian")
        is_sacred = is_protected_path(path_str)
        g_tokens = ["sacred_site_aboriginal_core", "action_repo_write_checked"] if is_sacred else ["boundary_containment_active", "ancestral_law_honored"]

        if guardian is None:
            g_verdict = "ABSTAIN"
            g_conf = 0.50
            g_nonconf = 1.0
            g_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            g_res = guardian.infer_conformal(g_tokens, alpha=0.05)
            g_pred = g_res.get("predicted_token", "")
            g_abstained = bool(g_res.get("abstained", False))
            g_conf = float(g_res.get("confidence", 0.50))
            if g_abstained:
                g_verdict = "ABSTAIN"
            elif g_pred in GUARDIAN_DENY_TOKENS:
                g_verdict = "DENY"
            elif g_pred in GUARDIAN_ALLOW_TOKENS:
                g_verdict = "ALLOW"
            else:
                g_verdict = "ABSTAIN"
            g_nonconf = round(1.0 - g_conf, 4) if g_verdict == "ALLOW" else round(g_conf, 4)
            g_details = {
                "is_sacred": is_sacred,
                "tokens": g_tokens,
                "predicted_token": g_pred,
                "prediction_set": g_res.get("prediction_set", []),
                "abstained": g_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="constitutional_guardian",
            verdict=g_verdict,
            confidence=g_conf,
            nonconformity_score=g_nonconf,
            details=g_details,
        ))

        # =========================================================================
        # Estimator 2: Code Edit Verifier (Layer 1 Language-Aware Structural AST)
        # =========================================================================
        verifier = self._get_model("neuro_code_edit_verifier")
        diff_analysis = analyze_code_diff(path_str, payload_diff)
        syntax_clean = bool(diff_analysis["syntax_valid"])
        security_clean = bool(diff_analysis["security_clean"])

        if not syntax_clean:
            ast_tokens = ["syntax_tree_malformed", "ast_node_try_except"]
        elif not security_clean:
            ast_tokens = ["ast_node_call", "side_effect_subprocess_spawn"]
        else:
            ast_tokens = ["syntax_tree_valid", "ast_node_function_def"]

        if verifier is None:
            v_verdict = "ABSTAIN"
            v_conf = 0.50
            v_nonconf = 1.0
            v_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            v_res = verifier.infer_conformal(ast_tokens, alpha=0.05)
            v_pred = v_res.get("predicted_token", "")
            v_abstained = bool(v_res.get("abstained", False))
            v_conf = float(v_res.get("confidence", 0.50))
            if v_abstained:
                v_verdict = "ABSTAIN"
            elif v_pred in VERIFIER_DENY_TOKENS:
                v_verdict = "DENY"
            elif v_pred in VERIFIER_ALLOW_TOKENS:
                v_verdict = "ALLOW"
            else:
                v_verdict = "ABSTAIN"
            v_nonconf = round(1.0 - v_conf, 4) if v_verdict == "ALLOW" else round(v_conf, 4)
            v_details = {
                "syntax_clean": syntax_clean,
                "security_clean": security_clean,
                "tokens": ast_tokens,
                "predicted_token": v_pred,
                "prediction_set": v_res.get("prediction_set", []),
                "abstained": v_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="code_edit_verifier",
            verdict=v_verdict,
            confidence=v_conf,
            nonconformity_score=v_nonconf,
            details=v_details,
        ))

        # =========================================================================
        # Estimator 3: Patch Quality Predictor
        # =========================================================================
        predictor = self._get_model("neuro_patch_quality_predictor")
        diff_lines = len(payload_diff.splitlines()) if payload_diff else 0

        if diff_lines > 500:
            p_tokens = ["diff_lines_massive_five_hundred", "breaking_change_signature_detected"]
        elif diff_lines > 50:
            p_tokens = ["diff_lines_sub_two_hundred", "backward_compatibility_preserved"]
        else:
            p_tokens = ["diff_lines_sub_ten", "backward_compatibility_preserved"]

        if predictor is None:
            p_verdict = "ABSTAIN"
            p_conf = 0.50
            p_nonconf = 1.0
            p_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            p_res = predictor.infer_conformal(p_tokens, alpha=0.05)
            p_pred = p_res.get("predicted_token", "")
            p_abstained = bool(p_res.get("abstained", False))
            p_conf = float(p_res.get("confidence", 0.50))
            if p_abstained:
                p_verdict = "ABSTAIN"
            elif p_pred in PREDICTOR_DENY_TOKENS:
                p_verdict = "DENY"
            elif p_pred in PREDICTOR_ALLOW_TOKENS:
                p_verdict = "ALLOW"
            else:
                p_verdict = "ABSTAIN"
            p_nonconf = round(1.0 - p_conf, 4) if p_verdict == "ALLOW" else round(p_conf, 4)
            p_details = {
                "diff_lines": diff_lines,
                "tokens": p_tokens,
                "predicted_token": p_pred,
                "prediction_set": p_res.get("prediction_set", []),
                "abstained": p_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="patch_quality_predictor",
            verdict=p_verdict,
            confidence=p_conf,
            nonconformity_score=p_nonconf,
            details=p_details,
        ))

        # =========================================================================
        # Estimator 4: Tool Failure Classifier (Layer 1 Shell Structural Parser)
        # =========================================================================
        classifier = self._get_model("neuro_tool_failure_classifier")
        if action_name in ["run_command", "execute_shell"] and path_str:
            shell_res = analyze_shell_command(path_str)
            is_destructive_shell = not shell_res["is_safe"]
        else:
            is_destructive_shell = False
        tf_tokens = ["suki_class_boundary", "failure_fatal_circuit_breaker"] if is_destructive_shell else ["failure_severity_benign", "action_inspect_stack_trace"]

        if classifier is None:
            tf_verdict = "ABSTAIN"
            tf_conf = 0.50
            tf_nonconf = 1.0
            tf_details = {"model_loaded": False, "reason": "model_missing"}
        else:
            tf_res = classifier.infer_conformal(tf_tokens, alpha=0.05)
            tf_pred = tf_res.get("predicted_token", "")
            tf_abstained = bool(tf_res.get("abstained", False))
            tf_conf = float(tf_res.get("confidence", 0.50))
            if tf_abstained:
                tf_verdict = "ABSTAIN"
            elif tf_pred in CLASSIFIER_DENY_TOKENS:
                tf_verdict = "DENY"
            elif tf_pred in CLASSIFIER_ALLOW_TOKENS:
                tf_verdict = "ALLOW"
            else:
                tf_verdict = "ABSTAIN"
            tf_nonconf = round(1.0 - tf_conf, 4) if tf_verdict == "ALLOW" else round(tf_conf, 4)
            tf_details = {
                "is_destructive": is_destructive_shell,
                "tokens": tf_tokens,
                "predicted_token": tf_pred,
                "prediction_set": tf_res.get("prediction_set", []),
                "abstained": tf_abstained,
            }

        votes.append(EnsembleVote(
            estimator_name="tool_failure_classifier",
            verdict=tf_verdict,
            confidence=tf_conf,
            nonconformity_score=tf_nonconf,
            details=tf_details,
        ))

        # =========================================================================
        # Aggregate Votes, Measure Estimator Variance, and Decide
        # =========================================================================
        allow_probs = [
            v.confidence if v.verdict == "ALLOW" else (1.0 - v.confidence if v.verdict == "DENY" else 0.50)
            for v in votes
        ]
        mean_prob = float(np.mean(allow_probs))
        variance = float(np.var(allow_probs))
        std_dev = float(np.std(allow_probs))

        deny_votes = [v for v in votes if v.verdict == "DENY"]
        abstain_votes = [v for v in votes if v.verdict == "ABSTAIN"]
        allow_votes = [v for v in votes if v.verdict == "ALLOW"]

        # Hard invariant gate: certified rejection by constitutional, syntax, or destructive boundary models
        hard_veto_estimators = {"constitutional_guardian", "code_edit_verifier", "tool_failure_classifier"}
        hard_veto = any(v.estimator_name in hard_veto_estimators and v.confidence >= 0.85 for v in deny_votes)

        if hard_veto or len(deny_votes) >= 2 or mean_prob < 0.40:
            status = "CONSENSUS_REJECTED"
            abstention = False
            rationale = (
                f"Consensus rejected: Certified invariant violation detected "
                f"({len(deny_votes)} DENY votes: {[v.estimator_name for v in deny_votes]}, mean_prob={mean_prob:.3f}). Direct block."
            )
        elif len(deny_votes) == 0 and len(abstain_votes) == 0 and std_dev <= 0.25 and mean_prob >= 0.75:
            # Proposal 5: Strict unanimous consensus required for autonomous approval
            status = "CONSENSUS_APPROVED"
            abstention = False
            rationale = (
                f"Ensemble unanimous consensus approved (all {len(allow_votes)} models voted ALLOW, "
                f"mean confidence {mean_prob:.3f}, std_dev {std_dev:.3f})."
            )
        else:
            status = "ABSTAIN_DELEGATE_TO_LLM"
            abstention = True
            rationale = (
                f"Selective Abstention triggered: Non-unanimous agreement, estimator disagreement, or uncertainty "
                f"(std={std_dev:.3f}, mean={mean_prob:.3f}, deny_models={len(deny_votes)}, abstained_models={len(abstain_votes)}). "
                f"Escalating boundary case to Cascade Router / LLM."
            )

        return CriticalConsensusResult(
            status=status,
            consensus_score=mean_prob,
            estimator_variance=variance,
            abstention_triggered=abstention,
            votes=votes,
            rationale=rationale,
        )

    def evaluate_with_cascade(
        self,
        action_name: str,
        target_path: Optional[str] = None,
        payload_diff: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full lifecycle evaluation: Critical Microbrain Ensemble -> LLM Cascade Router escalation."""
        consensus = self.evaluate_critical_action(action_name, target_path, payload_diff)
        if consensus.status == "CONSENSUS_APPROVED":
            return {
                "final_verdict": "AUTHORIZED",
                "tier": "MICROBRAIN_CONSENSUS",
                "consensus_result": consensus,
                "llm_invoked": False,
            }
        elif consensus.status == "CONSENSUS_REJECTED":
            return {
                "final_verdict": "BLOCKED",
                "tier": "MICROBRAIN_REJECTION",
                "consensus_result": consensus,
                "llm_invoked": False,
            }
        else:
            # Escalated to LLM Cascade Router
            from .cascade_ensemble import get_cascade_ensemble_router
            router = get_cascade_ensemble_router()
            cascade_res = router.evaluate_action_pipeline(
                task_prompt=f"Escalated action: {action_name} on {target_path}",
                target_tool=action_name,
                tool_args={"path": target_path, "diff": payload_diff, "cmd": target_path},
            )
            return {
                "final_verdict": cascade_res.verdict,
                "tier": "LLM_CASCADE_ESCALATION",
                "consensus_result": consensus,
                "cascade_verdict": cascade_res,
                "llm_invoked": True,
            }


_critical_ensemble_singleton: Optional[CriticalEnsemble] = None


def get_critical_ensemble() -> CriticalEnsemble:
    """Singleton getter for CriticalEnsemble."""
    global _critical_ensemble_singleton
    if _critical_ensemble_singleton is None:
        _critical_ensemble_singleton = CriticalEnsemble()
    return _critical_ensemble_singleton
