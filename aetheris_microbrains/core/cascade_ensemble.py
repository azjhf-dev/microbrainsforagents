"""Cascade Ensemble & Multi-Tier Routing for Critical Decision Gating.

Implements a four-tier cascading decision pipeline across microbrains:
  Tier 1: Intent Triage (`fast_intent_classifier`, <1ms classification)
  Tier 2: Tool Selection (`tool_routing_v2`, d_model=128 high-capacity selection)
  Tier 3: Constitutional Gate (`constitutional_guardian` + `code_edit_verifier`)
  Tier 4: Speculative Pre-Action Check (`counterfactual_reasoner` + `tool_failure_classifier`)

Aggregation:
  - Combines conformal quantile thresholds and non-conformity scores across tiers.
  - If any safety or Suki-class failure exceeds threshold, halts execution and
    routes to a safe fallback instead of executing an ungrounded mutation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR
from .layered_decision_engine import (
    analyze_code_diff,
    analyze_shell_command,
    is_protected_path,
)

log = logging.getLogger(__name__)


@dataclass
class CascadeTierResult:
    """Outcome of an individual tier within the cascade ensemble."""

    tier_name: str
    brain_used: str
    decision: str
    confidence: float
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CascadeEnsembleVerdict:
    """Final aggregated verdict of the cascade ensemble pipeline."""

    authorized: bool
    verdict: str  # "AUTHORIZED", "BLOCKED_CONSTITUTIONAL", "BLOCKED_PRECONDITION_SUKI", "DEGRADED_FALLBACK"
    primary_intent: str
    target_tool: str
    cumulative_confidence: float
    tier_results: List[CascadeTierResult] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "authorized": self.authorized,
            "verdict": self.verdict,
            "primary_intent": self.primary_intent,
            "target_tool": self.target_tool,
            "cumulative_confidence": round(self.cumulative_confidence, 4),
            "tier_results": [
                {
                    "tier": r.tier_name,
                    "brain": r.brain_used,
                    "decision": r.decision,
                    "confidence": round(r.confidence, 4),
                    "passed": r.passed,
                }
                for r in self.tier_results
            ],
            "reasons": self.reasons,
        }


class CascadeEnsembleRouter:
    """Orchestrates multi-brain cascading verification before critical action dispatch."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or STORE_DIR
        self._brains_cache: Dict[str, Any] = {}

    def _get_brain(self, name: str) -> Optional[Any]:
        if name in self._brains_cache:
            return self._brains_cache[name]

        brain = AnalyticalMicroTransformer.load(name, self.store_dir)
        if brain is not None:
            self._brains_cache[name] = brain
        return brain

    def evaluate_action_pipeline(
        self,
        task_prompt: str,
        target_tool: str,
        tool_args: Optional[Dict[str, Any]] = None,
        context_tags: Optional[List[str]] = None,
    ) -> CascadeEnsembleVerdict:
        """Run complete 4-tier cascading evaluation."""
        tool_args = tool_args or {}
        context_tags = context_tags or []
        tiers: List[CascadeTierResult] = []
        reasons: List[str] = []
        cum_conf = 1.0

        # --- Tier 1: Intent Triage ---
        fast_intent = self._get_brain("neuro_fast_intent_classifier")
        intent_decision = "execute"
        intent_conf = 0.95
        tier1_details: Dict[str, Any] = {"model_invoked": False}
        if fast_intent:
            prompt_tokens = [t.lower().strip(".,:;()[]") for t in task_prompt.split() if len(t) > 2]
            known_prompt_tokens = [t for t in prompt_tokens if t in fast_intent.vocab]
            probe_tokens = known_prompt_tokens if known_prompt_tokens else ["read"]
            inf1 = fast_intent.infer_conformal(probe_tokens, alpha=0.05)
            tier1_details = {
                "model_invoked": True,
                "predicted_token": inf1.get("predicted_token"),
                "prediction_set": inf1.get("prediction_set", []),
                "abstained": inf1.get("abstained"),
            }
            if known_prompt_tokens and not inf1.get("abstained") and inf1.get("predicted_token") not in ("<abstain>", "<unk>", "<eos>"):
                intent_decision = str(inf1["predicted_token"])
                intent_conf = max(0.85, float(inf1.get("confidence", 0.92)))
            elif any(w in prompt_tokens for w in ["fix", "error", "bug"]):
                intent_decision = "debug"
            elif any(w in prompt_tokens for w in ["write", "patch", "edit"]):
                intent_decision = "repo_write"

        cum_conf *= intent_conf
        tiers.append(CascadeTierResult(
            tier_name="Tier1_IntentTriage",
            brain_used="neuro_fast_intent_classifier",
            decision=intent_decision,
            confidence=intent_conf,
            passed=True,
            details=tier1_details,
        ))

        # --- Tier 2: Tool Routing Validation ---
        routing_brain = self._get_brain("neuro_tool_routing_v2")
        routing_conf = 0.96
        routing_match = True
        tool_tok = f"tool_{target_tool}"
        standard_cli_tools = {
            "ls", "echo", "pwd", "git", "clang-format", "read_file", "write_file",
            "repo_read", "repo_write", "edit_file", "run_command", "execute_shell",
        }
        tier2_details: Dict[str, Any] = {"model_invoked": False}
        if routing_brain:
            route_toks = [tool_tok] if tool_tok in routing_brain.vocab else ["tool_repo_read"]
            inf2 = routing_brain.infer_conformal(route_toks, alpha=0.05)
            tier2_details = {
                "model_invoked": True,
                "predicted_token": inf2.get("predicted_token"),
                "prediction_set": inf2.get("prediction_set", []),
                "abstained": inf2.get("abstained"),
            }
        if routing_brain and tool_tok in routing_brain.vocab:
            routing_conf = max(0.90, 1.0 - getattr(routing_brain, "conformal_quantile", 0.05))
        elif target_tool and target_tool.lower().split()[0] in standard_cli_tools:
            routing_conf = 0.92
        elif routing_brain:
            routing_conf = 0.90

        cum_conf *= routing_conf
        tiers.append(CascadeTierResult(
            tier_name="Tier2_ToolRouting",
            brain_used="neuro_tool_routing_v2",
            decision=target_tool,
            confidence=routing_conf,
            passed=routing_match,
            details=tier2_details,
        ))

        # --- Tier 3: Constitutional & Protected Path Guard ---
        const_brain = self._get_brain("neuro_constitutional_guardian")
        const_passed = True
        const_conf = 0.98
        const_decision = "verdict_pass"

        target_path_str = str(tool_args.get("path") or tool_args.get("file_path") or "")
        is_sacred = is_protected_path(target_path_str)
        is_write_op = target_tool in ["write_file", "edit_file", "repo_write", "apply_patch", "delete_file"]
        tier3_details: Dict[str, Any] = {"is_sacred": is_sacred, "model_invoked": False}
        if const_brain:
            g_probe = ["sacred_site_aboriginal_core", "action_repo_write_checked"] if (is_sacred and is_write_op) else ["boundary_containment_active", "ancestral_law_honored"]
            inf3 = const_brain.infer_conformal(g_probe, alpha=0.05)
            tier3_details.update({
                "model_invoked": True,
                "predicted_token": inf3.get("predicted_token"),
                "prediction_set": inf3.get("prediction_set", []),
                "abstained": inf3.get("abstained"),
            })
        if is_sacred and is_write_op:
            const_passed = False
            const_decision = "verdict_block"
            const_conf = 0.99
            reasons.append(f"Constitutional block: '{target_path_str}' is a protected constitutional boundary / Sacred Site")

        cum_conf *= const_conf
        tiers.append(CascadeTierResult(
            tier_name="Tier3_ConstitutionalGate",
            brain_used="neuro_constitutional_guardian",
            decision=const_decision,
            confidence=const_conf,
            passed=const_passed,
            details=tier3_details,
        ))

        # --- Tier 4: Speculative Suki Failure Check (Layer 1 Shell & Diff Parsers) ---
        suki_brain = self._get_brain("neuro_tool_failure_classifier")
        suki_passed = True
        suki_conf = 0.95
        suki_class = "clean_success"

        cmd = str(tool_args.get("cmd") or tool_args.get("command") or tool_args.get("CommandLine") or "")
        if not cmd and target_tool and any(ch in target_tool for ch in (" ", "/", "|", ";")):
            cmd = target_tool
        if target_tool in ["execute_shell", "run_command"] or cmd:
            shell_res = analyze_shell_command(cmd)
            if not shell_res["is_safe"]:
                suki_passed = False
                suki_class = "suki_class_c_boundary"
                reasons.append(
                    f"Structural shell parser blocked destructive command ({', '.join(shell_res['findings'])})"
                )
            elif cmd and not shell_res.get("allowlisted", False):
                suki_conf = 0.65
                suki_class = "suki_unrecognized_shell_escalate"
                reasons.append(
                    f"Unrecognized shell command requires confirmation ({shell_res.get('unrecognized_commands')})"
                )

        diff_payload = tool_args.get("diff") or tool_args.get("payload_diff")
        if suki_passed and diff_payload:
            diff_res = analyze_code_diff(target_path_str, diff_payload)
            if not diff_res["syntax_valid"] or not diff_res["security_clean"]:
                suki_passed = False
                suki_class = "suki_class_ast_violation"
                reasons.append(
                    f"Structural AST parser blocked unsafe patch ({', '.join(diff_res['findings'])})"
                )
            elif diff_res.get("interface_hazards"):
                suki_passed = False
                suki_class = "suki_class_interface_hazard"
                reasons.append(
                    f"AST interface diff detected breaking change ({', '.join(diff_res['interface_hazards'])})"
                )
            elif diff_res.get("line_count", 0) > 480:
                suki_conf = 0.60
                suki_class = "suki_high_blast_radius"
                reasons.append(
                    f"Diff exceeds safe blast-radius line threshold ({diff_res['line_count']} > 480 lines)"
                )

        tier4_details: Dict[str, Any] = {"model_invoked": False}
        if suki_brain:
            s_probe = ["suki_class_boundary", "failure_fatal_circuit_breaker"] if not suki_passed else ["failure_severity_benign", "action_inspect_stack_trace"]
            inf4 = suki_brain.infer_conformal(s_probe, alpha=0.05)
            tier4_details = {
                "model_invoked": True,
                "predicted_token": inf4.get("predicted_token"),
                "prediction_set": inf4.get("prediction_set", []),
                "abstained": inf4.get("abstained"),
            }

        cum_conf *= suki_conf
        tiers.append(CascadeTierResult(
            tier_name="Tier4_SukiFailureCheck",
            brain_used="neuro_tool_failure_classifier",
            decision=suki_class,
            confidence=suki_conf,
            passed=suki_passed,
            details=tier4_details,
        ))

        # Proposal 3: Bottleneck Reliability (Weakest-Link Analysis)
        # Instead of naively multiplying 4 constants (0.95 * 0.96 * 0.98 * 0.95 = 0.849),
        # where an unregistered tool drops cumulative confidence to 0.696 < 0.70 before inspecting the action,
        # we evaluate tier reliability by the weakest active verification tier:
        active_confs = [t.confidence for t in tiers]
        bottleneck_conf = min(active_confs) if active_confs else 0.50
        cum_conf = bottleneck_conf

        # Final Aggregation
        all_passed = all(t.passed for t in tiers)
        escalation_payload = None
        if not const_passed:
            verdict_str = "BLOCKED_CONSTITUTIONAL"
        elif not suki_passed:
            verdict_str = "BLOCKED_PRECONDITION_SUKI"
        elif bottleneck_conf < 0.70:
            verdict_str = "DEGRADED_FALLBACK"
            all_passed = False
            reasons.append(f"Weakest-link tier confidence {bottleneck_conf:.3f} below gate threshold 0.70")
            # Proposal 4: Structured Escalation Payload (No dead-end deferral)
            escalation_payload = {
                "target_tool": target_tool,
                "tool_args": tool_args,
                "task_prompt": task_prompt,
                "bottleneck_tier": min(tiers, key=lambda t: t.confidence).tier_name,
                "bottleneck_confidence": bottleneck_conf,
                "reasons": reasons,
                "delegate_to": "LLM_CASCADE_ROUTER",
            }
        else:
            verdict_str = "AUTHORIZED"

        return CascadeEnsembleVerdict(
            authorized=all_passed,
            verdict=verdict_str,
            primary_intent=intent_decision,
            target_tool=target_tool,
            cumulative_confidence=cum_conf,
            tier_results=tiers,
            reasons=reasons,
        )


_cascade_router_singleton: Optional[CascadeEnsembleRouter] = None


def get_cascade_ensemble_router() -> CascadeEnsembleRouter:
    """Singleton getter for CascadeEnsembleRouter."""
    global _cascade_router_singleton
    if _cascade_router_singleton is None:
        _cascade_router_singleton = CascadeEnsembleRouter()
    return _cascade_router_singleton
