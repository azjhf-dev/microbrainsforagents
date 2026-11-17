"""Counterfactual Reasoner: "What-If" lookahead simulation for Ouroboros.

Evaluates proposed actions, code edits, and tool invocations before execution
to predict cascading failure modes, downstream breaking changes, and error classes (Suki).
Powered by Aetheris Microbrain 'counterfactual_reasoner' + AST dependency analysis.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Outcome of a counterfactual lookahead simulation."""

    action: str
    target: str
    is_safe: bool
    risk_score: float
    hazards: List[str] = field(default_factory=list)
    requires_sandbox: bool = False
    suggested_guard: Optional[str] = None
    neural_prediction: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "is_safe": self.is_safe,
            "risk_score": round(self.risk_score, 3),
            "hazards": self.hazards,
            "requires_sandbox": self.requires_sandbox,
            "suggested_guard": self.suggested_guard,
            "neural_prediction": self.neural_prediction,
            "confidence": round(self.confidence, 3),
        }


def _find_downstream_dependents(repo_dir: Path, target_module: str) -> List[str]:
    """Find other python files in repo that import target_module."""
    dependents: List[str] = []
    mod_stem = Path(target_module).stem
    ouroboros_dir = repo_dir / "ouroboros"
    if not ouroboros_dir.is_dir():
        return dependents

    for py_file in ouroboros_dir.rglob("*.py"):
        if py_file.stem == mod_stem:
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if f"import {mod_stem}" in content or f"from ouroboros.{mod_stem}" in content or f"from ouroboros import {mod_stem}" in content:
                rel = py_file.relative_to(repo_dir).as_posix()
                dependents.append(rel)
                if len(dependents) >= 5:
                    break
        except Exception:
            continue
    return dependents


def simulate_counterfactual(
    repo_dir: Path,
    action: str,
    target: str,
    proposed_content: Optional[str] = None,
    recent_tools: Optional[List[str]] = None,
) -> CounterfactualResult:
    """Run counterfactual lookahead on an intended action.

    Args:
        repo_dir: Root directory of Ouroboros repo
        action: Tool or operation name (e.g., 'repo_write', 'str_replace_editor', 'git_commit')
        target: Target file path or command string
        proposed_content: Code snippet or patch being applied
        recent_tools: Names of recently called tools in this loop turn
    """
    hazards: List[str] = []
    risk_score = 0.0
    requires_sandbox = False
    suggested_guard = None
    recent_tools = recent_tools or []

    # Delegate lookahead simulation to the grounded Structural Causal World Model
    try:
        try:
            from core.causal_world_model import get_causal_world_model
        except ImportError:
            from ouroboros.causal_world_model import get_causal_world_model

        cwm = get_causal_world_model(repo_dir)
        scm_res = cwm.evaluate(
            action=action,
            target=target,
            proposed_code=proposed_content,
            recent_tools=recent_tools,
        )
        hazards.extend(scm_res.causal_mechanisms_triggered)
        if scm_res.suggested_guards:
            suggested_guard = scm_res.suggested_guards[0]
        risk_score = max(risk_score, scm_res.risk_score)
        if scm_res.predicted_outcomes.get("ancestral_core_impact"):
            requires_sandbox = True
    except Exception as exc:
        log.debug("SCM evaluation fallback: %s", exc)

    # 1. Precondition Cascade: Read-Before-Write check
    if action in ("repo_write", "str_replace_editor", "verified_patch"):
        if not recent_tools or recent_tools[-1] != "repo_read":
            hazards.append("precondition_breach: file-modifying tool without immediate preceding repo_read")
            risk_score += 0.4
            suggested_guard = f"Execute repo_read('{target}') first."

    # 2. Irreversibility check
    if any(k in target.lower() for k in ("rm -rf", "git reset --hard", "drop table", "clean -f")):
        hazards.append("irreversible_action: destructive operation detected")
        risk_score += 0.8
        requires_sandbox = True
        suggested_guard = "Route action to an isolated worktree branch instead of live repo."

    # 3. Core Kinship / Ancestral Site Hazard
    if any(ancestor in target.replace("\\", "/") for ancestor in ("ouroboros/agent.py", "ouroboros/loop.py", "ouroboros/context.py")):
        hazards.append("ancestral_site_hazard: target is a foundational core module")
        risk_score += 0.35
        requires_sandbox = True

    # 4. AST Syntax validation if python code proposed
    if proposed_content and target.endswith(".py"):
        try:
            ast.parse(proposed_content)
        except SyntaxError as syn_err:
            hazards.append(f"syntax_error: proposed code fails AST parse: {syn_err.msg} (line {syn_err.lineno})")
            risk_score += 0.7
            suggested_guard = "Fix Python syntax error before writing to disk."

    # 5. Downstream blast radius check
    if target.endswith(".py"):
        deps = _find_downstream_dependents(repo_dir, target)
        if len(deps) >= 3:
            hazards.append(f"high_blast_radius: {len(deps)} dependent modules import this file ({', '.join(deps[:3])})")
            risk_score += 0.25

    # 6. Neural Microbrain Counterfactual Consultation
    neural_pred = None
    confidence = 0.0
    try:
        try:
            from core.brain_moe_orchestrator import consult_microbrain
            res = consult_microbrain("counterfactual_reasoner", ["breaking_change" if risk_score >= 0.5 else "simulate_commit"])
            if res:
                neural_pred = res.get("prediction")
                confidence = float(res.get("confidence") or 0.0)
        except ImportError:
            from ouroboros.tools.moe_router import consult_expert_microbrain
            token = "breaking_change" if risk_score >= 0.5 else ("irreversible_edit" if requires_sandbox else "simulate_commit")
            neural_res = consult_expert_microbrain("counterfactual_reasoner", [token])
            if neural_res.get("available"):
                neural_pred = neural_res.get("prediction")
                confidence = float(neural_res.get("confidence") or 0.0)
    except Exception as exc:
        log.debug("Counterfactual neural consultation skipped: %s", exc)

    risk_score = min(1.0, max(0.0, risk_score))
    is_safe = risk_score < 0.5

    return CounterfactualResult(
        action=action,
        target=target,
        is_safe=is_safe,
        risk_score=risk_score,
        hazards=hazards,
        requires_sandbox=requires_sandbox,
        suggested_guard=suggested_guard,
        neural_prediction=neural_pred,
        confidence=confidence,
    )
