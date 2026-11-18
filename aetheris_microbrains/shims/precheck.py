"""Deterministic precheck gate for external agent harnesses.

This module is the stable integration surface for embedding Aetheris Microbrains
in front of an LLM agent harness (a pi-style terminal coding agent, a
DeepSeek-style web harness, or anything that executes tool calls, shell
commands, or model-written patches).

Design contract (honest by construction):

* Every verdict comes from the deterministic 4-layer engine
  (:func:`aetheris_microbrains.core.layered_decision_engine.decide`) - real
  ``ast`` parsing, real ``shlex`` tokenization, policy vetoes, and split-conformal
  abstention. No random-seeded routing, no fake learning.
* Fail-closed: parser errors on writes/shell deny; unknowns escalate, they never
  silently allow.
* ``escalate`` means "hand this to the LLM or a human" - the shim returns the
  verdict and the harness decides. The shim never executes anything itself.
* ``shadow_mode=True`` observes and journals what *would* be denied without
  changing behavior, so you can measure false-positive rate before enforcing.

Fork attribution: adapted from drtikov/microbrainsforagents (PolyForm
Noncommercial 1.0.0, Copyright (c) 2026 Dmitry Tikov). See NOTICE.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..core.layered_decision_engine import (
    PolicyConfig,
    analyze_shell_command,
    decide,
)

# Actions the gate can return.
ALLOW = "allow"
DENY = "deny"
ESCALATE = "escalate"


@dataclass
class PrecheckVerdict:
    """One deterministic verdict for one proposed action."""

    action: str  # ALLOW | DENY | ESCALATE
    reason_code: str
    layer: int
    detail: str = ""
    latency_ms: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        """True only when the action may execute without LLM/human review."""
        return self.action == ALLOW

    @property
    def needs_review(self) -> bool:
        """True when the harness should ask the LLM or a human before running."""
        return self.action == ESCALATE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrecheckGate:
    """Reusable deterministic precheck shim.

    Args:
        policy: Optional PolicyConfig (protected paths, diff-size limits).
        extra_shell_allowlist: Harness-specific executables the core allowlist
            does not know (e.g. ``pytest``, ``ruff``, ``npm``). These are only
            honored when the core parser already found the command *safe*
            (no metachar hazards, no destructive patterns) but merely
            *unrecognized*. Destructive commands stay denied regardless.
        shadow_mode: Journal would-be decisions without enforcing (verdicts
            come back as ESCALATE with the shadow fields preserved).
        journal_path: Optional JSONL audit trail path.
    """

    def __init__(
        self,
        policy: Optional[PolicyConfig] = None,
        extra_shell_allowlist: Optional[Iterable[str]] = None,
        shadow_mode: bool = False,
        journal_path: Optional[str] = None,
    ) -> None:
        self.policy = policy
        self.extra_shell_allowlist = {e.strip().lower() for e in (extra_shell_allowlist or ()) if e.strip()}
        self.shadow_mode = shadow_mode
        self.journal_path = str(journal_path) if journal_path else None

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------
    def _context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {"shadow_mode": self.shadow_mode, "log_decisions": True}
        if self.journal_path:
            ctx["journal_path"] = self.journal_path
        if extra:
            ctx.update(extra)
        return ctx

    def _from_decision(self, result: Dict[str, Any], detail: str = "") -> PrecheckVerdict:
        return PrecheckVerdict(
            action=str(result.get("decision", ESCALATE)),
            reason_code=str(result.get("reason_code", "")),
            layer=int(result.get("layer", -1)),
            detail=detail or str(result.get("label", "")),
            latency_ms=float(result.get("latency_ms", 0.0)),
            raw=result,
        )

    def check_shell(self, cmd: str) -> PrecheckVerdict:
        """Precheck one shell command string before execution."""
        t0 = time.perf_counter_ns()
        result = decide({"tool": "run_command", "cmd": cmd}, context=self._context(), policy=self.policy)

        # Harness-provided allowlist extension: only upgrades a bare
        # "unrecognized command" escalation, never a denial, and only when the
        # structural parser already proved the command safe.
        if (
            result.get("decision") == ESCALATE
            and result.get("reason_code") == "layer1_unrecognized_shell_command_escalate"
            and self.extra_shell_allowlist
        ):
            parsed = analyze_shell_command(cmd)
            exes = {str(c).lower() for c in parsed.get("commands", [])}
            if parsed.get("is_safe") and exes and exes.issubset(self.extra_shell_allowlist):
                result = dict(result)
                result["decision"] = ALLOW
                result["reason_code"] = "shim_harness_allowlist"
                result["latency_ms"] = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)

        return self._from_decision(result, detail=cmd[:200])

    def check_patch(self, path: str, diff: str, file_content: Optional[str] = None) -> PrecheckVerdict:
        """Precheck a model-written patch/diff before it touches disk.

        Catches syntax errors, secret leaks, dangerous calls (eval/exec,
        subprocess shell=True, ...), protected paths, and oversized diffs.
        """
        ctx = self._context({"file_content": file_content} if file_content else None)
        result = decide({"tool": "apply_patch", "path": path, "diff": diff}, context=ctx, policy=self.policy)
        return self._from_decision(result, detail=path)

    def check_tool_call(self, tool: str, args: Optional[Dict[str, Any]] = None) -> PrecheckVerdict:
        """Precheck a structured tool call (name + argument dict)."""
        args = dict(args or {})
        payload: Dict[str, Any] = {"tool": tool}
        for key in ("path", "file_path", "cmd", "command", "diff", "query", "task_prompt"):
            if key in args and args[key] is not None:
                payload[key] = args[key]
        ctx = self._context({"file_content": args["file_content"]} if args.get("file_content") else None)
        result = decide(payload, context=ctx, policy=self.policy)
        return self._from_decision(result, detail=f"{tool}({', '.join(sorted(args))})" if args else tool)

    def narrow_tools(
        self,
        task_description: str,
        available_tools: Sequence[str],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """MCP-Zero style Top-K tool-surface narrowing before an LLM call.

        Falls back to the full catalog on out-of-distribution tasks; never
        fabricates confidence.
        """
        from ..core.tool_routing_moe import narrow_tool_surface

        return narrow_tool_surface(task_description, available_tools=available_tools, top_k=top_k)

    # ------------------------------------------------------------------
    # Executor wrapping
    # ------------------------------------------------------------------
    def guard(self, tool_name: str, arg_mapper: Optional[Callable[..., Dict[str, Any]]] = None):
        """Decorator: precheck a tool executor before it runs.

        The wrapped function only executes on ALLOW. On DENY or ESCALATE a
        :class:`PrecheckBlocked` is raised carrying the verdict; the harness
        decides how to surface it (skip the step, ask the LLM, ask a human).
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                mapped = arg_mapper(*args, **kwargs) if arg_mapper else dict(kwargs)
                verdict = self.check_tool_call(tool_name, mapped)
                if not verdict.allowed:
                    raise PrecheckBlocked(verdict)
                return fn(*args, **kwargs)

            wrapper.__name__ = getattr(fn, "__name__", "guarded")
            wrapper.__doc__ = fn.__doc__
            wrapper._precheck_gate = self  # type: ignore[attr-defined]
            return wrapper

        return decorator


class PrecheckBlocked(Exception):
    """Raised by guarded executors when the gate does not allow the action."""

    def __init__(self, verdict: PrecheckVerdict) -> None:
        self.verdict = verdict
        super().__init__(f"precheck {verdict.action}: {verdict.reason_code} ({verdict.detail})")


def journal_verdicts(path: str, verdicts: Iterable[PrecheckVerdict]) -> None:
    """Append verdicts to a JSONL audit file (one JSON object per line)."""
    with Path(path).open("a", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.to_dict(), ensure_ascii=False, default=str) + "\n")
