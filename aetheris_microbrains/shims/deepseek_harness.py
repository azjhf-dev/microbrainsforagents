"""Adapter: DeepSeek-style web harness (OpenAI-compatible function calling).

A DeepSeek-style harness talks to the model over an OpenAI-compatible chat API
and executes ``tool_calls`` from the response message::

    {"role": "assistant",
     "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "run_shell",
                                  "arguments": "{\\"command\\": \\"ls\\"}"}}]}

This adapter intercepts those tool calls before execution and vets them with
the deterministic precheck gate. It also cuts schema-token bloat on the way
*in* by narrowing a large tool registry to the Top-K relevant schemas before
the request is sent (MCP-Zero style, with honest fallback to the full catalog
on out-of-distribution tasks).

Fork attribution: adapted from drtikov/microbrainsforagents (PolyForm
Noncommercial 1.0.0, Copyright (c) 2026 Dmitry Tikov). See NOTICE.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .precheck import PrecheckGate, PrecheckVerdict

SHELL_TOOL_NAMES = {"run_shell", "shell", "bash", "run_command", "execute", "terminal"}
PATCH_TOOL_NAMES = {"apply_patch", "write_file", "edit_file", "patch", "str_replace_editor", "create_file"}


class DeepSeekHarnessPrecheck:
    """Precheck shim for a DeepSeek-style (OpenAI-compatible) web harness."""

    def __init__(
        self,
        extra_shell_allowlist: Optional[Sequence[str]] = None,
        shadow_mode: bool = False,
        journal_path: Optional[str] = None,
        **gate_kwargs: Any,
    ) -> None:
        self.gate = PrecheckGate(
            extra_shell_allowlist=extra_shell_allowlist or (),
            shadow_mode=shadow_mode,
            journal_path=journal_path,
            **gate_kwargs,
        )

    # ------------------------------------------------------------------
    # Inbound: narrow the tool surface before the API call
    # ------------------------------------------------------------------
    def narrow_tool_schemas(
        self,
        task: str,
        tool_schemas: Sequence[Dict[str, Any]],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Shrink OpenAI-format tool schemas to the Top-K relevant ones.

        Returns the subset to send plus routing metadata. On OOD/ambiguous
        tasks the full schema list is returned unchanged (honest fallback).
        """
        names = []
        by_name: Dict[str, Dict[str, Any]] = {}
        for schema in tool_schemas:
            fn = schema.get("function", {}) if isinstance(schema, dict) else {}
            name = fn.get("name") or schema.get("name")
            if name:
                names.append(str(name))
                by_name[str(name)] = schema

        narrowed = self.gate.narrow_tools(task, names, top_k=top_k)
        keep = narrowed.get("candidate_tools", names)
        return {
            "mode": narrowed.get("mode"),
            "tool_schemas": [by_name[n] for n in keep if n in by_name],
            "kept": [n for n in keep if n in by_name],
            "dropped": [n for n in names if n not in keep],
            "conformal_passed": narrowed.get("conformal_passed"),
            "reduction_ratio": narrowed.get("reduction_ratio"),
        }

    # ------------------------------------------------------------------
    # Outbound: vet tool calls before execution
    # ------------------------------------------------------------------
    def intercept_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[Dict[str, Any], PrecheckVerdict]:
        """Vet one OpenAI-format tool_call. Returns ``(tool_call, verdict)``.

        Executes nothing. The harness runs the tool only when
        ``verdict.allowed``; on ``needs_review`` it should route the verdict
        back to the model or a human; on deny it should refuse and tell the
        model why.
        """
        fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = str(fn.get("name", "")).strip().lower()
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except json.JSONDecodeError:
                verdict = PrecheckVerdict(
                    action="escalate",
                    reason_code="shim_unparseable_tool_arguments",
                    layer=1,
                    detail=f"{name}: arguments are not valid JSON",
                )
                return tool_call, verdict
        elif isinstance(raw_args, dict):
            args = dict(raw_args)
        else:
            args = {}

        cmd = args.get("command") or args.get("cmd")
        if name in SHELL_TOOL_NAMES and cmd:
            return tool_call, self.gate.check_shell(str(cmd))

        diff = args.get("diff") or args.get("patch") or args.get("content")
        path = args.get("path") or args.get("file_path") or args.get("filename") or ""
        if name in PATCH_TOOL_NAMES and diff is not None:
            return tool_call, self.gate.check_patch(str(path), str(diff), file_content=args.get("file_content"))

        return tool_call, self.gate.check_tool_call(name, args)

    def intercept_message(self, assistant_message: Dict[str, Any]) -> List[Tuple[Dict[str, Any], PrecheckVerdict]]:
        """Vet every tool_call on an assistant response message."""
        calls = assistant_message.get("tool_calls") or []
        return [self.intercept_tool_call(c) for c in calls]
