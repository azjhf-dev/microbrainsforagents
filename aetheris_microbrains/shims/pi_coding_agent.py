"""Adapter: pi-style terminal coding agent.

A pi-style coding agent loops: model proposes -> tool executes (read file,
write file, apply patch, run shell). This adapter inserts the deterministic
precheck gate between "model proposes" and "tool executes" for every action
that can touch the filesystem or a shell.

What it catches before execution:
  * malformed patches (ast syntax errors) - saves a failed-test round trip
  * secret leaks and dangerous calls inside model-written code
  * destructive shell commands (rm -rf, dd, mkfs, curl|sh, DROP DATABASE)
  * writes to protected paths (constitution/, memory/bible.md, ...)
  * oversized diffs (blast-radius escalation)

What it deliberately does NOT do: pick tools for the model. A pi agent has a
fixed, tiny tool menu; routing adds nothing there. The gate only vets actions.

Fork attribution: adapted from drtikov/microbrainsforagents (PolyForm
Noncommercial 1.0.0, Copyright (c) 2026 Dmitry Tikov). See NOTICE.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .precheck import ALLOW, DENY, ESCALATE, PrecheckGate, PrecheckVerdict

# pi-style tool surface -> the argument keys this adapter knows how to vet.
# Unknown tools still get a policy + risk-class pass via check_tool_call.
PI_SHELL_TOOLS = {"bash", "shell", "run_command", "execute_shell", "terminal"}
PI_PATCH_TOOLS = {"apply_patch", "edit_file", "write_file", "patch", "str_replace_editor"}
PI_PATH_KEYS = ("path", "file_path", "filename", "target_file")
PI_CMD_KEYS = ("cmd", "command", "command_line")
PI_DIFF_KEYS = ("diff", "patch", "content", "new_str", "replacement")

# Commands a coding agent legitimately runs that the core allowlist does not
# already know (it covers ls/pwd/cat/git-read-only/pytest/ruff/black/mypy/
# grep/rg/find and friends). Safe-only extension: destructive forms of these
# are still denied by the structural parser before this list is consulted.
PI_DEFAULT_EXTRA_ALLOWLIST = (
    "python", "python3", "pip", "node", "npm", "npx", "tsc",
    "cargo", "rustc", "go", "make", "gcc", "g++", "clang",
    "jq", "sed", "awk", "diff", "touch", "mkdir", "cp",
)


class PiAgentPrecheck:
    """Precheck shim for a pi-style coding agent loop.

    Usage::

        gate = PiAgentPrecheck(journal_path="precheck.jsonl")
        verdict = gate.before_tool_call("bash", {"command": "rm -rf /"})
        if verdict.allowed:
            run_the_tool()
        elif verdict.needs_review:
            ask_llm_or_human(verdict)
        else:
            reject(verdict)

    Or wrap executors directly::

        run_bash = gate.wrap("bash", run_bash)
    """

    def __init__(
        self,
        extra_shell_allowlist: Optional[List[str]] = None,
        shadow_mode: bool = False,
        journal_path: Optional[str] = None,
        **gate_kwargs: Any,
    ) -> None:
        allowlist = list(PI_DEFAULT_EXTRA_ALLOWLIST)
        if extra_shell_allowlist:
            allowlist.extend(extra_shell_allowlist)
        self.gate = PrecheckGate(
            extra_shell_allowlist=allowlist,
            shadow_mode=shadow_mode,
            journal_path=journal_path,
            **gate_kwargs,
        )

    # ------------------------------------------------------------------
    def before_tool_call(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> PrecheckVerdict:
        """Vet one proposed tool call. Call this before executing any tool."""
        args = dict(args or {})
        clean = str(tool_name or "").strip().lower()

        cmd = next((args[k] for k in PI_CMD_KEYS if args.get(k)), None)
        if clean in PI_SHELL_TOOLS and cmd:
            return self.gate.check_shell(str(cmd))

        if clean in PI_PATCH_TOOLS:
            path = next((args[k] for k in PI_PATH_KEYS if args.get(k)), "") or ""
            diff = next((args[k] for k in PI_DIFF_KEYS if args.get(k)), None)
            if diff is not None:
                return self.gate.check_patch(str(path), str(diff), file_content=args.get("file_content"))

        return self.gate.check_tool_call(clean, args)

    def wrap(self, tool_name: str, executor: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a tool executor so every call is prechecked first.

        Raises ``PrecheckBlocked`` on deny/escalate; catch it in the agent loop
        and route the verdict back to the model or a human.
        """
        from .precheck import PrecheckBlocked

        def guarded(*args: Any, **kwargs: Any) -> Any:
            verdict = self.before_tool_call(tool_name, kwargs)
            if not verdict.allowed:
                raise PrecheckBlocked(verdict)
            return executor(*args, **kwargs)

        guarded.__name__ = getattr(executor, "__name__", "guarded_tool")
        return guarded

    # ------------------------------------------------------------------
    def narrow_tools(self, task: str, available_tools: List[str], top_k: int = 4) -> Dict[str, Any]:
        """Optional: shrink a large tool registry before the model call.

        Only useful if the pi deployment exposes many MCP tools; with the
        default fixed menu this is a no-op win.
        """
        return self.gate.narrow_tools(task, available_tools, top_k=top_k)
