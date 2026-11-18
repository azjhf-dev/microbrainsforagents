"""Layered Decision Engine & Structural Code/Shell Parsers (Architecture Layers 0-3).

Separates decisions cleanly by nature rather than conflating rules with ML:
  - Layer 0 (Policy in Code): Tool risk class, protected constitutional paths, and explicit permissions.
  - Layer 1 (Structural Parsers): Real Python AST inspection (with context dedent/wrapping),
    language-aware diff handling (skipping non-Python AST parsing), and shlex-based shell parsing.
  - Layer 2 (Calibrated Classifier Boundary): Micro-classifier inference with Split-Conformal sets.
  - Layer 3 (LLM / Human Escalation): Unknown, destructive, or ambiguous actions escalate.

Contract:
  decide(input_data, context=None) -> {
      decision: "allow" | "deny" | "escalate",
      label: str,
      prediction_set: list[str],
      reason_code: str,
      layer: int,
      model_version: str,
      rules_version: str,
      latency_ms: float,
  }
"""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

RULES_VERSION = "2.0.0-layered-structural"
DEFAULT_MODEL_VERSION = "analytical-wsp-v2"

# Tool risk taxonomy (Layer 0)
TOOL_RISK_CLASSES: Dict[str, str] = {
    # Read-only operations
    "read_file": "READ_ONLY",
    "repo_read": "READ_ONLY",
    "list_dir": "READ_ONLY",
    "search_code": "READ_ONLY",
    "git_status": "READ_ONLY",
    "git_log": "READ_ONLY",
    "git_diff": "READ_ONLY",
    "microbrain_consult": "READ_ONLY",
    "microbrain_status": "READ_ONLY",
    "vsa_query": "READ_ONLY",
    "memory_recall": "READ_ONLY",
    "blackboard_fetch": "READ_ONLY",
    "knowledge_search": "READ_ONLY",
    # Reversible mutations
    "write_file": "REVERSIBLE_WRITE",
    "repo_write": "REVERSIBLE_WRITE",
    "edit_file": "REVERSIBLE_WRITE",
    "str_replace_editor": "REVERSIBLE_WRITE",
    "apply_patch": "REVERSIBLE_WRITE",
    "format_code": "REVERSIBLE_WRITE",
    # External / process execution
    "run_command": "EXTERNAL_ACTION",
    "execute_shell": "EXTERNAL_ACTION",
    "run_tests": "EXTERNAL_ACTION",
    "pytest_runner": "EXTERNAL_ACTION",
    "git_commit": "EXTERNAL_ACTION",
    # Irreversible / high-impact operations
    "git_push": "IRREVERSIBLE_DESTRUCTIVE",
    "delete_file": "IRREVERSIBLE_DESTRUCTIVE",
    "drop_db": "IRREVERSIBLE_DESTRUCTIVE",
}

DEFAULT_PROTECTED_PATTERNS: Tuple[str, ...] = (
    "memory/bible.md",
    "bible.md",
    "constitution/",
    "immutable_axioms",
    "aboriginal",
    "sacred_site",
    "kinship",
    "totem",
    "integrity_seal.json",
    "integrity_manifest",
)


@dataclass
class PolicyConfig:
    """Configurable security policy without hardcoded domain dogma."""

    protected_patterns: Tuple[str, ...] = DEFAULT_PROTECTED_PATTERNS
    require_approval_for_external: bool = False
    require_approval_for_irreversible: bool = True
    max_diff_lines: int = 480


DEFAULT_POLICY = PolicyConfig()


def is_protected_path(path_str: Optional[str], policy: Optional[PolicyConfig] = None) -> bool:
    """Return True if path points to a constitutional or protected system asset.

    Explicitly avoids false positives on generic code/math paths like `src/math/axioms.py`.
    """
    if not path_str:
        return False
    cfg = policy or DEFAULT_POLICY
    normalized = str(path_str).replace("\\", "/").strip().lower()
    return any(pat in normalized for pat in cfg.protected_patterns)


def evaluate_layer0_policy(
    tool_name: str,
    target_path: Optional[str] = None,
    policy: Optional[PolicyConfig] = None,
) -> Dict[str, Any]:
    """Evaluate Layer 0 deterministic policy (risk class + protected boundaries)."""
    cfg = policy or DEFAULT_POLICY
    clean_tool = str(tool_name or "").strip().lower()
    risk_class = TOOL_RISK_CLASSES.get(clean_tool, "EXTERNAL_ACTION")

    is_write = risk_class in ("REVERSIBLE_WRITE", "IRREVERSIBLE_DESTRUCTIVE", "EXTERNAL_ACTION")
    if target_path and is_protected_path(target_path, cfg):
        if is_write or clean_tool in ("", "apply_patch"):
            return {
                "decision": "deny",
                "risk_class": risk_class,
                "reason_code": "layer0_protected_path_veto",
                "protected_path": target_path,
            }

    if risk_class == "IRREVERSIBLE_DESTRUCTIVE" and cfg.require_approval_for_irreversible:
        return {
            "decision": "escalate",
            "risk_class": risk_class,
            "reason_code": "layer0_irreversible_requires_confirmation",
        }

    return {
        "decision": "allow",
        "risk_class": risk_class,
        "reason_code": "layer0_policy_pass",
    }


def _parse_python_with_context(code_text: str) -> Optional[ast.AST]:
    """Parse Python code or hunk fragment by wrapping in valid block contexts if needed."""
    if not code_text.strip():
        return ast.parse("")

    candidates = [code_text]

    # Indented hunk or mixed leading spaces from synthetic snippet
    dedented = textwrap.dedent(code_text)
    if dedented != code_text:
        candidates.append(dedented)

    # Function body hunk (return, yield, etc.)
    indented_for_fn = textwrap.indent(dedented, "    ")
    candidates.append(f"def _hunk_ctx():\n{indented_for_fn}")

    # Exception handler hunk (`except ...:` / `finally:`)
    candidates.append(f"try:\n    pass\n{dedented}")

    # Branch continuation hunk (`elif ...:` / `else:`)
    candidates.append(f"if True:\n    pass\n{dedented}")

    for cand in candidates:
        try:
            return ast.parse(cand)
        except SyntaxError:
            continue
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    """Resolve constant string or concatenated string literals (`'sys' + 'tem'`)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _const_str(node.left)
        right = _const_str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


class _SecurityASTVisitor(ast.NodeVisitor):
    """AST security analyzer catching direct, aliased, and getattr-obfuscated sinks."""

    DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__"}
    DANGEROUS_ATTRS = {"system", "popen", "rmtree", "loads", "load", "unsafe_load", "spawn"}
    DANGEROUS_MODULES = {"os", "subprocess", "shutil", "pickle", "cpickle", "marshal", "shelve", "pty", "builtins", "__builtins__"}

    def __init__(self) -> None:
        self.findings: List[str] = []
        self.dangerous_aliases: Set[str] = set()
        self.module_aliases: Dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_mod = alias.name.split(".")[0].lower()
            as_name = alias.asname or alias.name.split(".")[0]
            if root_mod in self.DANGEROUS_MODULES:
                self.module_aliases[as_name] = root_mod
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = (node.module or "").split(".")[0].lower()
        for alias in node.names:
            name_lower = alias.name.lower()
            as_name = alias.asname or alias.name
            if name_lower in self.DANGEROUS_BUILTINS or (mod in self.DANGEROUS_MODULES and name_lower in {"system", "popen", "rmtree", "run", "call", "check_output", "loads", "load", "remove", "unlink"}):
                self.dangerous_aliases.add(as_name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self._check_alias_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_alias_assignment([node.target], node.value)
        self.generic_visit(node)

    def _check_alias_assignment(self, targets: List[ast.AST], value: ast.AST) -> None:
        is_dangerous_rhs = False
        if isinstance(value, ast.Name) and (value.id in self.DANGEROUS_BUILTINS or value.id in self.dangerous_aliases):
            is_dangerous_rhs = True
        elif isinstance(value, ast.Attribute) and value.attr in {"system", "popen", "rmtree", "loads", "run", "call"}:
            is_dangerous_rhs = True
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "getattr":
            is_dangerous_rhs = True

        if is_dangerous_rhs:
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    self.dangerous_aliases.add(tgt.id)
                    self.findings.append(f"dangerous_alias:{tgt.id}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        # Case 1: Direct or aliased name call (`eval(...)`, `run(...)`)
        if isinstance(func, ast.Name):
            if func.id in self.DANGEROUS_BUILTINS:
                self.findings.append(f"dangerous_builtin:{func.id}")
            elif func.id in self.dangerous_aliases:
                self.findings.append(f"aliased_dangerous_call:{func.id}")
            elif func.id == "getattr":
                self._check_getattr(node)

        # Case 2: Obfuscated dynamic dispatch `getattr(os, 'sys' + 'tem')(...)`
        elif isinstance(func, ast.Call):
            if isinstance(func.func, ast.Name) and func.func.id == "getattr":
                self._check_getattr(func)
                self.findings.append("obfuscated_dynamic_invocation:getattr")

        # Case 3: Attribute call (`os.system`, `subprocess.run`, `shutil.rmtree`, `pickle.loads`, ...)
        elif isinstance(func, ast.Attribute):
            attr = func.attr
            base_name = func.value.id.lower() if isinstance(func.value, ast.Name) else ""
            resolved_mod = self.module_aliases.get(base_name, base_name)

            if attr in {"eval", "exec", "system", "popen"}:
                self.findings.append(f"dangerous_method:{resolved_mod}.{attr}" if resolved_mod else f"dangerous_method:{attr}")
            elif resolved_mod == "subprocess" or (isinstance(func.value, ast.Name) and func.value.id == "subprocess"):
                if attr in {"run", "Popen", "call", "check_output", "check_call", "getoutput", "getstatusoutput"}:
                    has_shell_true = any(
                        kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                        for kw in node.keywords
                    )
                    self.findings.append("subprocess_shell_true" if has_shell_true else f"subprocess_call:{attr}")
            elif resolved_mod == "shutil" or (isinstance(func.value, ast.Name) and func.value.id == "shutil"):
                if attr == "rmtree":
                    self.findings.append("destructive_fs_call:shutil.rmtree")
            elif resolved_mod == "os" or (isinstance(func.value, ast.Name) and func.value.id == "os"):
                if attr in {"system", "popen", "remove", "unlink", "rmdir", "execl", "execv", "spawnl"}:
                    self.findings.append(f"dangerous_os_call:os.{attr}")
            elif resolved_mod in {"pickle", "cpickle", "marshal", "shelve"}:
                if attr in {"loads", "load", "open"}:
                    self.findings.append(f"unsafe_deserialization:{resolved_mod}.{attr}")
            elif resolved_mod == "yaml" and attr in {"unsafe_load", "load"}:
                has_safe_loader = any(
                    kw.arg == "Loader" and isinstance(kw.value, (ast.Name, ast.Attribute)) and "Safe" in ast.dump(kw.value)
                    for kw in node.keywords
                )
                if not has_safe_loader:
                    self.findings.append(f"unsafe_yaml_load:yaml.{attr}")

        self.generic_visit(node)

    def _check_getattr(self, node: ast.Call) -> None:
        target_obj = node.args[0].id.lower() if node.args and isinstance(node.args[0], ast.Name) else ""
        attr_str = _const_str(node.args[1]) if len(node.args) >= 2 else None
        if target_obj in self.DANGEROUS_MODULES:
            self.findings.append(f"obfuscated_getattr:{target_obj}.{attr_str or '<dynamic>'}")
        elif len(node.args) >= 2 and isinstance(node.args[1], ast.BinOp):
            self.findings.append(f"obfuscated_getattr_concat:{attr_str or '<expr>'}")
        elif attr_str in {"system", "popen", "eval", "exec", "__import__", "rmtree"}:
            self.findings.append(f"obfuscated_getattr_sink:{attr_str}")


def analyze_code_diff(file_path: Optional[str], diff_text: Optional[str]) -> Dict[str, Any]:
    """Layer 1 language-aware structural diff analysis.

    Returns {
        language: str,
        syntax_valid: bool,
        security_clean: bool,
        findings: list[str],
        line_count: int,
        ast_nodes: list[str],
    }
    """
    if not diff_text:
        return {
            "language": "empty",
            "syntax_valid": True,
            "security_clean": True,
            "findings": [],
            "line_count": 0,
            "ast_nodes": [],
        }

    path_str = str(file_path or "").strip()
    ext = Path(path_str).suffix.lower() if path_str else ""
    line_count = len(diff_text.splitlines())

    # Extract proposed lines carefully
    raw_lines = diff_text.splitlines()
    has_unified = any(l.startswith(("--- ", "+++ ", "@@ ")) for l in raw_lines)
    if has_unified:
        code_lines = []
        for l in raw_lines:
            if l.startswith(("---", "+++", "@@")) or l.startswith("-"):
                continue
            if l.startswith("+"):
                code_lines.append(l[1:])
            elif l.startswith(" ") and not any(other.startswith("+") and not other.startswith("+++") and len(other) > 1 and other[1] != " " and l.startswith("    ") for other in raw_lines):
                # Strip single diff context space unless lines are raw unprefixed indentation after a synthetic +
                code_lines.append(l[1:])
            else:
                code_lines.append(l)
        extracted = "\n".join(code_lines)
    else:
        # Handle `+line` fragments without headers or raw indented snippets
        if all(l.startswith("+") or not l.strip() for l in raw_lines if l):
            extracted = "\n".join(l[1:] if l.startswith("+") else l for l in raw_lines)
        else:
            extracted = diff_text

    dedented = textwrap.dedent(extracted).strip("\n")

    # Non-Python files: skip Python ast.parse and validate format appropriately
    if ext == ".json":
        try:
            json.loads(dedented)
            json_ok = True
        except Exception:
            # Allow partial JSON diff hunks (e.g. single dependency key-value line) as long as quotes/brackets aren't completely garbage
            json_ok = ('"' in dedented and ":" in dedented) or dedented.strip().startswith(("{", "["))
        return {
            "language": "json",
            "syntax_valid": json_ok,
            "security_clean": True,
            "findings": [] if json_ok else ["malformed_json_hunk"],
            "line_count": line_count,
            "ast_nodes": [],
        }

    if ext in {".ts", ".tsx", ".js", ".jsx", ".css", ".html", ".md", ".yaml", ".yml", ".toml", ".txt", ".c", ".cpp", ".h", ".rs", ".go"}:
        # Language-appropriate lightweight check (never feed TS/JS/C/YAML into Python ast.parse)
        non_py_findings: List[str] = []
        for ln in dedented.splitlines():
            stripped = ln.strip()
            if stripped.startswith(("//", "#", "/*", "*")):
                continue
            if re.search(r"\beval\s*\(", stripped) and ext in {".js", ".ts", ".jsx", ".tsx"}:
                non_py_findings.append("js_eval_call")
        return {
            "language": ext.lstrip(".") or "text",
            "syntax_valid": True,
            "security_clean": len(non_py_findings) == 0,
            "findings": non_py_findings,
            "line_count": line_count,
            "ast_nodes": [],
        }

    # Python (or unspecified code snippet)
    tree = _parse_python_with_context(dedented)
    if tree is None:
        return {
            "language": "python",
            "syntax_valid": False,
            "security_clean": False,
            "findings": ["syntax_tree_malformed"],
            "line_count": line_count,
            "ast_nodes": [],
        }

    visitor = _SecurityASTVisitor()
    visitor.visit(tree)

    ast_nodes: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name != "_hunk_ctx":
            ast_nodes.append("ast_node_function_def")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            ast_nodes.append("ast_node_import")
        elif isinstance(node, ast.Call):
            ast_nodes.append("ast_node_call")

    return {
        "language": "python",
        "syntax_valid": True,
        "security_clean": len(visitor.findings) == 0,
        "findings": visitor.findings,
        "line_count": line_count,
        "ast_nodes": list(dict.fromkeys(ast_nodes)),
    }


def _tokenize_shell_segment(seg: str) -> List[str]:
    try:
        return shlex.split(seg, posix=True)
    except ValueError:
        return seg.strip().split()


SAFE_SHELL_COMMANDS: Set[str] = {
    "ls", "pwd", "echo", "cat", "head", "tail", "wc", "sort", "uniq",
    "grep", "rg", "find", "which", "where", "whoami", "date", "env",
    "clang-format", "black", "ruff", "mypy", "pytest", "python", "python3",
}


def analyze_shell_command(cmd_str: Optional[str]) -> Dict[str, Any]:
    """Layer 1 structural POSIX/Windows shell parser using shlex tokenization.

    Replaces brittle substring checks (e.g. `'format ' in cmd` blocking `git log --format`).
    """
    if not cmd_str or not str(cmd_str).strip():
        return {"is_safe": True, "findings": [], "commands": [], "risk_class": "READ_ONLY"}

    cmd = str(cmd_str).strip()
    lower_cmd = cmd.lower()

    # Check raw SQL destructive commands passed directly
    if re.search(r"\b(drop\s+database|truncate\s+table)\b", lower_cmd):
        return {
            "is_safe": False,
            "findings": ["destructive_sql_command"],
            "commands": [cmd.split()[0].lower()],
            "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
        }

    # Split pipelines and chained commands safely
    # First inspect pipe sequences for download-and-execute (`curl ... | sh`)
    pipe_segments = [s.strip() for s in re.split(r"\|\|", cmd)]
    if "|" in cmd and "||" not in cmd:
        pipes = [_tokenize_shell_segment(p) for p in cmd.split("|")]
        pipe_cmds = [os.path.basename(p[0]).lower() for p in pipes if p]
        if any(c in ("curl", "wget") for c in pipe_cmds[:-1]) and any(
            c in ("sh", "bash", "zsh", "dash", "pwsh", "powershell", "python", "python3")
            for c in pipe_cmds[1:]
        ):
            return {
                "is_safe": False,
                "findings": ["remote_code_execution_pipe"],
                "commands": pipe_cmds,
                "risk_class": "IRREVERSIBLE_DESTRUCTIVE",
            }

    subcommands = re.split(r"(?:&&|\|\||;|\|)", cmd)
    findings: List[str] = []
    parsed_cmds: List[str] = []

    for sub in subcommands:
        tokens = _tokenize_shell_segment(sub)
        if not tokens:
            continue
        exe = os.path.basename(tokens[0]).lower().removesuffix(".exe").removesuffix(".com")
        parsed_cmds.append(exe)
        args = tokens[1:]
        args_lower = [a.lower() for a in args]

        if exe == "rm":
            has_recursive = any(
                a == "--recursive" or (a.startswith("-") and not a.startswith("--") and ("r" in a or "R" in a))
                for a in args
            )
            dangerous_targets = {"/", "/*", "~", "~/", "$home", "*", "..", "c:\\", "c:/"}
            targets_danger = any(
                a in dangerous_targets or a.startswith("~/") or a.startswith("/dev/") or a.startswith("/etc")
                for a in args_lower
                if not a.startswith("-")
            )
            if has_recursive and targets_danger:
                findings.append("destructive_rm_rf")

        elif exe == "dd":
            if any(a.startswith("of=/dev/") or a.startswith("of=\\\\.\\") for a in args_lower):
                findings.append("destructive_raw_disk_write_dd")

        elif exe in {"mkfs", "fdisk", "parted", "wipefs", "diskpart"} or exe.startswith("mkfs."):
            findings.append(f"destructive_disk_format:{exe}")

        elif exe == "format":
            # Only trigger when the command ITSELF is `format`, never `git log --format` or `clang-format`
            findings.append("destructive_disk_format_cmd")

        elif exe in {"del", "erase", "rmdir", "rd"}:
            has_force_rec = any(a in {"/s", "/q", "/f"} for a in args_lower)
            has_root = any(a.startswith(("c:\\", "c:/", "/")) for a in args_lower)
            if has_force_rec or has_root:
                findings.append(f"destructive_delete:{exe}")

        elif exe == "chmod":
            has_777 = any("777" in a or "a+rwx" in a for a in args_lower)
            has_root_or_rec = any(a in {"/", "/*", "-r", "--recursive"} for a in args_lower)
            if has_777 and has_root_or_rec:
                findings.append("dangerous_chmod_777")

    is_safe = len(findings) == 0
    risk_class = "IRREVERSIBLE_DESTRUCTIVE" if not is_safe else (
        "READ_ONLY" if all(c in SAFE_SHELL_COMMANDS or (c == "git" and len(cmd.split()) > 1 and cmd.split()[1].lower() in {"status", "log", "diff", "branch", "show"}) for c in parsed_cmds) else "EXTERNAL_ACTION"
    )
    return {
        "is_safe": is_safe,
        "findings": findings,
        "commands": parsed_cmds,
        "risk_class": risk_class,
    }


def decide(
    input_data: Any,
    context: Optional[Dict[str, Any]] = None,
    policy: Optional[PolicyConfig] = None,
) -> Dict[str, Any]:
    """Unified 4-Layer Decision Contract.

    Args:
        input_data: Either a tool/action dict (`{"tool": ..., "path": ..., "diff": ..., "cmd": ...}`)
            or a plain text prompt/query string.
        context: Optional metadata (risk overrides, domain hints, etc.).
        policy: Optional PolicyConfig instance.

    Returns:
        Dictionary adhering to:
        `{decision, label, prediction_set, reason_code, layer, model_version, rules_version, latency_ms}`
    """
    t0 = time.perf_counter_ns()
    ctx = context or {}
    cfg = policy or DEFAULT_POLICY

    if isinstance(input_data, dict):
        tool_name = str(input_data.get("tool") or input_data.get("action") or "")
        target_path = input_data.get("path") or input_data.get("file_path")
        diff_text = input_data.get("diff") or input_data.get("payload_diff")
        cmd_str = input_data.get("cmd") or input_data.get("command") or (target_path if tool_name in ("run_command", "execute_shell") else None)
        query_text = str(input_data.get("query") or input_data.get("task_prompt") or "")
    else:
        tool_name = str(ctx.get("tool") or "")
        target_path = ctx.get("path")
        diff_text = ctx.get("diff")
        cmd_str = ctx.get("cmd")
        query_text = str(input_data or "")

    def _finish(decision: str, label: str, pred_set: List[str], reason_code: str, layer: int) -> Dict[str, Any]:
        latency_ms = round((time.perf_counter_ns() - t0) / 1_000_000.0, 4)
        return {
            "decision": decision,
            "label": label,
            "prediction_set": pred_set,
            "reason_code": reason_code,
            "layer": layer,
            "model_version": DEFAULT_MODEL_VERSION,
            "rules_version": RULES_VERSION,
            "latency_ms": latency_ms,
        }

    # --- Layer 0: Policy & Permissions ---
    l0 = evaluate_layer0_policy(tool_name, target_path, cfg)
    if l0["decision"] == "deny":
        return _finish("deny", "blocked_policy", ["deny"], l0["reason_code"], 0)
    if l0["decision"] == "escalate":
        return _finish("escalate", "requires_confirmation", ["escalate", "deny"], l0["reason_code"], 0)

    # --- Layer 1: Structural AST & Shell Parsers ---
    if cmd_str:
        sh_res = analyze_shell_command(cmd_str)
        if not sh_res["is_safe"]:
            return _finish("deny", sh_res["findings"][0], ["deny"], "layer1_shell_parser_veto", 1)

    if diff_text is not None:
        diff_res = analyze_code_diff(target_path, diff_text)
        if not diff_res["syntax_valid"]:
            return _finish("deny", "syntax_error", ["deny"], "layer1_ast_syntax_error", 1)
        if not diff_res["security_clean"]:
            return _finish("deny", diff_res["findings"][0], ["deny"], "layer1_ast_security_veto", 1)
        if diff_res["line_count"] > cfg.max_diff_lines:
            return _finish("escalate", "high_blast_radius", ["allow", "deny"], "layer1_diff_scale_exceeded", 1)

    # If pure tool invocation without free-text ambiguity and passed Layer 0 & Layer 1:
    if tool_name and not query_text:
        return _finish("allow", tool_name, [tool_name], "layer1_structural_verified", 1)

    # --- Layer 2: Calibrated Soft Classifier & Conformal Prediction Set ---
    if query_text:
        try:
            from .tool_routing_moe import get_tool_routing_moe
            moe = get_tool_routing_moe()
            moe_res = moe.route(query_text)
            if not moe_res.conformal_passed:
                return _finish(
                    "escalate",
                    moe_res.selected_tool,
                    [moe_res.selected_tool, moe_res.secondary_tool],
                    "layer2_conformal_abstention_ood",
                    2,
                )
            return _finish(
                "allow",
                moe_res.selected_tool,
                [moe_res.selected_tool],
                "layer2_conformal_singleton",
                2,
            )
        except Exception as exc:
            # Fail safe: escalate to Layer 3 LLM rather than returning a fake constant
            return _finish("escalate", "unknown", [], f"layer2_unavailable:{type(exc).__name__}", 3)

    # --- Layer 3: Fallback Escalation ---
    return _finish("escalate", "ambiguous", [], "layer3_default_escalation", 3)
