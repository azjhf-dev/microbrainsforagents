"""Example: precheck shim for a DeepSeek-style (OpenAI-compatible) web harness.

Run: python examples/06_deepseek_harness_shim.py
"""

import json

from aetheris_microbrains.shims import DeepSeekHarnessPrecheck

gate = DeepSeekHarnessPrecheck(extra_shell_allowlist=["npm", "cargo", "python3"])

# A big registry of OpenAI-format tool schemas; only a few matter per task.
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": {"type": "object", "properties": {}}}}
    for n in ["run_shell", "apply_patch", "read_file", "write_file", "search_web", "open_url",
              "list_dir", "git_log", "git_diff", "create_issue", "comment_issue", "send_email",
              "query_db", "run_tests", "format_code", "rename_symbol", "extract_function", "grep"]
]

# 1. Inbound: cut schema tokens before the API call (falls back to full catalog on OOD).
res = gate.narrow_tool_schemas("git diff the staged change and show the commit log", TOOL_SCHEMAS, top_k=4)
print("narrow mode:", res["mode"], "| kept:", res["kept"], "| reduction:", res["reduction_ratio"])
res_ood = gate.narrow_tool_schemas("compose a symphony about distant galaxies", TOOL_SCHEMAS, top_k=4)
print("OOD mode:", res_ood["mode"], "| kept all:", len(res_ood["kept"]) == len(TOOL_SCHEMAS))

# 2. Outbound: vet every tool_call the model returns before executing it.
assistant_message = {
    "role": "assistant",
    "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "run_shell", "arguments": json.dumps({"command": "npm test"})}},
        {"id": "c2", "type": "function",
         "function": {"name": "run_shell", "arguments": json.dumps({"command": "dd if=/dev/zero of=/dev/sda"})}},
        {"id": "c3", "type": "function",
         "function": {"name": "apply_patch", "arguments": json.dumps({"path": "x.py", "diff": "def nope(:"})}},
        {"id": "c4", "type": "function",
         "function": {"name": "apply_patch", "arguments": json.dumps({"path": "x.py", "diff": "+x = 1"})}},
    ],
}

for call, verdict in gate.intercept_message(assistant_message):
    name = call["function"]["name"]
    print(f"{name:12s} -> {verdict.action:9s} {verdict.reason_code}")
