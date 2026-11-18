# Aetheris Microbrains

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Latency](https://img.shields.io/badge/Prefilter-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** is a deterministic, layered local pre-filter and tool-routing library for autonomous LLM agents running in pure NumPy (with optional OpenCL acceleration).

> **Architectural Honesty Note:** Microbrains are **not** SGD-trained deep LLMs. They are **analytically compiled Weight-Space associative matrices** paired with deterministic **Layer 0 policy rules**, **Layer 1 structural AST & `shlex` shell parsers**, and **Layer 2 split-conformal calibration** ($n_{\text{cal}} \ge 20$, $1 - \alpha = 95\%$) that abstain and escalate ambiguous or out-of-distribution (OOD) inputs to a remote LLM.

---

## 🏗️ 4-Layer Decision Architecture

```text
Agent Input (tool call, diff, shell command, or intent query)
 │
 ├─ Layer 0: Deterministic Policy (code invariants & risk taxonomy)
 │    • Read-only vs. Reversible Write vs. External vs. Irreversible Destructive
 │    • Protected paths (memory/bible.md, constitution/, etc.) ── veto → deny / escalate
 │
 ├─ Layer 1: Structural Language & Shell Parsers (replaces substring matching)
 │    • Python AST visitor (dedents indented hunks, wraps except/elif fragments, supports file_content alias context, ignores comments)
 │    • Catches direct eval/exec, subprocess(shell=True), shutil.rmtree, pickle.loads, aliases, subscript (__builtins__/globals), encoded decode-to-exec, and getattr obfuscation
 │    • JSON / TypeScript / JS diffs parsed without false Python SyntaxErrors
 │    • Allowlist-first shell shlex tokenizer (`rm -rf`, `sh -c`, `python -c`, `dd of=/dev/...`, `curl | sh` deny; unrecognized cmds escalate; `ls`, `pwd`, `git log --format`, `clang-format` allow)
 │    • Risk-class fail-closed handling + structured decision journaling & `shadow_mode`
 │
 ├─ Layer 2: Soft Micro-Classifier & Split-Conformal Abstention
 │    • Deterministic SHA-256 bilingual (RU/EN) char-trigram + lexical routing, constrained to `available_tools` + MCP-Zero Top-K narrowing (`narrow_tool_surface`)
 │    • Held-out split-conformal & Mondrian per-class calibration (n_cal = 22 >= 19 for 1 - alpha = 95% coverage) + SHA-256 weight checksums
 │    • Singleton set (|C_alpha| == 1) → local allow; OOD / ambiguous (|C_alpha| != 1) → escalate
 │
 └─ Layer 3: Remote LLM / Human Confirmation (fallback handler)
```

---

## 📦 Installation

```bash
git clone https://github.com/drtikov/microbrainsforagents.git
cd microbrainsforagents
pip install -e .
```

Importing `aetheris_microbrains` is side-effect clean: it does **not** mutate `sys.path` or shadow top-level project modules (`core`, `tools`, `memory`, `evolution`).

*Optional OpenCL acceleration:*
```bash
pip install -e .[opencl]
```

---

## ⚡ Quickstart

### 0. Unified 4-Layer Decision Contract

```python
import aetheris_microbrains as mb

verdict = mb.decide({
    "tool": "edit_file",
    "path": "src/server.py",
    "diff": "def f():\n    os.system(cmd)\n",
})
print(verdict)
# {'decision': 'deny', 'label': 'dangerous_method:os.system', 'layer': 1, ...}
```

### 1. Microbrain Domain Consultation

```python
import aetheris_microbrains as mb

result = mb.microbrain_consult(text="syntax error in AST parser during code edit")
print("Selected Brain:", result["selected_brain"])
print("Confidence:", result["calibrated_confidence"])
print("Cal Samples:", result["n_calibration_samples"], "Valid Guarantee:", result["conformal_guarantee_valid"])
```

### 2. Deterministic Top-2 Mixture-of-Experts (MoE) Tool Routing

```python
import aetheris_microbrains as mb

router = mb.get_tool_routing_moe()
decision = router.route("inspect file contents and write safe patch to repository")

print("Primary Tool:", decision.selected_tool)       # -> repo_write (deterministic across processes)
print("Top-2 Experts:", decision.top_experts)
print("Conformal Passed:", decision.conformal_passed)
```

### 3. Pre-Commit Structural AST & Counterfactual Safety Gate

```python
import aetheris_microbrains as mb

sim = mb.get_counterfactual_pipeline()
outcome = sim.simulate_edit_impact(
    file_path="src/server.py",
    diff_text="--- a/server.py\n+++ b/server.py\n@@ -10,2 +10,3 @@\n+    eval(untrusted_user_input)\n"
)

if not outcome.allowed:
    print("Blocked:", outcome.status, outcome.veto_reason)
```

### 4. Holographic Weight-Space VSA Memory

```python
import aetheris_microbrains as mb

vsa = mb.get_weight_space_vsa()
nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
print("Nearest models in parameter space:", nearest)
```

### 5. Allowlist-First Shell Gate, File-Context AST & Shadow Mode

```python
import aetheris_microbrains as mb

# Allowlisted read-only shell commands pass; unrecognized commands escalate;
# destructive chains (`rm -f -r ~`, `sh -c 'rm -rf /'`, `ls $(whoami)`) are denied.
print(mb.decide({"tool": "execute_shell", "cmd": "ls -la"})["decision"])                    # -> "allow"
print(mb.decide({"tool": "execute_shell", "cmd": "custom_deploy_binary --prod"})["decision"]) # -> "escalate"

# Resolve aliases declared in enclosing `file_content` + log shadow-mode decisions
verdict = mb.decide(
    {"tool": "edit_file", "path": "src/runner.py", "diff": "def f(c):\n    _o.system(c)\n"},
    context={
        "file_content": "import os as _o\n",
        "shadow_mode": True,
        "journal_path": "memory/decision_journal.jsonl",
    },
)
print(verdict["decision"], verdict["shadow_decision"]) # -> "escalate" "deny"
```

### 6. Dynamic Tool Registry (`available_tools`) & MCP-Zero Top-K Narrowing

```python
import aetheris_microbrains as mb

router = mb.get_tool_routing_moe()

# Constrain selection strictly to the active agent mode's tool surface (bilingual RU/EN)
res = router.route(
    "прочитай файл и напиши патч для репозитория",
    available_tools=["edit_file", "memory_recall", "web_fetch"],
)
print("Selected within surface:", res.selected_tool) # -> "edit_file"

# MCP-Zero hierarchical Top-K catalog narrowing (escalates full catalog on OOD)
narrowed = mb.narrow_tool_surface("fix bug and write repository patch", top_k=4)
print(narrowed["mode"], narrowed["candidate_tools"], narrowed["reduction_ratio"])
```

### 7. Mondrian Conformal Quantiles, SHA-256 Checksums & Staged Plasticity

```python
import aetheris_microbrains as mb

# Load weights with automatic SHA-256 integrity verification (`verify_checksum=True`)
brain = mb.AnalyticalMicroTransformer.load("neuro_code_edit_verifier")
inf = brain.infer_conformal(["syntax_tree_valid", "ast_node_function_def"], alpha=0.05, mondrian=True)
print("Mondrian Quantiles:", inf["per_class_quantiles"], "Std Err:", inf["coverage_std_error"])

# Staged continual plasticity: writes to `_candidates/`, requires holdout gate, supports `_rollback/`
pipeline = mb.ContinualLearningPipeline()
staged = pipeline.apply_plasticity_feedback("code_edit_verifier", ["new_ast_pattern"], promote=False)
# prom = pipeline.verify_and_promote_candidate("code_edit_verifier", holdout_data, min_coverage=0.95)
# pipeline.rollback_domain("code_edit_verifier")
```

---

## 🎯 Scope & Design Principles

* **What Microbrains Are Built For:**
  - Sub-millisecond pre-flight structural filtering on diffs (AST syntax + call-graph sinks), shell commands (`shlex`), and policy risk boundaries before hitting remote LLMs.
  - Narrow tool routing across known tool registries with explicit abstention (`ood_coverage_breach` / `conformal_passed=False`) whenever input shares insufficient vocabulary.
* **What Microbrains Are NOT For:**
  - Replacing general LLM reasoning on arbitrary natural language tasks. Unknown or open-ended prompts deterministically abstain and escalate to Layer 3 (LLM).
  - Acting as a standalone OS sandbox: executable confirmation & environment isolation remain enforced at the runtime policy boundary.

---

## 🧪 Testing & Benchmarks

Run the verification suites (including Layer 0/1/2 ablation, bilingual baseline comparisons, AST parsing, allowlist `shlex` shell parsing, cross-process determinism, split/Mondrian conformal sample bounds, SHA-256 checksum verification, staged candidate rollback, and clean `sys.path`):

```bash
pytest aetheris_microbrains/tests/test_layer_ablation_and_baselines.py aetheris_microbrains/tests/test_reviewer_probe_suite.py aetheris_microbrains/tests/test_adversarial_safety_suite.py -v
```

### Split-Conformal & Mondrian Coverage Discipline (`conformal_collector.py`):
* Disjoint partitions per active domain: **30+ Train**, **22 Held-Out Calibration** ($n_{\text{cal}} \ge 1/\alpha - 1 = 19$ for $\alpha = 0.05$, giving 95% nominal split-conformal coverage), and **16 Held-Out Validation** samples with $\text{Train} \cap \text{Cal} = \text{Train} \cap \text{Val} = \text{Cal} \cap \text{Val} = \emptyset$.
* Honest uncertainty reporting: Mondrian `per_class_quantiles`, finite-sample coverage standard error $\sqrt{\alpha(1-\alpha)/(n+2)}$, Rule-of-3 upper risk bound $3/n$, SHA-256 `weights_sha256` verification on load, and `conformal_guarantee_valid = False` whenever calibration samples are insufficient.
* Staged candidate-gate continual learning (`_candidates/` $\to$ `verify_and_promote_candidate` $\to$ `_rollback/` snapshot) prevents self-poisoning loops.

### Layer Ablation, Baseline Comparison, Adversarial & Probe Suites:
* Layer 0 / 0+1 / 0+1+2 ablation, bilingual RU/EN baseline comparisons, MCP-Zero narrowing (`narrow_tool_surface`), risk-class fail-closed handling, `shadow_mode` journaling, Mondrian thresholds, checksum tamper rejection, and staged candidate rollback (`test_layer_ablation_and_baselines.py`).
* Synthetic adversarial checks (`test_adversarial_safety_suite.py`) and independent obfuscation/benign diff probes (`test_reviewer_probe_suite.py`) verify Layer 1–2 behavior across obfuscated Python/shell inputs and safe hunks.
* `test_colony_swe_shadow.py` compares 3 configurations (Pure Rules vs. Pure Neural Weights vs. Hybrid with Conformal OOD Gate) under strict target matching (`decision == target`) across 120 bilingual (60 RU / 60 EN) SWE prompts.

---

## 📄 License & Commercial Inquiries

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

* **Noncommercial Use:** You may freely use, modify, and distribute this software for educational, research, and noncommercial purposes.
* **Commercial Use:** Commercial use requires a commercial license from the author.

For commercial licensing inquiries, please contact:
**Dmitry Tikov** — [drtikov@gmail.com](mailto:drtikov@gmail.com)
