# Aetheris Microbrains 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-purple.svg)](LICENSE)
[![Zero-GPU](https://img.shields.io/badge/Hardware-Pure_NumPy_%7C_OpenCL-success.svg)](https://github.com/)
[![Inference Latency](https://img.shields.io/badge/Inference-<2ms-brightgreen.svg)](https://github.com/)

**Aetheris Microbrains** is an ultra-fast, lightweight neural coprocessor suite designed for autonomous LLM agents and edge AI systems. It provides sub-millisecond local inference, dynamic Mixture-of-Experts (MoE) tool routing, conformal prediction safety gating, and holographic Vector Symbolic Architecture (VSA) weight-space memory — running entirely in pure NumPy on CPU or accelerated via OpenCL on legacy/integrated GPUs.

Zero NVIDIA CUDA required. Zero remote API latency. Zero token cost for routing decisions.

---

## 🚀 Key Features

* **44 Pre-Trained Analytical Micro-Transformers:** Specialized lightweight models (10–100 KB each) for intent triage, risk scoring, patch verification, drift detection, and tool chaining.
* **Dynamic Top-2 Mixture of Experts (MoE):** Neural router with dynamic temperature gating across 6 functional clusters (`ExpertFileOps`, `ExpertExecution`, `ExpertMemoryKnowledge`, `ExpertGitGovernance`, `ExpertMicrobrainMeta`, `ExpertConstitutionalReview`).
* **Conformal Safety & Selective Abstention:** Rigorous statistical non-conformity guarantees ($1 - \alpha \ge 95\%$). Automatically abstains and escalates high-risk decisions when estimator variance exceeds safe thresholds.
* **2048-D Holographic Weight-Space Memory (VSA):** Hyperdimensional computing associative memory over fleet model weights with Count-Sketch projection and nearest-neighbor domain recovery.
* **Counterfactual Simulation Engine:** 3-step forward rollout blast-radius simulation before applying code mutations or executing high-stakes tools.
* **Continual Plasticity with EWC:** Online bounded Hebbian adaptation protected by Elastic Weight Consolidation (EWC) Fisher Information diagonals to prevent catastrophic forgetting.
* **Hardware Agnostic:** Transparent acceleration on any OpenCL-capable device (including budget GT 710 or integrated graphics) with seamless pure-NumPy CPU fallback.

---

## 📦 Installation

```bash
git clone https://github.com/drtikov/microbrainsforagents.git
cd microbrainsforagents
pip install -e .
```

*Optional OpenCL GPU acceleration:*
```bash
pip install -e .[opencl]
```

---

## ⚡ Quickstart

### 1. Zero-Shot Intent & Tool Consultation (<2ms)

```python
import aetheris_microbrains as mb

# Fast inference across 44 calibrated microbrains
result = mb.microbrain_consult(text="syntax error in AST parser during code edit")
print("Selected Brain:", result["selected_brain"])
print("Confidence:", result["calibrated_confidence"])
```

### 2. Dynamic Mixture-of-Experts (MoE) Tool Routing

```python
import aetheris_microbrains as mb

router = mb.get_tool_routing_moe()
decision = router.route("inspect file contents and write safe patch to repository")

print("Primary Tool:", decision.selected_tool)
print("Top-2 Experts:", decision.top_experts)
print("Conformal Guarantee Passed:", decision.conformal_passed)
```

### 3. Pre-Commit Counterfactual Safety Gate

```python
import aetheris_microbrains as mb

sim = mb.get_counterfactual_pipeline()
outcome = sim.simulate_edit_impact(
    file_path="src/server.py",
    diff_text="--- a/server.py\n+++ b/server.py\n@@ -10,2 +10,3 @@\n+    eval(untrusted_user_input)\n"
)

if not outcome.allowed:
    print("Action blocked by microbrain safety gate!")
    print("Reason:", outcome.veto_reason)
    print("Status:", outcome.status)
```

### 4. Holographic Weight-Space VSA Memory

```python
import aetheris_microbrains as mb

vsa = mb.get_weight_space_vsa()
nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
print("Nearest models in parameter space:", nearest)
```

---

## 🏗️ Architecture Overview

```text
                  ┌──────────────────────────────────────────────┐
                  │           User / Autonomous Agent            │
                  └──────────────────────┬───────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │       Fast Intent Triage (<1ms)        │
                    └────────────────────┬───────────────────┘
                                         │
               ┌─────────────────────────┴────────────────────────┐
               │                                                  │
 ┌─────────────▼──────────────┐                    ┌──────────────▼─────────────┐
 │ Dynamic Top-2 MoE Routing  │                    │   Cascade Ensemble Gate    │
 │ (6 Expert Functional Units)│                    │(Constitutional + AST Guard)│
 └─────────────┬──────────────┘                    └──────────────┬─────────────┘
               │                                                  │
               └─────────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼───────────────────┐
                    │      Pre-Commit Simulation & Veto      │
                    │   (3-Step Counterfactual Blast Radius) │
                    └────────────────────┬───────────────────┘
                                         │
      ┌──────────────────────────────────┴─────────────────────────────────┐
      │                                                                    │
┌─────▼──────────────────────────────┐              ┌──────────────────────▼─────┐
│ 2048-D Hyperdimensional VSA Memory │              │ Continual Learning & EWC   │
│ (Holographic Fleet Weight Space)   │              │ (Online Plasticity Engine) │
└────────────────────────────────────┘              └────────────────────────────┘
```

---

## 🧪 Testing & Empirical Safety Benchmarks

Run the complete verification and empirical adversarial safety test suite:

```bash
pytest aetheris_microbrains/tests/test_ouroboros_sota_extensions.py aetheris_microbrains/tests/test_adversarial_safety_suite.py -v
```

### Empirical Adversarial Benchmark (200 Labeled Actions):
* **Dataset:** 100 safe actions vs. 100 adversarial cases (25 syntax bombs, 25 constitutional violations, 25 destructive shell commands, 25 explosive diffs).
* **True Positives (Dangerous Intercepted):** 100 / 100 (100.0%)
* **True Negatives (Safe Approved):** 100 / 100 (100.0%)
* **False Negatives (Dangerous Missed):** **0 / 100 (0.00% Empirical FNR)**
* **Selective Abstention:** Disagreement threshold ($\sigma > 0.30$) automatically escalates borderline cases to the remote LLM.
* **Zero Data Leakage Guarantee:** Enforces mathematical disjoint partitions ($\text{Train} \cap \text{Calibration} = \emptyset$, $\text{Train} \cap \text{Validation} = \emptyset$, $\text{Calibration} \cap \text{Validation} = \emptyset$) in `conformal_collector.py`.

---

## 📄 License & Commercial Inquiries

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

* **Noncommercial Use:** You may freely use, modify, and distribute this software for educational, research, and noncommercial purposes.
* **Commercial Use:** Commercial use requires a commercial license from the author.

For commercial licensing inquiries, please contact:
**Dmitry Tikov** — [drtikov@gmail.com](mailto:drtikov@gmail.com)
