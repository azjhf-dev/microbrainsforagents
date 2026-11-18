# Aetheris Microbrains (`aetheris_microbrains`)

Standalone, portable suite of pure-NumPy analytical microbrains, a 4-layer structural AST/`shlex` pre-filter (`layered_decision_engine.py`), deterministic SHA-256 MoE tool routing, and held-out split-conformal abstention gates.

> **Architectural Honesty Note:** Microbrains are **not** SGD-trained deep LLMs. They combine **Layer 0 deterministic policy & risk classes**, **Layer 1 Python AST & `shlex` shell parsers**, **Layer 2 analytically compiled Hopfield/Attention matrices with SHA-256 embeddings and split-conformal calibration** ($n_{\text{cal}} = 22 \ge 19$ for $1 - \alpha = 95\%$ coverage), and **Layer 3 escalation** to a remote LLM or human confirmation when OOD or ambiguous (`|C_alpha| != 1`).

## Directory Structure

- `__init__.py` — Clean Python SDK entrypoint (does **not** mutate `sys.path` or shadow host modules) re-exporting the 4-layer decision engine (`decide`, `PolicyConfig`, `analyze_code_diff`, `analyze_shell_command`), core microbrain architectures, GPU/OpenCL acceleration, SOTA neural extensions, and agent tool wrappers.
- `core/` — Neural, structural parsing, and cognitive core modules (63 modules + `__init__.py`), including:
  - **4-Layer Structural Decision Engine**: `layered_decision_engine.py` (`decide()`, `PolicyConfig`, `evaluate_layer0_policy`, `is_protected_path`, `analyze_code_diff` AST visitor + `original_source` context, allowlist-first `analyze_shell_command` `shlex` tokenizer, risk-class fail-closed handling, `shadow_mode`, and JSONL decision journaling)
  - **OpenCL GPU Co-Processor**: `opencl_microbrain_accelerator.py` (PyOpenCL kernels with transparent NumPy CPU fallback)
  - **Microbrain Architectures**: `dendritic_tree_microbrain.py`, `kolmogorov_arnold_microbrain.py`, `linear_state_space_microbrain.py`, `microbrain_hypernetwork.py`, `liquid_state_reservoir.py`, `predictive_coding_engine.py`, `bitnet_ternary_engine.py`, `hyperdimensional_vsa_engine.py`
  - **Analytical Micro-Transformers & Nursery**: `analytical_micro_transformer.py` (Mondrian per-class quantiles, $\sqrt{\alpha(1-\alpha)/(n+2)}$ & $3/n$ bounds, SHA-256 `weights_sha256` verification on load), `brain_nursery.py`, `brain_moe_orchestrator.py`, `brain_merger.py`, `brain_cellular_fission.py`, `brain_plasticity_updater.py`, `brain_speculative_drafter.py`, `wsp_neural_surgeon.py`, `microbrain_server.py`
  - **Meta-Cognitive Layer**:
    - `causal_world_model.py`: Structural Causal Model (SCM) & Pearl $do$-calculus DAG.
    - `counterfactual_reasoner.py`: What-If lookahead pre-execution simulation with AST blast radius analysis.
    - `intent_persistence_tracker.py`: Trajectory-level Intent Drift Scoring (IDS) and dynamic context re-anchoring.
    - `metacognitive_calibrator.py`: Split-conformal action set coverage and epistemic/aleatoric uncertainty calibration.
    - `skill_lifecycle_manager.py`: Autonomous domain incubation, validation, degraded status, and fleet hygiene.
  - **SOTA Neural Extensions**:
    - `weight_space_vsa.py`: 2048-D Vector Symbolic Architecture / Hyperdimensional Computing associative memory over fleet model weights.
    - `tool_routing_moe.py`: Bilingual (RU/EN) SHA-256 Top-2 Mixture of Experts (MoE) router constrained to active `available_tools` plus MCP-Zero Top-$K$ narrowing (`narrow_tool_surface`) and OOD abstention.
    - `critical_ensemble_abstention.py`: Multi-brain consensus voting integrated with Layer 0/1 AST & path checks and selective conformal abstention.
    - `cascade_ensemble.py`: Four-tier cascading decision pipeline routed through Layer 0/1 structural AST/`shlex` checks (`ls`/`pwd` pass without false blocks).
    - `continual_learning_pipeline.py`: Staged candidate-gate EWC plasticity (`_candidates/` $\to$ `verify_and_promote_candidate` $\to$ `_rollback/` rollback).
    - `plasticity_cycle.py`: Automated scheduled plasticity candidate staging with measured rule-retention telemetry.
    - `microbrain_telemetry.py`: Nanosecond-precision inference telemetry with 10MB log rotation, ECE calculation, and health dashboard export.
    - `counterfactual_simulation_pipeline.py`: Pre-commit blast-radius gate backed by Layer 1 Python AST & hunk analysis.
- `evolution/` — Weight-Space Programming (WSP) & client modules (13 modules + `__init__.py`):
  - `wsp_engine.py`, `wsp_substrate.py`, `wsp_lenses.py`, `wsp_sae.py`, `wsp_merger.py`, `wsp_hebbian.py`, `wsp_sleep_consolidation.py`, `wsp_liquid_router.py`, `wsp_ab_probe.py`, `wsp_epistemic_oracle.py`, `wsp_sandbox_tracer.py`, `dsh_microbrain_client.py`, `opencl_microbrain_accelerator.py` (compatibility shim)
- `tools/` — Agent tool wrappers (`microbrain.py`, `microbrain_lab.py`, `neuro_brain_incubator.py`, `neuro_route.py`, `wsp_tensor_surgeon.py`, `conformal_collector.py`)
- `memory/` — Fleet registries (`fleet_lifecycle_registry.json`), weight space VSA store (`store/weight_space_vsa.npz`), analytically compiled & SHA-256 verified `store/neuro_brains/` fleet (87 files), and `opencl_crossover.json`
- `tests/` — Verification suites (15 `test_*.py` suites + `conftest.py`), including `test_layer_ablation_and_baselines.py`, `test_reviewer_probe_suite.py`, `test_adversarial_safety_suite.py`, `test_colony_swe_shadow.py`, and `test_ouroboros_sota_extensions.py`

## Exported SDK Symbols

| Category | Symbols |
| --- | --- |
| **4-Layer Decision Engine** | `decide`, `PolicyConfig`, `evaluate_layer0_policy`, `is_protected_path`, `analyze_code_diff`, `analyze_shell_command`, `record_decision_journal`, `get_recent_decision_journal`, `narrow_tool_surface` |
| **GPU / OpenCL Co-Processor** | `OpenCLMicroBrainAccelerator`, `get_opencl_accelerator` |
| **Microbrain Cores** | `DendriticTreeMicroBrain`, `KolmogorovArnoldMicroBrain`, `LinearStateSpaceMicroBrain`, `MicroBrainHypernetwork`, `LiquidStateReservoir`, `HierarchicalPredictiveCodingEngine` (`PredictiveCodingEngine`), `PredictiveCodingLayer`, `BitLinearLayer`, `BitNetQuantizer` (`BitNetTernaryEngine`), `HyperdimensionalVSAEngine` |
| **Nursery & Surgery** | `AnalyticalMicroTransformer`, `STORE_DIR`, `AutonomousBrainNursery` (`BrainNursery`), `get_brain_nursery`, `WSPNeuralSurgeon` |
| **Meta-Cognitive Layer** | `StructuralCausalModel` (`CausalWorldModel`), `get_causal_world_model`, `CausalVariable`, `CausalIntervention`, `CausalSimulationResult`, `CounterfactualResult`, `simulate_counterfactual`, `IntentPersistenceTracker`, `IntentDriftAssessment`, `MetacognitiveActionVerdict`, `CalibratedConfidence`, `calibrate_action_confidence`, `SkillLifecycleState`, `SkillTelemetry`, `FleetHygieneReport`, `SkillLifecycleManager`, `get_skill_lifecycle_manager` |
| **SOTA Neural Extensions** | `WeightSpaceVSA`, `VSAResult`, `get_weight_space_vsa`, `ToolRoutingMoE`, `MoERoutingDecision`, `get_tool_routing_moe`, `CriticalEnsemble`, `CriticalConsensusResult`, `EnsembleVote`, `get_critical_ensemble`, `CascadeEnsembleRouter`, `CascadeTierResult`, `CascadeEnsembleVerdict`, `get_cascade_ensemble_router`, `ContinualLearningPipeline`, `ContinualUpdateResult`, `PlasticityCycleScheduler`, `PlasticityCycleReport`, `get_plasticity_scheduler`, `MicrobrainTelemetry`, `TelemetryEvent`, `DashboardReport`, `get_microbrain_telemetry`, `CounterfactualSimulationPipeline`, `SimulationOutcome`, `get_counterfactual_pipeline` |
| **Agent Tools** | `microbrain_consult`, `microbrain_status`, `consult_domain_hint`, `run_microbrain_lab`, `run_neuro_incubator`, `run_neuro_route`, `run_wsp_surgeon`, `ConformalCalibrationReport`, `run_conformal_calibration_pipeline` |

## Quickstart Examples

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

### 1. Direct Python SDK Import & Split-Conformal Domain Consultation

Install in editable mode (`pip install -e .`) and import cleanly without mutating `sys.path`:

```python
import aetheris_microbrains as mb

# Inspect the pre-compiled & calibrated neuro-brains fleet
status = mb.microbrain_status()
print(f"Loaded domains: {status['count']}")

# Consult the best-matching brain with conformal calibration metadata
reply = mb.microbrain_consult(text="syntax error in AST parser during code edit")
print("Selected Brain:", reply["selected_brain"])
print("Calibrated Confidence:", reply["calibrated_confidence"])
print("Calibration Samples:", reply["n_calibration_samples"], "Valid:", reply["conformal_guarantee_valid"])
```

### 2. Deterministic Top-2 Mixture of Experts (MoE) Tool Routing

```python
import aetheris_microbrains as mb

moe = mb.get_tool_routing_moe()
decision = moe.route("inspect file contents and write safe patch to repository")
print("Selected Tool:", decision.selected_tool)   # -> repo_write (deterministic across runs)
print("Top Experts:", decision.top_experts)
print("Conformal Guarantee:", decision.conformal_passed)
```

### 3. Holographic Weight Space VSA Memory

```python
import aetheris_microbrains as mb

vsa = mb.get_weight_space_vsa()
nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
print("Nearest Domains in Weight Space:", nearest)
```

### 4. Pre-Commit Structural AST & Counterfactual Safety Gate

```python
import aetheris_microbrains as mb

sim = mb.get_counterfactual_pipeline()
outcome = sim.simulate_edit_impact(
    file_path="src/parser.py",
    diff_text="--- a/src/parser.py\n+++ b/src/parser.py\n@@ -1,2 +1,3 @@\n def parse():\n-    pass\n+    return 42\n"
)
print("Allowed:", outcome.allowed)
print("Status:", outcome.status)
print("Step Rollouts:", outcome.step_predictions)
```

### 5. Running Tests & the FastAPI Microbrain Server

```bash
pytest aetheris_microbrains/tests/test_reviewer_probe_suite.py -v
python -m aetheris_microbrains.core.microbrain_server
```
