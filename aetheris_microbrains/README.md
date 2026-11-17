# Aetheris Microbrains (`aetheris_microbrains`)

Standalone, portable copy of the Aetheris CPU/GPU microbrain suite for reuse by external and peer agents. All original files in `i:\aether` remain intact and unmodified.

## Directory Structure

- `__init__.py` — Unified Python SDK entrypoint that configures `sys.path` and re-exports core microbrain architectures, GPU/OpenCL acceleration, SOTA Ouroboros neural extensions, and agent tool wrappers.
- `core/` — Neural & cognitive core modules (62 modules + `__init__.py`), including:
  - **OpenCL GPU Co-Processor**: `opencl_microbrain_accelerator.py` (16 PyOpenCL kernels with transparent NumPy CPU fallback)
  - **Microbrain Architectures**: `dendritic_tree_microbrain.py`, `kolmogorov_arnold_microbrain.py`, `linear_state_space_microbrain.py`, `microbrain_hypernetwork.py`, `liquid_state_reservoir.py`, `predictive_coding_engine.py`, `bitnet_ternary_engine.py`, `hyperdimensional_vsa_engine.py`
  - **Analytical Micro-Transformers & Nursery**: `analytical_micro_transformer.py`, `brain_nursery.py`, `brain_moe_orchestrator.py`, `brain_merger.py`, `brain_cellular_fission.py`, `brain_plasticity_updater.py`, `brain_speculative_drafter.py`, `wsp_neural_surgeon.py`, `microbrain_server.py`
  - **Meta-Cognitive Layer (Ported & Adapted from Ouroboros)**:
    - `causal_world_model.py`: Structural Causal Model (SCM) & Pearl $do$-calculus DAG.
    - `counterfactual_reasoner.py`: What-If lookahead pre-execution simulation with AST blast radius analysis.
    - `intent_persistence_tracker.py`: Trajectory-level Intent Drift Scoring (IDS) and dynamic context re-anchoring.
    - `metacognitive_calibrator.py`: Split-conformal action set coverage and epistemic/aleatoric uncertainty calibration.
    - `skill_lifecycle_manager.py`: Autonomous domain incubation, validation, degraded status, and fleet hygiene.
  - **SOTA Ouroboros Neural Extensions**:
    - `weight_space_vsa.py`: 2048-D Vector Symbolic Architecture / Hyperdimensional Computing associative memory over fleet model weights.
    - `tool_routing_moe.py`: Dynamic Top-2 Mixture of Experts (MoE) neural router with temperature gating across 6 specialized clusters.
    - `critical_ensemble_abstention.py`: Multi-brain consensus voting with selective prediction & conformal abstention (variance > 0.09 escalation).
    - `cascade_ensemble.py`: Four-tier cascading decision pipeline (Intent Triage -> Tool Selection -> Constitutional Gate -> Pre-Action Check).
    - `continual_learning_pipeline.py`: Online plasticity and continual learning with Elastic Weight Consolidation (EWC) Fisher Information protection.
    - `plasticity_cycle.py`: Automated scheduled plasticity cycles for continual adaptation without catastrophic forgetting.
    - `microbrain_telemetry.py`: Nanosecond-precision inference telemetry with 10MB log rotation, ECE calculation, and health dashboard export.
    - `counterfactual_simulation_pipeline.py`: Three-step forward rollout blast-radius simulation before applying code mutations.
- `evolution/` — Weight-Space Programming (WSP) & client modules (12 modules + `opencl_microbrain_accelerator.py` compatibility shim):
  - `wsp_engine.py`, `wsp_substrate.py`, `wsp_lenses.py`, `wsp_sae.py`, `wsp_merger.py`, `wsp_hebbian.py`, `wsp_sleep_consolidation.py`, `wsp_liquid_router.py`, `wsp_ab_probe.py`, `wsp_epistemic_oracle.py`, `wsp_sandbox_tracer.py`, `dsh_microbrain_client.py`
- `tools/` — Agent tool wrappers (`microbrain.py`, `microbrain_lab.py`, `neuro_brain_incubator.py`, `neuro_route.py`, `wsp_tensor_surgeon.py`, `conformal_collector.py`)
- `memory/` — Fleet registries (`fleet_lifecycle_registry.json`), weight space VSA store (`store/weight_space_vsa.npz`), and pre-trained `store/neuro_brains/` weight & metadata fleet (83 files) plus `opencl_crossover.json`
- `tests/` — Comprehensive test suites including `test_ouroboros_sota_extensions.py` (11 test files)

## Exported SDK Symbols

| Category | Symbols |
| --- | --- |
| **GPU / OpenCL Co-Processor** | `OpenCLMicroBrainAccelerator`, `get_opencl_accelerator` |
| **Microbrain Cores** | `DendriticTreeMicroBrain`, `KolmogorovArnoldMicroBrain`, `LinearStateSpaceMicroBrain`, `MicroBrainHypernetwork`, `LiquidStateReservoir`, `HierarchicalPredictiveCodingEngine` (`PredictiveCodingEngine`), `PredictiveCodingLayer`, `BitLinearLayer`, `BitNetQuantizer` (`BitNetTernaryEngine`), `HyperdimensionalVSAEngine` |
| **Nursery & Surgery** | `AnalyticalMicroTransformer`, `STORE_DIR`, `AutonomousBrainNursery` (`BrainNursery`), `get_brain_nursery`, `WSPNeuralSurgeon` |
| **Meta-Cognitive Layer** | `StructuralCausalModel` (`CausalWorldModel`), `get_causal_world_model`, `CausalVariable`, `CausalIntervention`, `CausalSimulationResult`, `CounterfactualResult`, `simulate_counterfactual`, `IntentPersistenceTracker`, `IntentDriftAssessment`, `MetacognitiveActionVerdict`, `CalibratedConfidence`, `calibrate_action_confidence`, `SkillLifecycleState`, `SkillTelemetry`, `FleetHygieneReport`, `SkillLifecycleManager`, `get_skill_lifecycle_manager` |
| **SOTA Neural Extensions** | `WeightSpaceVSA`, `VSAResult`, `get_weight_space_vsa`, `ToolRoutingMoE`, `MoERoutingDecision`, `get_tool_routing_moe`, `CriticalEnsemble`, `CriticalConsensusResult`, `EnsembleVote`, `get_critical_ensemble`, `CascadeEnsembleRouter`, `CascadeTierResult`, `CascadeEnsembleVerdict`, `get_cascade_ensemble_router`, `ContinualLearningPipeline`, `ContinualUpdateResult`, `PlasticityCycleScheduler`, `PlasticityCycleReport`, `get_plasticity_scheduler`, `MicrobrainTelemetry`, `TelemetryEvent`, `DashboardReport`, `get_microbrain_telemetry`, `CounterfactualSimulationPipeline`, `SimulationOutcome`, `get_counterfactual_pipeline` |
| **Agent Tools** | `microbrain_consult`, `microbrain_status`, `consult_domain_hint`, `run_microbrain_lab`, `run_neuro_incubator`, `run_neuro_route`, `run_wsp_surgeon`, `ConformalCalibrationReport`, `run_conformal_calibration_pipeline` |

## Quickstart Examples

### 1. Direct Python SDK Import (Works from Any Working Directory)

```python
import sys
sys.path.insert(0, r"i:\aether")

import aetheris_microbrains as mb

# Inspect the pre-trained neuro-brains fleet
status = mb.microbrain_status()
print(f"Loaded domains: {status['count']}")

# Consult the best-matching brain for a token query
reply = mb.microbrain_consult(text="synthesize new cognitive concepts from memory")
print(reply)
```

### 2. SOTA Top-2 Mixture of Experts (MoE) Dynamic Tool Routing

```python
import aetheris_microbrains as mb

moe = mb.get_tool_routing_moe()
decision = moe.route("inspect file contents and write safe patch")
print("Selected Tool:", decision.selected_tool)
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

### 4. Cascade Ensemble & Pre-Commit Counterfactual Simulation

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

### 5. Running the FastAPI Microbrain Server

From inside `i:\aether\aetheris_microbrains`:

```bash
python core/microbrain_server.py
```
