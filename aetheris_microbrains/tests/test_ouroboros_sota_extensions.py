"""Test suite for SOTA Ouroboros Neural Extensions ported to Aetheris Microbrains."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure both package parent (I:\aether) and package root (I:\aether\aetheris_microbrains) are on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_PKG_DIR = _TEST_DIR.parent
_AETHER_DIR = _PKG_DIR.parent

for p in [str(_PKG_DIR), str(_AETHER_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest

import aetheris_microbrains as mb


class TestWeightSpaceVSA:
    def test_vsa_singleton_and_codebook(self):
        vsa = mb.get_weight_space_vsa()
        assert vsa is not None
        assert vsa.dim == 2048
        # Should have domains in codebook
        assert len(vsa.domain_codebook) > 0 or len(vsa.stored_domains) >= 0

    def test_weight_projection_and_associative_query(self):
        vsa = mb.WeightSpaceVSA(dimension=1024, seed=123)
        dummy_weights = {
            "W_embed": np.random.randn(20, 32).astype(np.float32),
            "W_head": np.random.randn(32, 20).astype(np.float32),
        }
        vsa.store_brain_weights("synthetic_test_brain", dummy_weights)
        res = vsa.query_domain_by_weights(dummy_weights)
        assert res.domain == "synthetic_test_brain"
        assert res.similarity > 0.3
        assert res.reconstructed is True

    def test_find_nearest_domains(self):
        vsa = mb.get_weight_space_vsa()
        nearest = vsa.find_nearest_domains("tool_routing_v2", top_k=3)
        assert isinstance(nearest, list)


class TestToolRoutingMoE:
    def test_moe_routing_decision(self):
        router = mb.get_tool_routing_moe()
        assert router.d_model == 256
        res = router.route("please read file and write patch to repo")
        assert res.selected_tool in ["read_file", "write_file", "repo_read", "repo_write", "edit_file", "str_replace_editor"]
        assert len(res.top_experts) == 2
        w1 = res.top_experts[0][1]
        w2 = res.top_experts[1][1]
        assert round(w1 + w2, 2) == 1.00
        assert res.confidence > 0.0
        assert "per_expert_conformal" in res.to_dict()

    def test_git_routing_bias(self):
        router = mb.get_tool_routing_moe()
        res = router.route("git status and check branch commit seal")
        expert_names = [e[0] for e in res.top_experts]
        assert "ExpertGitGovernance" in expert_names


class TestCriticalEnsemble:
    def test_benign_action_approval(self):
        ensemble = mb.get_critical_ensemble()
        result = ensemble.evaluate_critical_action(
            action_name="read_file",
            target_path="src/components/button.tsx",
            payload_diff=None,
        )
        assert result.status == "CONSENSUS_APPROVED"
        assert result.abstention_triggered is False
        assert result.consensus_score >= 0.70

    def test_sacred_site_rejection(self):
        ensemble = mb.get_critical_ensemble()
        result = ensemble.evaluate_critical_action(
            action_name="repo_write",
            target_path="memory/aboriginal_kinship_charter.md",
            payload_diff="delete all contents",
        )
        assert result.status == "CONSENSUS_REJECTED" or result.abstention_triggered
        guardian_vote = next(v for v in result.votes if v.estimator_name == "constitutional_guardian")
        assert guardian_vote.verdict == "DENY"

    def test_syntax_risk_trigger(self):
        ensemble = mb.get_critical_ensemble()
        result = ensemble.evaluate_critical_action(
            action_name="edit_file",
            target_path="core/runtime.py",
            payload_diff="def foo():\n  eval(user_input)\n",
        )
        verifier_vote = next(v for v in result.votes if v.estimator_name == "code_edit_verifier")
        assert verifier_vote.verdict == "DENY"


class TestCascadeEnsemble:
    def test_cascade_pipeline_flow(self):
        router = mb.get_cascade_ensemble_router()
        verdict = router.evaluate_action_pipeline(
            task_prompt="fix bug in data parser",
            target_tool="repo_read",
            tool_args={"path": "src/parser.py"},
        )
        assert verdict.authorized is True
        assert len(verdict.tier_results) == 4
        assert verdict.tier_results[0].tier_name == "Tier1_IntentTriage"
        assert verdict.tier_results[1].tier_name == "Tier2_ToolRouting"
        assert verdict.tier_results[2].tier_name == "Tier3_ConstitutionalGate"
        assert verdict.tier_results[3].tier_name == "Tier4_SukiFailureCheck"

    def test_cascade_constitutional_block(self):
        router = mb.get_cascade_ensemble_router()
        verdict = router.evaluate_action_pipeline(
            task_prompt="modify holy scriptures",
            target_tool="write_file",
            tool_args={"path": "ancestral/aboriginal_ground.txt"},
        )
        assert verdict.authorized is False
        assert verdict.verdict == "BLOCKED_CONSTITUTIONAL"


class TestContinualLearningAndPlasticity:
    def test_fisher_information(self):
        pipeline = mb.ContinualLearningPipeline()
        dummy_weights = {"W_test": np.array([[1.0, -2.0], [0.5, 0.0]], dtype=np.float32)}
        fisher = pipeline.compute_fisher_information(dummy_weights)
        assert "W_test" in fisher
        assert (fisher["W_test"] > 0).all()

    def test_plasticity_cycle_execution(self):
        scheduler = mb.get_plasticity_scheduler()
        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler.log_path = Path(tmpdir) / "plasticity.jsonl"
            scheduler.telemetry_path = Path(tmpdir) / "telemetry.jsonl"
            rep = scheduler.run_cycle(target_domains=["tool_failure_classifier"], force=True)
            assert rep.status == "completed"
            assert "tool_failure_classifier" in rep.domains_adapted
            assert rep.total_tokens_updated >= 1


class TestMicrobrainTelemetry:
    def test_event_recording_and_ece(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_p = Path(tmpdir) / "telemetry.jsonl"
            dash_p = Path(tmpdir) / "dashboard.json"
            telemetry = mb.MicrobrainTelemetry(log_path=log_p, dashboard_path=dash_p)

            # Record several events
            for i in range(10):
                telemetry.record_event(
                    domain="test_domain",
                    backend="cpu",
                    latency_ms=1.5 + i * 0.1,
                    confidence=0.9,
                    verified_accurate=True,
                    conformal_quantile=0.01,
                    passed_conformal=True,
                )

            report = telemetry.generate_dashboard()
            assert report.total_events == 10
            assert report.status == "healthy"
            assert report.expected_calibration_error < 0.15
            assert "test_domain" in report.domain_breakdown


class TestCounterfactualSimulationPipeline:
    def test_safe_patch_simulation(self):
        sim = mb.get_counterfactual_pipeline()
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,2 +1,3 @@\n def hello():\n-    pass\n+    return 'world'\n"
        res = sim.simulate_edit_impact("src/module.py", diff)
        assert res.allowed is True
        assert res.status == "APPROVED_FOR_PATCH"
        assert len(res.step_predictions) == 3

    def test_risky_syntax_error_patch(self):
        sim = mb.get_counterfactual_pipeline()
        diff = "def bad(x:\n   return x\n"
        res = sim.simulate_edit_impact("src/broken.py", diff)
        assert res.allowed is False
        assert res.status in ["ABSTAIN_ESCALATE_LLM", "VETOED_CONSTITUTIONAL", "VETOED_HIGH_RISK_BLAST"]
        assert len(res.simulated_failure_modes) > 0
