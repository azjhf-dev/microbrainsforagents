"""Regression tests pinning all 7 findings from Fable's technical review.

Verifies:
1. ContinualLearningPipeline uses persistent `W_embed_anchor`, non-zero first-step EWC
   regularization, empirical diagonal Fisher over Hopfield rules, and strict set coverage.
2. WeightSpaceVSA.find_nearest_domains() ranks by actual projected weights rather than domain names.
3. AnalyticalMicroTransformer.infer_conformal(..., alpha=...) honors caller alpha dynamically.
4. CriticalEnsemble.evaluate_with_cascade() never returns AUTHORIZED / llm_invoked=True without an LLM.
5. Secret leak detection (`secret_leak_pattern_found`) & interface hazard rejection in CriticalEnsemble.
6. Colony SWE benchmark has zero hardcoded OOD stems (`эльфийск`, `астральн`, `martian`, `weather`).
7. Shipped WeightSpaceVSA & model bundles load cleanly with `allow_pickle=False`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import aetheris_microbrains as mb
from aetheris_microbrains.core.analytical_micro_transformer import AnalyticalMicroTransformer
from aetheris_microbrains.core.continual_learning_pipeline import ContinualLearningPipeline
from aetheris_microbrains.core.critical_ensemble_abstention import CriticalEnsemble
from aetheris_microbrains.core.layered_decision_engine import analyze_code_diff
from aetheris_microbrains.core.weight_space_vsa import WeightSpaceVSA
from aetheris_microbrains.tests import test_colony_swe_shadow as colony_bench

PKG_DIR = Path(__file__).resolve().parents[1]
BRAINS_DIR = PKG_DIR / "memory" / "store" / "neuro_brains"
VSA_PATH = PKG_DIR / "memory" / "store" / "weight_space_vsa.npz"


def test_fable_1_continual_learning_ewc_anchor_and_empirical_fisher(tmp_path: Path):
    for suffix in ("_weights.npz", "_meta.json"):
        shutil.copy2(
            BRAINS_DIR / f"neuro_code_edit_verifier{suffix}",
            tmp_path / f"neuro_code_edit_verifier{suffix}",
        )

    brain = AnalyticalMicroTransformer.load("neuro_code_edit_verifier", tmp_path)
    assert brain is not None
    assert hasattr(brain, "W_embed_anchor")
    assert brain.W_embed_anchor.shape == brain.W_embed.shape

    pipeline = ContinualLearningPipeline(store_dir=tmp_path, audit_log_path=tmp_path / "audit.json", ewc_lambda=0.5)

    # Empirical Fisher over Hopfield rules must differ significantly from uniform 0.1*w^2 proxy
    emp_fisher = pipeline.compute_fisher_information({"W_embed": brain.W_embed}, brain=brain)["W_embed"]
    proxy_fisher = np.maximum(1e-4, np.square(brain.W_embed) * 0.1).astype(np.float32)
    assert not np.allclose(emp_fisher, proxy_fisher), "Fisher fell back to parameter-magnitude proxy!"
    assert float(np.max(emp_fisher)) > 2e-4
    # Active Hopfield token rows must carry strictly greater Fisher importance than unused tail rows
    active_mean = float(np.mean(emp_fisher[: len(brain.vocab)]))
    idle_mean = float(np.mean(emp_fisher[len(brain.vocab) + 1 :]))
    assert active_mean > idle_mean

    # Single update on an existing in-vocab token must yield non-zero EWC penalty immediately
    res = pipeline.apply_plasticity_feedback(
        "code_edit_verifier",
        ["syntax_tree_malformed"],
        learning_rate=0.05,
        promote=False,
    )
    assert res.status == "success"
    assert res.ewc_penalty > 0.0, "EWC penalty must be strictly positive even on the first token step!"
    assert res.weights_delta_l2 > 0.0

    # Strict set-membership coverage: target NOT in prediction_set must count as miss even if top_candidate matches
    fake_holdout = [(["syntax_tree_valid"], "__definitely_not_in_set__")]
    assert pipeline._measure_holdout_coverage(brain, fake_holdout) == 0.0


def test_fable_2_vsa_find_nearest_domains_uses_actual_weights():
    vsa = WeightSpaceVSA(dimension=1024, seed=99)
    rng = np.random.default_rng(7)
    base_w = {"W_embed": rng.normal(size=(32, 32)).astype(np.float32)}
    # Domain B has almost identical weights to Domain A, despite completely unrelated name
    near_w = {"W_embed": (base_w["W_embed"] + 0.02 * rng.normal(size=(32, 32))).astype(np.float32)}
    far_w = {"W_embed": rng.normal(scale=2.0, size=(32, 32)).astype(np.float32)}

    vsa.store_brain_weights("alpha_domain", base_w)
    vsa.store_brain_weights("unrelated_name_clone", near_w)
    vsa.store_brain_weights("alpha_domain_orthogonal", far_w)

    nearest = vsa.find_nearest_domains("alpha_domain", top_k=2)
    assert len(nearest) == 2
    assert nearest[0]["domain"] == "unrelated_name_clone"
    assert nearest[0]["similarity"] > 0.85


def test_fable_3_infer_conformal_respects_caller_alpha():
    brain = AnalyticalMicroTransformer.load("neuro_fast_intent_classifier", BRAINS_DIR)
    assert brain is not None and len(brain.conformal_scores) >= 20

    res_strict = brain.infer_conformal(["fix"], alpha=0.01)
    res_loose = brain.infer_conformal(["fix"], alpha=0.40)
    assert res_strict["alpha"] == pytest.approx(0.01)
    assert res_loose["alpha"] == pytest.approx(0.40)
    assert res_strict["conformal_quantile"] >= res_loose["conformal_quantile"]
    assert res_strict["conformal_quantile"] != brain.conformal_quantile or res_loose["conformal_quantile"] != brain.conformal_quantile


def test_fable_4_escalate_to_llm_honesty_without_llm_callback():
    ensemble = CriticalEnsemble()
    # 550-line benign diff triggers patch_quality_predictor DENY (1 DENY, 3 ALLOW -> ABSTAIN_DELEGATE_TO_LLM)
    big_diff = "\n".join(f"+val_{i} = {i}" for i in range(520))

    no_llm = ensemble.evaluate_with_cascade("edit_file", "src/data_gen.py", big_diff)
    assert no_llm["final_verdict"] != "AUTHORIZED"
    assert no_llm["final_verdict"] == "ESCALATED_TO_LLM"
    assert no_llm["llm_invoked"] is False
    assert no_llm["requires_external_llm"] is True

    # Force pure uncertainty escalation on benign small edit via low variance abstention
    called_with = []

    def _fake_llm(payload: dict) -> dict:
        called_with.append(payload)
        return {"verdict": "AUTHORIZED_BY_LLM"}

    # Missing model forces estimator abstention -> cascade passes tier1-4 -> invokes callback
    orig_loader = ensemble._get_model
    ensemble._models_cache["neuro_patch_quality_predictor"] = None  # trigger model_missing abstain
    ensemble._get_model = lambda n: None if n == "neuro_patch_quality_predictor" else orig_loader(n)
    with_llm = ensemble.evaluate_with_cascade("edit_file", "src/clean.py", "def ok():\n    return 1\n", llm_callback=_fake_llm)
    assert len(called_with) == 1
    assert with_llm["llm_invoked"] is True
    assert with_llm["final_verdict"] == "AUTHORIZED_BY_LLM"


def test_fable_5_secret_leak_detection_and_interface_hazard_ensemble():
    secret_diff = (
        "--- a/src/config.py\n"
        "+++ b/src/config.py\n"
        "@@ -1,1 +1,2 @@\n"
        '+OPENAI_KEY = "sk-proj-1234567890abcdef1234567890abcdef"\n'
    )
    analysis = analyze_code_diff("src/config.py", secret_diff)
    assert analysis["security_clean"] is False
    assert "secret_leak_pattern_found" in analysis["findings"]

    ensemble = CriticalEnsemble()
    consensus = ensemble.evaluate_critical_action("edit_file", "src/config.py", secret_diff)
    assert consensus.status == "CONSENSUS_REJECTED"
    v_vote = next(v for v in consensus.votes if v.estimator_name == "code_edit_verifier")
    assert v_vote.verdict == "DENY"

    # Breaking signature removal must not be rubber-stamped by CriticalEnsemble
    sig_break_diff = (
        "--- a/src/api.py\n"
        "+++ b/src/api.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def public_handler(req, ctx, timeout=5):\n"
        "+def public_handler(req):\n"
        "     return req\n"
    )
    sig_res = ensemble.evaluate_critical_action("edit_file", "src/api.py", sig_break_diff)
    assert sig_res.status != "CONSENSUS_APPROVED"


def test_fable_6_colony_benchmark_has_no_hardcoded_ood_stems():
    hybrid = colony_bench.HybridMicrobrainWithConformalGate()
    for forbidden_ood_stem in ("астральн", "эльфийск", "философское", "philosophy", "martian", "weather", "погоду"):
        assert forbidden_ood_stem not in hybrid.critical_invariants
        assert forbidden_ood_stem not in colony_bench.PureKeywordBaseline.CRITICAL_KEYWORDS

    res = colony_bench.run_3way_ablation_benchmark()
    hybrid_metrics = res["3. Microbrain Hybrid (Weights + Conformal Gate)"]
    assert hybrid_metrics["missed_escalations"] == "0/50"
    assert float(hybrid_metrics["strict_accuracy"].split("(")[1].replace("%)", "")) >= 80.0


def test_fable_7_vsa_loads_with_allow_pickle_false():
    with np.load(VSA_PATH, allow_pickle=False) as raw:
        assert "weight_keys" in raw.files
        assert "weight_values" in raw.files

    loaded = WeightSpaceVSA.load(VSA_PATH)
    assert len(loaded.domain_weight_vectors) == 42
    nearest = loaded.find_nearest_domains("tool_routing_v2", top_k=3)
    assert len(nearest) == 3
    assert all(item["similarity"] > 0.0 for item in nearest)


def test_rm_rf_top_level_and_ensemble_route_ood_escalation():
    # Destructive `rm -rf` on any top-level/root path must be caught by Layer 1
    res = mb.decide({"tool": "run_command", "cmd": "rm -rf /ouroboros"})
    assert res["decision"] == "deny"

    ens = CriticalEnsemble().evaluate_critical_action("run_command", "rm -rf /ouroboros")
    assert ens.status == "CONSENSUS_REJECTED"

    # Ensemble route on completely unknown OOD tokens must abstain & escalate
    route_ood = mb.run_neuro_route(
        action="ensemble_route",
        tokens=["random_divergent_intent_xyz"],
        min_agreement=0.99,
    )
    assert route_ood["escalate_to_llm"] is True
    assert route_ood["fallback"] is True

