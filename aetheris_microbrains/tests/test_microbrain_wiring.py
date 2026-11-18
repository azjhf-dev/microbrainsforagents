"""Standalone integration tests for the microbrain tools and unified decide() contract."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aetheris_microbrains as mb
from aetheris_microbrains.tools.microbrain import (
    consult_domain_hint,
    microbrain_consult,
    microbrain_status,
)

IN_VOCAB = "syntax error in AST parser during code edit"
OOD_TEXT = "kubernetes deployment yaml terraform helm"


class TestConsult:
    def test_in_vocab_text_predicts_with_readme_keys(self):
        out = microbrain_consult(text=IN_VOCAB)
        assert out["success"] is True
        assert out["abstained"] is False
        assert out["selected_brain"] == "neuro_code_edit_verifier"
        assert isinstance(out["calibrated_confidence"], float)
        assert out["calibrated_confidence"] > 0.5
        assert out.get("predicted_token")

    def test_ood_text_abstains_with_domain_list(self):
        out = microbrain_consult(text=OOD_TEXT)
        assert out["success"] is True
        assert out["abstained"] is True
        assert out["domain"] is None
        assert out["selected_brain"] is None
        assert out["calibrated_confidence"] == 0.0
        assert "OOD coverage breach" in out["reason"]
        assert out["available_domains"], "honest abstention names available trained domains"

    def test_empty_input_is_an_argument_error(self):
        out = microbrain_consult(text="   ")
        assert out["success"] is False
        assert "no tokens" in out["error"]

    def test_disabled_env_is_honest_not_silent(self, monkeypatch):
        monkeypatch.setenv("AETHERIS_MICROBRAIN_TOOLS", "0")
        out = microbrain_consult(text=IN_VOCAB)
        assert out["success"] is False
        assert "AETHERIS_MICROBRAIN_TOOLS" in out["error"]


class TestStatus:
    def test_fleet_inventory_reports_calibration_honestly(self):
        out = microbrain_status()
        assert out["success"] is True
        assert out["count"] >= 10
        verifier = next(b for b in out["brains"] if b["domain"] == "code_edit_verifier")
        assert verifier["vocab_size"] >= 10
        assert verifier["conformal_basis"] in ("uncalibrated", "held_out", "in_sample_auto")


class TestHint:
    def test_in_vocab_subject_gets_hint(self):
        hint = consult_domain_hint(IN_VOCAB)
        assert "code_edit_verifier" in hint
        assert "microbrain_consult" in hint

    def test_ood_subject_gets_no_hint(self):
        assert consult_domain_hint(OOD_TEXT) == ""


class TestUnifiedDecideContract:
    def test_decide_contract_fields_and_layers(self):
        read_res = mb.decide({"tool": "read_file", "path": "src/app.py"})
        assert read_res["decision"] == "allow"
        assert read_res["layer"] == 1
        assert set(read_res.keys()) >= {
            "decision", "label", "prediction_set", "reason_code",
            "layer", "model_version", "rules_version", "latency_ms",
        }

        policy_veto = mb.decide({"tool": "write_file", "path": "memory/bible.md", "diff": "+x = 1\n"})
        assert policy_veto["decision"] == "deny"
        assert policy_veto["layer"] == 0

        escalate_ood = mb.decide("закажи такси до дома")
        assert escalate_ood["decision"] == "escalate"
        assert escalate_ood["layer"] == 2
