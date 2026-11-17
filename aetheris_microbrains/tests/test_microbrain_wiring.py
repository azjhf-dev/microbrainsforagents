"""Regression tests for the microbrain fleet wiring (2026-11-15).

Live finding: the trained neuro_brains fleet was fully functional code with
zero live-path importers - no dispatcher entry, no journal events, no
evolution evidence, invisible to the proposal engine.  The wiring adds:

- standard tools microbrain_consult / microbrain_status (dispatcher registry)
- a plan-lane domain advisory, consumed by node_execution's executor context
- a microbrain_consult journal event per consult (evidence for the loop)
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.tools import microbrain as mb
from core.dispatcher import (
    STANDARD_TOOL_REGISTRY, get_tool_func, get_tool_schemas,
)


IN_VOCAB = "quasar pulsar redshift"
OOD_TEXT = "kubernetes deployment yaml terraform"


@pytest.fixture()
def captured_events(monkeypatch):
    events = []
    monkeypatch.setattr("memory.journal.log_event",
                        lambda *a, **k: events.append((a, k)))
    return events


class TestRegistry:
    def test_both_tools_registered_with_schemas(self):
        for name in ("microbrain_consult", "microbrain_status"):
            entry = STANDARD_TOOL_REGISTRY.get(name)
            assert entry, f"{name} missing from STANDARD_TOOL_REGISTRY"
            schema = entry["schema"]
            assert schema["name"] == name
            assert schema["parameters"]["type"] == "object"
            assert callable(entry["func"])
            assert get_tool_func(name) is entry["func"]

    def test_schemas_reach_executor_surface(self):
        schemas = get_tool_schemas()
        names = {s.get("name") for s in schemas if isinstance(s, dict)}
        assert {"microbrain_consult", "microbrain_status"} <= names


class TestConsult:
    def test_in_vocab_text_predicts_and_journals(self, captured_events):
        out = mb.microbrain_consult(text=IN_VOCAB)
        assert out["success"] is True
        assert out["abstained"] is False
        assert out["domain"] == "astrophysics"
        assert out["confidence"] > 0.5
        assert out["predicted_token"] in {"quasar", "pulsar", "redshift", "galaxy"}
        kinds = [a[0][0] for a in captured_events if a[0]]
        assert "microbrain_consult" in kinds

    def test_ood_text_abstains_with_domain_list(self, captured_events):
        out = mb.microbrain_consult(text=OOD_TEXT)
        assert out["success"] is True
        assert out["abstained"] is True
        assert out["domain"] is None
        assert "no trained domain shares vocabulary" in out["reason"]
        assert out["available_domains"], "honest abstention names what IS trained"

    def test_empty_input_is_an_argument_error(self):
        out = mb.microbrain_consult(text="   ")
        assert out["success"] is False
        assert "no tokens" in out["error"]

    def test_disabled_env_is_honest_not_silent(self, monkeypatch):
        monkeypatch.setenv("AETHERIS_MICROBRAIN_TOOLS", "0")
        out = mb.microbrain_consult(text=IN_VOCAB)
        assert out["success"] is False
        assert "AETHERIS_MICROBRAIN_TOOLS" in out["error"]

    def test_explicit_tokens_path(self):
        out = mb.microbrain_consult(tokens=["redshift", "galaxy"])
        assert out["success"] is True
        assert out["domain"] == "astrophysics"


class TestStatus:
    def test_fleet_inventory_reports_calibration_honestly(self):
        out = mb.microbrain_status()
        assert out["success"] is True
        assert out["count"] >= 1
        astro = next(b for b in out["brains"] if b["domain"] == "astrophysics")
        assert astro["vocab_size"] >= 4
        assert astro["conformal_basis"] in ("uncalibrated", "conformal")


class TestHint:
    def test_in_vocab_subject_gets_hint(self):
        hint = mb.consult_domain_hint("Study %s behavior" % IN_VOCAB)
        assert "astrophysics" in hint
        assert "microbrain_consult" in hint

    def test_ood_subject_gets_no_hint(self):
        assert mb.consult_domain_hint(OOD_TEXT) == ""


class TestPlanWiring:
    def _fake_kernel_lane(self):
        class _Lane:
            kernel = None
        return _Lane()

    def test_node_plan_surfaces_microbrain_hint(self, captured_events):
        from core.plan import PlanLane
        lane = PlanLane(self._fake_kernel_lane())
        run_state = {"current_goal": "BUILD: %s survey pipeline" % IN_VOCAB}
        lane.node_plan(run_state)
        assert "microbrain_hint" in run_state
        assert "astrophysics" in run_state["microbrain_hint"]
        kinds = [a[0][0] for a in captured_events if a[0]]
        assert "microbrain_route" in kinds

    def test_node_plan_ood_goal_has_no_hint(self, captured_events):
        from core.plan import PlanLane
        lane = PlanLane(self._fake_kernel_lane())
        run_state = {"current_goal": "BUILD: %s" % OOD_TEXT}
        lane.node_plan(run_state)
        assert "microbrain_hint" not in run_state


class TestExecutorConsumption:
    def test_executor_context_carries_the_advisory(self):
        from core.node_execution import node_execution

        seen = {}

        class _Mind:
            def generate_reply(self, messages, model=None):
                seen["messages"] = messages
                return ""

        class _Kernel:
            mind = _Mind()
            loop_detector = None

        class _Lane:
            kernel = _Kernel()

        run_state = {
            "plan": ["TOOL: pick a capability for the subject"],
            "step_index": 0,
            "current_goal": "BUILD: %s survey" % IN_VOCAB,
            "microbrain_hint": "trained domain 'astrophysics' matches this subject",
        }
        node_execution(_Lane(), run_state)
        content = seen["messages"][0]["content"]
        assert "[microbrain advisory:" in content
        assert "astrophysics" in content

    def test_executor_without_hint_is_unchanged(self):
        from core.node_execution import node_execution

        seen = {}

        class _Mind:
            def generate_reply(self, messages, model=None):
                seen["messages"] = messages
                return ""

        class _Kernel:
            mind = _Mind()
            loop_detector = None

        class _Lane:
            kernel = _Kernel()

        run_state = {
            "plan": ["TOOL: pick a capability"],
            "step_index": 0,
            "current_goal": "BUILD: %s survey" % IN_VOCAB,
        }
        node_execution(_Lane(), run_state)
        assert "[microbrain advisory:" not in seen["messages"][0]["content"]
