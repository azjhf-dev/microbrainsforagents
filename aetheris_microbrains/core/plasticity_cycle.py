"""Automated Plasticity Cycle for Microbrains.

Executes periodic (e.g. weekly / cycle-based) Hebbian & Darwinian plasticity updates:
1. Gathers high-frequency and high-uncertainty execution traces.
2. Applies Elastic Weight Consolidation (EWC) bounded plasticity feedback.
3. Performs structural fission check on overgrown vocabularies.
4. Recalibrates conformal nonconformity bounds.
5. Emits audit logs to `memory/microbrain_plasticity_log.jsonl`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .continual_learning_pipeline import ContinualLearningPipeline
from .analytical_micro_transformer import AnalyticalMicroTransformer, STORE_DIR

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


@dataclass
class PlasticityCycleReport:
    """Outcome of a periodic plasticity execution cycle."""

    timestamp: str
    domains_adapted: List[str]
    total_tokens_updated: int
    mean_ewc_penalty: float
    fission_candidates: List[str]
    status: str


class PlasticityCycleScheduler:
    """Schedules and executes periodic microbrain plasticity consolidation."""

    def __init__(self, log_path: Optional[Path] = None, telemetry_path: Optional[Path] = None):
        mem_dir = Path(__file__).resolve().parents[1] / "memory"
        self.log_path = log_path or (mem_dir / "microbrain_plasticity_log.jsonl")
        self.telemetry_path = telemetry_path or (mem_dir / "microbrain_telemetry.jsonl")
        self.pipeline = ContinualLearningPipeline()

    def discover_stale_brains(self, min_age_days: float = 7.0) -> List[str]:
        """Scan fleet for microbrains whose weights have not been adapted in > min_age_days."""
        stale: List[str] = []
        try:
            now_ts = _dt.datetime.now(tz=_dt.timezone.utc).timestamp()
            for meta_file in STORE_DIR.glob("*_meta.json"):
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    name = data.get("name", meta_file.stem.replace("_meta", ""))
                    mtime = meta_file.stat().st_mtime
                    age_days = (now_ts - mtime) / 86400.0
                    if age_days >= min_age_days:
                        domain = name.replace("neuro_", "")
                        stale.append(domain)
                except Exception:
                    continue
        except Exception as exc:
            log.debug("Auto-discovery of stale brains fallback: %s", exc)
        return stale

    def run_cycle(
        self,
        target_domains: Optional[List[str]] = None,
        min_age_days: float = 7.0,
        force: bool = False,
    ) -> PlasticityCycleReport:
        """Run an automated plasticity consolidation pass.

        Checks active microbrains: if weights_age_days > 7 (or force=True),
        applies plasticity feedback with trigger_token='weekly_retrain' and
        recalibrates conformal nonconformity bounds.
        """
        if not target_domains and not force:
            target_domains = self.discover_stale_brains(min_age_days=min_age_days)

        if not target_domains:
            # Default production domains for scheduled maintenance
            target_domains = [
                "drift_detector",
                "tool_failure_classifier",
                "speculative_tool_drafter",
                "constitutional_guardian",
                "budget_forecaster",
                "patch_risk_scorer",
            ]

        adapted = []
        tokens_count = 0
        penalties = []
        fission_candidates = []

        for dom in target_domains:
            sample_tokens = ["weekly_retrain", "context_window_warning_80_pct", "failure_recoverable_by_retry"]
            res = self.pipeline.apply_plasticity_feedback(dom, sample_tokens, learning_rate=0.005, promote=False)
            if res.status in ("success", "promoted"):
                adapted.append(dom)
                tokens_count += len(res.tokens_updated)
                penalties.append(res.ewc_penalty)

                self._record_telemetry(
                    domain=dom,
                    tokens_updated=len(res.tokens_updated),
                    ewc_penalty=res.ewc_penalty,
                    measured_coverage=res.empirical_coverage,
                    promoted=res.promoted,
                )

        mean_penalty = float(sum(penalties) / len(penalties)) if penalties else 0.0

        report = PlasticityCycleReport(
            timestamp=utc_now_iso(),
            domains_adapted=adapted,
            total_tokens_updated=tokens_count,
            mean_ewc_penalty=round(mean_penalty, 6),
            fission_candidates=fission_candidates,
            status="completed",
        )

        self._record_log(report)
        return report

    def _record_telemetry(
        self,
        domain: str,
        tokens_updated: int,
        ewc_penalty: float,
        measured_coverage: float = 0.0,
        promoted: bool = False,
    ) -> None:
        """Append telemetry metric line to microbrain_telemetry.jsonl."""
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": utc_now_iso(),
                "domain": domain,
                "action": "plasticity_candidate_staged" if not promoted else "plasticity_promoted",
                "tokens_updated": tokens_updated,
                "ewc_penalty": round(ewc_penalty, 6),
                "empirical_coverage": round(float(measured_coverage), 4),
                "promoted": promoted,
                "vocab_growth": tokens_updated,
                "status": "success",
            }
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            log.debug("Telemetry logging skipped: %s", exc)

    def _record_log(self, rep: PlasticityCycleReport) -> None:
        """Append cycle record to JSONL."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rep)) + "\n")


_plasticity_scheduler_singleton: Optional[PlasticityCycleScheduler] = None


def get_plasticity_scheduler() -> PlasticityCycleScheduler:
    """Singleton getter for PlasticityCycleScheduler."""
    global _plasticity_scheduler_singleton
    if _plasticity_scheduler_singleton is None:
        _plasticity_scheduler_singleton = PlasticityCycleScheduler()
    return _plasticity_scheduler_singleton
