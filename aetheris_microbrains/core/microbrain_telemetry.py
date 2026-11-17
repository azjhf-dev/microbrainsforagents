"""Microbrain Observability & Telemetry Engine.

Tracks calibration health, execution metrics, and distribution shift:
1. Conformal coverage tracking across domains over sliding time windows.
2. Expected Calibration Error (ECE):
      ECE = Sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
3. Inference latency telemetry comparing OpenCL GPU vs CPU.
4. Out-of-Distribution (OOD) / representation drift detection.
5. Persistent dashboard export to `memory/microbrain_telemetry_dashboard.json`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


@dataclass
class TelemetryEvent:
    """A single microbrain execution/routing telemetry record."""

    timestamp: str
    domain: str
    backend: str  # "opencl_gpu" | "cpu"
    latency_ms: float
    confidence: float
    verified_accurate: bool
    ood_score: float
    conformal_quantile: float
    passed_conformal: bool
    action: str = "infer"
    prediction: str = ""


@dataclass
class DashboardReport:
    """Aggregated dashboard telemetry metrics."""

    total_events: int
    mean_latency_gpu_ms: float
    mean_latency_cpu_ms: float
    expected_calibration_error: float
    conformal_empirical_coverage: float
    ood_shift_ratio: float
    domain_breakdown: Dict[str, Dict[str, Any]]
    status: str


class MicrobrainTelemetry:
    """Observability manager for fleet microbrain health and performance."""

    MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB rotation limit

    def __init__(self, log_path: Optional[Path] = None, dashboard_path: Optional[Path] = None):
        mem_dir = Path(__file__).resolve().parents[1] / "memory"
        self.log_path = log_path or (mem_dir / "microbrain_telemetry.jsonl")
        self.dashboard_path = dashboard_path or (mem_dir / "microbrain_telemetry_dashboard.json")

    def _rotate_if_needed(self, path: Path) -> None:
        """Rotate log file if size exceeds MAX_LOG_BYTES (10MB)."""
        if path.exists() and path.stat().st_size >= self.MAX_LOG_BYTES:
            ts = time.time_ns()
            rotated = path.with_name(f"{path.stem}_{ts}.jsonl")
            try:
                path.rename(rotated)
            except Exception as exc:
                log.warning("MicrobrainTelemetry rotation error for %s: %s", path, exc)

    def record_event(
        self,
        domain: str,
        backend: str = "cpu",
        latency_ms: float = 0.0,
        confidence: float = 1.0,
        verified_accurate: bool = True,
        ood_score: float = 0.05,
        conformal_quantile: float = 0.0125,
        passed_conformal: bool = True,
        action: str = "infer",
        prediction: str = "",
    ) -> TelemetryEvent:
        """Log a telemetry event to JSONL with 10MB file rotation."""
        evt = TelemetryEvent(
            timestamp=utc_now_iso(),
            domain=domain,
            backend=backend,
            latency_ms=round(latency_ms, 3),
            confidence=round(confidence, 4),
            verified_accurate=verified_accurate,
            ood_score=round(ood_score, 4),
            conformal_quantile=round(conformal_quantile, 4),
            passed_conformal=passed_conformal,
            action=action,
            prediction=prediction,
        )

        record_line = json.dumps(asdict(evt)) + "\n"

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(self.log_path)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(record_line)
        except Exception as exc:
            log.debug("Telemetry logging failed for %s: %s", self.log_path, exc)

        return evt

    def compute_ece(self, confidences: List[float], accuracies: List[bool], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE) across M equal-width confidence bins."""
        if not confidences or len(confidences) != len(accuracies):
            return 0.0

        n = len(confidences)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            b_min, b_max = bins[i], bins[i + 1]
            idx = [
                j for j, c in enumerate(confidences)
                if (b_min <= c < b_max) or (i == n_bins - 1 and b_min <= c <= b_max)
            ]
            if not idx:
                continue

            bin_size = len(idx)
            bin_acc = float(np.mean([1.0 if accuracies[j] else 0.0 for j in idx]))
            bin_conf = float(np.mean([confidences[j] for j in idx]))

            ece += (bin_size / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def generate_dashboard(self) -> DashboardReport:
        """Parse telemetry events and synthesize an executive health dashboard."""
        events: List[TelemetryEvent] = []
        if self.log_path.exists():
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            events.append(TelemetryEvent(**d))
                        except Exception:
                            continue

        if not events:
            return DashboardReport(
                total_events=0,
                mean_latency_gpu_ms=0.0,
                mean_latency_cpu_ms=0.0,
                expected_calibration_error=0.0,
                conformal_empirical_coverage=1.0,
                ood_shift_ratio=0.0,
                domain_breakdown={},
                status="no_data",
            )

        gpu_latencies = [e.latency_ms for e in events if e.backend == "opencl_gpu"]
        cpu_latencies = [e.latency_ms for e in events if e.backend == "cpu"]

        confidences = [e.confidence for e in events]
        accuracies = [e.verified_accurate for e in events]
        ece = self.compute_ece(confidences, accuracies)

        conformal_passes = sum(1 for e in events if e.passed_conformal)
        cov = conformal_passes / len(events)

        ood_events = sum(1 for e in events if e.ood_score > 0.40)
        ood_ratio = ood_events / len(events)

        # Domain breakdown
        domain_data: Dict[str, List[TelemetryEvent]] = {}
        for e in events:
            domain_data.setdefault(e.domain, []).append(e)

        breakdown = {}
        for dom, evts in domain_data.items():
            breakdown[dom] = {
                "count": len(evts),
                "mean_latency_ms": round(float(np.mean([e.latency_ms for e in evts])), 2),
                "conformal_coverage": round(sum(1 for e in evts if e.passed_conformal) / len(evts), 3),
                "ece": round(self.compute_ece([e.confidence for e in evts], [e.verified_accurate for e in evts]), 4),
            }

        report = DashboardReport(
            total_events=len(events),
            mean_latency_gpu_ms=round(float(np.mean(gpu_latencies)), 2) if gpu_latencies else 0.0,
            mean_latency_cpu_ms=round(float(np.mean(cpu_latencies)), 2) if cpu_latencies else 0.0,
            expected_calibration_error=round(ece, 4),
            conformal_empirical_coverage=round(cov, 4),
            ood_shift_ratio=round(ood_ratio, 4),
            domain_breakdown=breakdown,
            status="healthy" if (cov >= 0.90 and ece <= 0.15) else "attention_required",
        )

        self.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        self.dashboard_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return report


_telemetry_singleton: Optional[MicrobrainTelemetry] = None


def get_microbrain_telemetry() -> MicrobrainTelemetry:
    """Singleton getter for MicrobrainTelemetry."""
    global _telemetry_singleton
    if _telemetry_singleton is None:
        _telemetry_singleton = MicrobrainTelemetry()
    return _telemetry_singleton
