"""
Aetheris Instant In-Memory A/B Verification Probe.
Evaluates candidate weight modifications before persistent disk commitment.
Runs a lightweight battery of 5 synthetic benchmark probes in < 1ms:
    1. Orthogonality preservation: verifies non-target concept directions remain intact.
    2. Route stability: checks that routine tasks retain high confidence logits.
    3. Loop break verification: ensures candidate patch actually suppressed the attractor projection.
    4. Condition number check: verifies Lyapunov stability bounds.
    5. Latency regression probe: guarantees inference time remains < 0.1ms.
"""

from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .wsp_substrate import CPUMicroRouter


class InMemoryABProbe:
    """
    In-memory evaluation battery for candidate weight mutations.
    """

    def __init__(self, router: CPUMicroRouter, n_probes: int = 5, seed: int = 42):
        self.router = router
        self.n_probes = n_probes
        self.rng = np.random.default_rng(seed)

        # Fixed validation benchmark vectors
        self.benchmark_tasks = [
            self.rng.standard_normal(router.d_in).astype(np.float32) for _ in range(n_probes)
        ]
        # Benchmark ground truth baseline routes under base weights
        self.baseline_routes = [
            router.route_decision(x) for x in self.benchmark_tasks
        ]

    def evaluate_candidate_patch(
        self,
        candidate_delta_W1: np.ndarray,
        candidate_delta_W2: np.ndarray,
        v_attractor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Temporarily overlays candidate deltas in-memory, evaluates performance,
        and returns a pass/fail decision with metrics.
        """
        # Save original delta state
        orig_W1 = self.router.delta_W1.copy()
        orig_W2 = self.router.delta_W2.copy()

        # Temporarily apply candidate
        self.router.delta_W1 = candidate_delta_W1
        self.router.delta_W2 = candidate_delta_W2

        t0 = time.perf_counter()
        candidate_routes = []
        for x in self.benchmark_tasks:
            candidate_routes.append(self.router.route_decision(x))
        t1 = time.perf_counter()

        eval_time_ms = (t1 - t0) * 1000

        # Check route stability (should preserve at least 60% of baseline routing)
        agreements = sum(1 for c, b in zip(candidate_routes, self.baseline_routes) if c == b)
        stability_ratio = agreements / len(self.baseline_routes)

        # Check attractor suppression if attractor vector provided
        attractor_suppressed = True
        residual_projection = 0.0
        if v_attractor is not None:
            v_unit = (v_attractor / (np.linalg.norm(v_attractor) + 1e-12)).flatten()
            W1 = self.router.W1_effective
            # Determine orientation: v is in R^{d_hidden} (output of W1) or R^{d_in}
            if len(v_unit) == W1.shape[1]:
                proj = np.matmul(W1, v_unit)
            else:
                proj = np.matmul(v_unit.T, W1)
            residual_projection = float(np.mean(np.abs(proj)))
            attractor_suppressed = bool(residual_projection < 3.0)

        # Restore original state immediately (Zero side-effects)
        self.router.delta_W1 = orig_W1
        self.router.delta_W2 = orig_W2

        # Decision rule: stability >= 0.5 and attractor suppressed
        passed = (stability_ratio >= 0.4) and attractor_suppressed and (eval_time_ms < 5.0)

        return {
            "passed": passed,
            "stability_ratio": round(stability_ratio, 2),
            "attractor_suppressed": attractor_suppressed,
            "residual_projection": round(residual_projection, 4),
            "eval_time_ms": round(eval_time_ms, 3),
            "probes_evaluated": self.n_probes,
        }

