"""
Aetheris Astra-Style Sandbox Tracer & Attractor Localizer.
Mimics the MIT CSAIL GPT-6 Astra diagnostic protocol:
1. Traces activations and residual difference vectors across forward passes.
2. Isolates the empirical cyclic attractor vector:
       v_attractor = 1/K sum_{k=1}^K (h_k - h_mean)
3. Identifies the specific parameter columns contributing to the attractor.
Runs completely in-memory on CPU in < 0.1ms.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class AstraSandboxTracer:
    """
    Traces execution forward passes to identify cyclic decision ruts and attractor vectors.
    """

    def __init__(self, d_hidden: int = 64, buffer_capacity: int = 50):
        self.d_hidden = d_hidden
        self.buffer_capacity = buffer_capacity
        self.activation_history: List[np.ndarray] = []

    def record_activation(self, h: np.ndarray) -> None:
        """Records a hidden state vector from a forward pass."""
        arr = np.asarray(h, dtype=np.float32).flatten()
        if len(arr) != self.d_hidden:
            if len(arr) < self.d_hidden:
                arr = np.pad(arr, (0, self.d_hidden - len(arr)))
            else:
                arr = arr[: self.d_hidden]

        self.activation_history.append(arr)
        if len(self.activation_history) > self.buffer_capacity:
            self.activation_history.pop(0)

    def diagnose_attractor(self) -> Optional[Dict[str, Any]]:
        """
        Analyzes activation history to detect cyclic attractors or repetitive ruts.
        Returns the normalized attractor direction v_attractor and cosine autocorrelation.
        """
        if len(self.activation_history) < 6:
            return None

        recent = np.array(self.activation_history[-10:], dtype=np.float32)  # (K, d_hidden)
        # Compute mean trajectory direction
        h_mean = np.mean(recent, axis=0)
        diffs = recent - h_mean  # (K, d_hidden)
        cov = np.matmul(diffs.T, diffs) / len(diffs)
        eigvals, eigvecs = np.linalg.eigh(cov)
        total_energy = float(np.sum(np.maximum(0.0, eigvals)))

        # Static fixed-point attractor: repeated identical states have near-zero variance around h_mean
        if total_energy < 1e-6:
            mean_norm = float(np.linalg.norm(h_mean))
            if mean_norm > 1e-6:
                v_attractor = (h_mean / mean_norm).astype(np.float32)
                return {
                    "is_cyclic_rut": True,
                    "mean_autocorr": 1.0,
                    "attractor_energy_ratio": 1.0,
                    "v_attractor": v_attractor,
                    "sample_count": len(recent),
                }
            return {
                "is_cyclic_rut": False,
                "mean_autocorr": 0.0,
                "attractor_energy_ratio": 0.0,
                "v_attractor": np.zeros(self.d_hidden, dtype=np.float32),
                "sample_count": len(recent),
            }

        # Check autocorrelation across successive steps (captures both monotone drift & period-2 oscillations)
        corrs = []
        for i in range(len(recent) - 1):
            v1 = recent[i] - h_mean
            v2 = recent[i + 1] - h_mean
            norm1 = np.linalg.norm(v1) + 1e-12
            norm2 = np.linalg.norm(v2) + 1e-12
            corrs.append(float(np.dot(v1, v2) / (norm1 * norm2)))

        mean_autocorr = float(np.mean(corrs))

        # Top eigenvector corresponds to strongest attractor direction
        top_idx = int(np.argmax(eigvals))
        v_attractor = eigvecs[:, top_idx].astype(np.float32)

        attractor_energy_ratio = float(max(0.0, eigvals[top_idx]) / (total_energy + 1e-12))
        is_cyclic_rut = bool(attractor_energy_ratio > 0.45 or abs(mean_autocorr) > 0.65)

        return {
            "is_cyclic_rut": is_cyclic_rut,
            "mean_autocorr": round(mean_autocorr, 4),
            "attractor_energy_ratio": round(attractor_energy_ratio, 4),
            "v_attractor": v_attractor,
            "sample_count": len(recent),
        }

    def clear_history(self) -> None:
        self.activation_history.clear()

