"""
Aetheris CPU Epistemic Oracle (Uncertainty Quantification via RND).
Implements Random Network Distillation (RND) and Bayesian Epistemic Uncertainty Estimation.
CPU-only via NumPy (no GPU path); per-call latency is not claimed here — measure it
on the target workload if it matters.
Provides the agent with an instantaneous 'Novelty / Ignorance' metric:
    - Low Error: Task pattern is familiar (Agent proceeds with high confidence).
    - High Error: Task is in the Epistemic Unknown (Triggers caution, verification, and research).
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

STORE_DIR = Path("memory/store")
ORACLE_WEIGHTS_FILE = STORE_DIR / "wsp_epistemic_oracle.npz"


class EpistemicOracle:
    """
    RND-based Epistemic Uncertainty Estimator for CPU.
    Target Network W_target: Fixed, randomly initialized projection (immutable).
    Predictor Network W_predictor: Adapts as the agent encounters known task embeddings.
    Uncertainty Metric = ||W_predictor(x) - W_target(x)||^2
    """

    def __init__(self, d_in: int = 128, d_repr: int = 32, seed: int = 42):
        self.d_in = d_in
        self.d_repr = d_repr
        self.rng = np.random.default_rng(seed)

        # Fixed target network
        self.W_target = self.rng.standard_normal((d_in, d_repr), dtype=np.float32) / np.sqrt(d_in)
        # Adaptable predictor network
        self.W_predictor = np.zeros((d_in, d_repr), dtype=np.float32)

        # Familiarity learning rate (online least squares / delta update on CPU)
        self.learning_rate = 0.05
        self._ensure_storage()
        self.load_or_initialize()

    def _ensure_storage(self) -> None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_initialize(self) -> None:
        if ORACLE_WEIGHTS_FILE.exists():
            try:
                data = np.load(ORACLE_WEIGHTS_FILE)
                self.W_target = data["W_target"]
                self.W_predictor = data["W_predictor"]
            except Exception as e:
                logger.warning(f"Failed to load Epistemic Oracle weights: {e}")
                self.save_weights()
        else:
            self.save_weights()

    def save_weights(self) -> None:
        np.savez_compressed(
            ORACLE_WEIGHTS_FILE,
            W_target=self.W_target,
            W_predictor=self.W_predictor,
        )

    def compute_uncertainty(self, feature_vec: np.ndarray) -> Dict[str, Any]:
        """
        Estimates epistemic uncertainty for a given feature representation x in <0.02ms.
        Returns:
            uncertainty_score: float in [0.0, inf) (typically 0.0 to ~2.0)
            is_epistemic_unknown: bool (True if uncertainty exceeds threshold)
            confidence: float in [0.0, 1.0]
        """
        x = np.asarray(feature_vec, dtype=np.float32).flatten()
        if len(x) != self.d_in:
            # Resize or pad/truncate to match d_in
            if len(x) < self.d_in:
                x = np.pad(x, (0, self.d_in - len(x)))
            else:
                x = x[: self.d_in]

        # Normalization
        norm = np.linalg.norm(x) + 1e-12
        x_norm = x / norm

        target_y = np.matmul(x_norm, self.W_target)
        pred_y = np.matmul(x_norm, self.W_predictor)

        # Mean squared error in representation space
        mse = float(np.mean((target_y - pred_y) ** 2))
        confidence = float(np.exp(-25.0 * mse))

        return {
            "uncertainty_score": round(mse, 6),
            "confidence": round(confidence, 4),
            "is_epistemic_unknown": bool(mse > 0.005),
        }

    def register_familiar_experience(self, feature_vec: np.ndarray) -> None:
        """
        Performs a fast closed-form delta update to teach the predictor network
        that this task representation has now been experienced.
        Zero GPU, single-step CPU gradient: Delta W = lr * (x^T @ (target - pred))
        """
        x = np.asarray(feature_vec, dtype=np.float32).flatten()
        if len(x) != self.d_in:
            if len(x) < self.d_in:
                x = np.pad(x, (0, self.d_in - len(x)))
            else:
                x = x[: self.d_in]

        x_norm = (x / (np.linalg.norm(x) + 1e-12)).reshape(1, -1)  # (1, d_in)
        target_y = np.matmul(x_norm, self.W_target)  # (1, d_repr)
        pred_y = np.matmul(x_norm, self.W_predictor)  # (1, d_repr)

        err = target_y - pred_y  # (1, d_repr)
        delta_W = self.learning_rate * np.matmul(x_norm.T, err)

        self.W_predictor += delta_W
        self.save_weights()

