"""
Aetheris Weight-Space Task Arithmetic & Model Merging Engine.
Implements SOTA weight merging algorithms purely on CPU via NumPy:
1. Task Arithmetic (vector addition/subtraction: tau = W_task - W_base).
2. TIES-Merging (Trim, Elect Sign, Merge): resolves parameter interference across multiple task lenses.
3. DARE (Drop And REscale): stochastic zero-gradient parameter sparsification & rescaling.
4. Model Soups: greedy/uniform soup aggregation of phylogenetic winning attempt deltas.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

STORE_DIR = Path("memory/store")
SOUPS_DIR = STORE_DIR / "wsp_soups"


class WeightSpaceMerger:
    """
    High-performance, CPU-native weight merger for LoRA lenses and task vectors.
    """

    def __init__(self, soups_dir: Optional[Path] = None):
        self.soups_dir = soups_dir or SOUPS_DIR
        self.soups_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. DARE: Drop And REscale
    # -------------------------------------------------------------------------
    @staticmethod
    def apply_dare(delta: np.ndarray, drop_rate: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
        """
        Applies DARE to a task delta matrix.
        Randomly drops (zeros out) fraction `drop_rate` of elements and rescales the rest by 1/(1 - drop_rate).
        Protects dense base representations from catastrophic parameter interference.
        """
        if drop_rate <= 0.0:
            return delta.copy()
        if drop_rate >= 1.0:
            return np.zeros_like(delta)

        rng = np.random.default_rng(seed)
        mask = rng.uniform(0.0, 1.0, size=delta.shape) >= drop_rate
        scale = 1.0 / (1.0 - drop_rate)
        return (delta * mask * scale).astype(np.float32)

    # -------------------------------------------------------------------------
    # 2. TIES-Merging: Trim, Elect Sign, Merge
    # -------------------------------------------------------------------------
    @staticmethod
    def ties_merge(
        deltas: List[np.ndarray],
        k_sparsity: float = 0.2,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        TIES-Merging across a list of task delta matrices:
        1. Trim: Keep top k_sparsity fraction of parameters by magnitude in each delta, zero out the rest.
        2. Elect Sign: Compute total signed direction for each parameter coordinate; elect majority sign.
        3. Merge: Average only parameters that agree with the elected sign.
        """
        if not deltas:
            raise ValueError("Deltas list cannot be empty for TIES-merging.")
        if len(deltas) == 1:
            return deltas[0].copy()

        shape = deltas[0].shape
        for d in deltas:
            if d.shape != shape:
                raise ValueError(f"Mismatched shapes in TIES merge: {d.shape} vs {shape}")

        n_models = len(deltas)
        w = np.array(weights if weights else [1.0 / n_models] * n_models, dtype=np.float32)
        w = w / np.sum(w)

        # Step 1: Trim (keep top k_sparsity by magnitude)
        trimmed_deltas = []
        for d in deltas:
            d_flat = d.flatten()
            n_keep = max(1, int(len(d_flat) * k_sparsity))
            thresh = np.partition(np.abs(d_flat), -n_keep)[-n_keep]
            mask = np.abs(d) >= thresh
            trimmed_deltas.append(d * mask)

        trimmed_stack = np.stack(trimmed_deltas, axis=0)  # (N, *shape)

        # Step 2: Elect Sign
        signed_sums = np.sum(trimmed_stack, axis=0)  # (*shape)
        elected_sign = np.sign(signed_sums)
        # Ties/zeros defaulted to positive
        elected_sign[elected_sign == 0] = 1.0

        # Step 3: Disjoint Merge (average only elements with matching sign)
        merged = np.zeros(shape, dtype=np.float32)
        counts = np.zeros(shape, dtype=np.float32)

        for i, td in enumerate(trimmed_deltas):
            matches = (np.sign(td) == elected_sign) & (td != 0.0)
            merged += w[i] * td * matches
            counts += w[i] * matches

        counts[counts == 0] = 1.0
        final_merged = merged / counts
        return final_merged.astype(np.float32)

    # -------------------------------------------------------------------------
    # 3. Task Arithmetic: Add / Subtract Lenses
    # -------------------------------------------------------------------------
    @staticmethod
    def task_arithmetic_combine(
        delta_pairs: List[Tuple[np.ndarray, float]]
    ) -> np.ndarray:
        """
        Linear task arithmetic: Delta_out = sum_i (alpha_i * Delta_i).
        Allows adding desired capabilities and subtracting negative biases.
        """
        if not delta_pairs:
            raise ValueError("Delta pairs cannot be empty.")
        out = np.zeros_like(delta_pairs[0][0], dtype=np.float32)
        for delta, alpha in delta_pairs:
            out += alpha * delta
        return out

    # -------------------------------------------------------------------------
    # 4. Model Soups: Crystallizing Winning Attempt Deltas
    # -------------------------------------------------------------------------
    def brew_model_soup(
        self,
        attempt_deltas: List[Dict[str, np.ndarray]],
        performance_scores: List[float],
        name: str = "crystallized_soup_v1",
        use_greedy: bool = True,
    ) -> Dict[str, Any]:
        """
        Brews a Model Soup from a population of successful execution attempt deltas.
        If use_greedy is True, weights attempts proportionally to their benchmark score.
        Saves the crystallized soup to disk.
        """
        if not attempt_deltas:
            raise ValueError("No attempt deltas provided to brew soup.")

        scores = np.array(performance_scores, dtype=np.float32)
        if use_greedy:
            # Softmax / normalized score weighting
            exp_s = np.exp(scores - np.max(scores))
            weights = (exp_s / np.sum(exp_s)).tolist()
        else:
            weights = [1.0 / len(attempt_deltas)] * len(attempt_deltas)

        # Merge each layer present (e.g. delta_W1, delta_W2)
        layer_names = list(attempt_deltas[0].keys())
        soup_dict: Dict[str, np.ndarray] = {}

        for l_name in layer_names:
            layer_deltas = [ad[l_name] for ad in attempt_deltas]
            # Use TIES-merging to prevent parameter conflict during soup crystallization
            soup_dict[l_name] = self.ties_merge(layer_deltas, k_sparsity=0.3, weights=weights)

        save_path = self.soups_dir / f"{name}.npz"
        np.savez_compressed(save_path, **soup_dict)

        return {
            "name": name,
            "path": str(save_path),
            "n_attempts_merged": len(attempt_deltas),
            "layers": layer_names,
            "weights": [round(float(w), 4) for w in weights],
        }

