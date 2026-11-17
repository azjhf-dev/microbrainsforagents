"""
Weight-Space Programming (WSP) Neural Surgeon for Aetheris.
Ported and evolved from MIT CSAIL SOTA (Takehi, Teoh, Zhang, Khattab, Nov 2026: "Sparks of Weight-Space Programming").

Enables direct zero-gradient neural network surgery and programmatic weight editing:
1. Attractor Diagnostic & Deflation: Identifies pathological cyclic attractors (dominant eigenvectors
   causing infinite loops like Qwen3.5-0.8B layers 15 & 19) and deflates them via closed-form SVD projection.
2. Direct Rank-1 Associative Fact Implantation (ROME/MEMIT style):
   Instantly implants a new key-value rule into the weight matrix without backprop:
   Delta_W = (v_target - W * k_key) * k_key^T / (k_key^T * k_key).
3. Colony-Wide Weight Sanitation: Audits all neuro-brains in memory/store/neuro_brains/ for degenerate collapse.
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger("aetheris.wsp_neural_surgeon")


@dataclass
class WSPSurgeryReport:
    brain_id: str
    was_pathological: bool
    dominant_eigen_ratio: float
    parameters_modified_count: int
    modified_ratio: float
    action_taken: str


@dataclass
class ImplantResult:
    key_recalled_similarity: float
    latency_ms: float
    implanted_norm: float


class WSPNeuralSurgeon:
    """
    Direct weight-space diagnostic and zero-gradient surgical editor.
    """

    _instance: Optional["WSPNeuralSurgeon"] = None

    def __init__(self, attractor_ratio_threshold: float = 3.5, deflation_factor: float = 0.65):
        self.ratio_threshold = attractor_ratio_threshold
        self.deflation_factor = deflation_factor

    @classmethod
    def get_instance(cls) -> "WSPNeuralSurgeon":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def diagnose_and_deflate_attractor(
        self,
        W: np.ndarray,
        brain_id: str = "unnamed_brain"
    ) -> Tuple[np.ndarray, WSPSurgeryReport]:
        """
        Diagnoses and excises degenerative reasoning loop attractors without backpropagation.
        Mirrors Astra's intervention on Qwen3.5-0.8B layers 15 & 19.
        """
        W_arr = np.asarray(W, dtype=np.float64)
        orig_shape = W_arr.shape
        if len(orig_shape) != 2 or min(orig_shape) < 2:
            return W_arr, WSPSurgeryReport(brain_id, False, 1.0, 0, 0.0, "skipped_incompatible_shape")

        U, s, Vt = np.linalg.svd(W_arr, full_matrices=False)
        sigma1 = float(s[0])
        sigma2 = float(s[1]) if len(s) > 1 else 1e-9
        eigen_ratio = sigma1 / max(1e-9, sigma2)

        # Check if the primary singular direction excessively dominates (attractor loop trap)
        if eigen_ratio >= self.ratio_threshold:
            # Deflate the primary pathological subspace
            u1 = U[:, 0:1]
            v1_t = Vt[0:1, :]
            correction = (self.deflation_factor * (sigma1 - sigma2)) * np.dot(u1, v1_t)
            W_sanitized = W_arr - correction

            # Identify parameters with substantial shift
            delta_mask = np.abs(correction) > (0.01 * np.mean(np.abs(W_arr)))
            mod_count = int(np.sum(delta_mask))
            mod_ratio = float(mod_count / W_arr.size)

            report = WSPSurgeryReport(
                brain_id=brain_id,
                was_pathological=True,
                dominant_eigen_ratio=round(eigen_ratio, 3),
                parameters_modified_count=mod_count,
                modified_ratio=round(mod_ratio, 5),
                action_taken=f"deflated_attractor_subspace (deflation={self.deflation_factor})"
            )
            return W_sanitized, report

        return W_arr, WSPSurgeryReport(
            brain_id=brain_id,
            was_pathological=False,
            dominant_eigen_ratio=round(eigen_ratio, 3),
            parameters_modified_count=0,
            modified_ratio=0.0,
            action_taken="healthy_eigen_spectrum_retained"
        )

    def direct_rank1_implant(
        self,
        W: np.ndarray,
        key_vector: np.ndarray,
        target_value_vector: np.ndarray
    ) -> Tuple[np.ndarray, ImplantResult]:
        """
        Closed-form rank-1 weight implant (ROME / MEMIT associative memory formulation).
        Directly programs W such that: W_new * key ≈ target_value
        Zero gradient descent, sub-millisecond execution.
        """
        import time
        t0 = time.perf_counter()

        W_curr = np.asarray(W, dtype=np.float64)
        k = np.asarray(key_vector, dtype=np.float64).flatten()
        v_target = np.asarray(target_value_vector, dtype=np.float64).flatten()

        # Dimension checks
        out_dim, in_dim = W_curr.shape
        if len(k) != in_dim or len(v_target) != out_dim:
            raise ValueError(f"Shape mismatch: W={W_curr.shape}, k={k.shape}, v={v_target.shape}")

        k_norm_sq = float(np.dot(k, k))
        if k_norm_sq < 1e-12:
            return W_curr, ImplantResult(0.0, 0.0, float(np.linalg.norm(W_curr)))

        # Prediction error under current weights
        current_pred = np.dot(W_curr, k)
        error = v_target - current_pred

        # Exact closed-form least-squares rank-1 update
        delta_W = np.outer(error, k) / k_norm_sq
        W_new = W_curr + delta_W

        # Verification of recall
        recalled = np.dot(W_new, k)
        norm_rec = np.linalg.norm(recalled)
        norm_tgt = np.linalg.norm(v_target)
        if norm_rec > 1e-12 and norm_tgt > 1e-12:
            cos_sim = float(np.dot(recalled, v_target) / (norm_rec * norm_tgt))
        else:
            cos_sim = 1.0

        latency = (time.perf_counter() - t0) * 1000.0

        return W_new, ImplantResult(
            key_recalled_similarity=round(cos_sim, 4),
            latency_ms=round(latency, 4),
            implanted_norm=round(float(np.linalg.norm(delta_W)), 4)
        )

    def direct_multifact_memit_implant(
        self,
        W: np.ndarray,
        key_matrix: Union[List[List[float]], np.ndarray],
        value_matrix: Union[List[List[float]], np.ndarray],
        l2_reg: float = 1e-4,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Closed-form batch multi-fact MEMIT weight surgery:
            Delta_W = (V^T - W @ K^T) @ K @ (K^T @ K + lambda * I)^{-1}
        Note: for K rows (N_facts, d_in) and V rows (N_facts, d_out):
            residual = V^T - W @ K^T  (shape d_out x N_facts)
            Using dual ridge form: Delta_W = residual @ (K @ K^T + lambda * I)^{-1} @ K
            or primal form: (d_in x d_in) solve. Dual form works cleanly for any N_facts <= d_in!
        """
        import time
        t0 = time.perf_counter()

        W_curr = np.asarray(W, dtype=np.float64)
        out_dim, in_dim = W_curr.shape

        K = np.asarray(key_matrix, dtype=np.float64)
        V = np.asarray(value_matrix, dtype=np.float64)
        if K.ndim == 1:
            K = K[None, :]
        if V.ndim == 1:
            V = V[None, :]

        # Accept either (N_facts, d_in) or transposed (d_in, N_facts)
        if K.shape[1] != in_dim and K.shape[0] == in_dim:
            K = K.T
        if V.shape[1] != out_dim and V.shape[0] == out_dim:
            V = V.T

        if K.shape[1] != in_dim or V.shape[1] != out_dim or K.shape[0] != V.shape[0]:
            raise ValueError(f"Shape mismatch: W={W_curr.shape}, K={K.shape}, V={V.shape}")

        n_facts = K.shape[0]
        if n_facts == 0:
            return W_curr, {"status": "skipped_empty_facts", "facts_count": 0}

        # Dual closed-form ridge MEMIT solution:
        # Let R = V.T - W @ K.T  (d_out x n_facts)
        # Delta_W = R @ solve(K @ K.T + lambda * I, K)
        K_Kt = K @ K.T + float(l2_reg) * np.eye(n_facts, dtype=np.float64)
        residual = V.T - np.dot(W_curr, K.T)
        try:
            proj = np.linalg.solve(K_Kt, K)  # (n_facts, in_dim)
        except np.linalg.LinAlgError:
            proj, _, _, _ = np.linalg.lstsq(K_Kt, K, rcond=None)
        delta_W = np.dot(residual, proj)
        W_new = W_curr + delta_W

        recalled = np.dot(K, W_new.T)  # (n_facts, out_dim)
        cos_list = []
        for idx in range(n_facts):
            r_v = recalled[idx]
            t_v = V[idx]
            nr = float(np.linalg.norm(r_v))
            nt = float(np.linalg.norm(t_v))
            if nr > 1e-12 and nt > 1e-12:
                cos_list.append(float(np.dot(r_v, t_v) / (nr * nt)))
            else:
                cos_list.append(1.0 if nr <= 1e-12 and nt <= 1e-12 else 0.0)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return W_new, {
            "status": "memit_multifact_implanted",
            "facts_count": int(n_facts),
            "mean_recall_cosine": round(float(np.mean(cos_list)), 4),
            "min_recall_cosine": round(float(np.min(cos_list)), 4),
            "implanted_norm": round(float(np.linalg.norm(delta_W)), 4),
            "latency_ms": round(latency_ms, 4),
        }

    def audit_neuro_brains_store(
        self,
        store_dir: str = "memory/store/neuro_brains",
        heal_pathological: bool = False,
    ) -> Dict[str, Any]:
        """
        Colony-wide health scan inspecting all neuro-brain transition matrices for
        degenerative attractor collapse (skipping vocabulary embeddings & Hopfield banks).
        """
        _p = Path(store_dir)
        p = _p if (_p.exists() or store_dir != "memory/store/neuro_brains") else (Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")
        if not p.exists():
            return {"status": "skipped", "message": f"Store not found: {store_dir}"}

        npz_files = list(p.glob("*_weights.npz"))
        reports = []
        pathological_count = 0
        healed_files_count = 0
        skipped_keys = {"hopfield_keys", "hopfield_values", "W_embed"}

        for f in npz_files:
            try:
                with np.load(f) as data:
                    arrays = {k: data[k] for k in data.files}
                file_healed = False
                for w_name, mat in list(arrays.items()):
                    if w_name in skipped_keys:
                        continue
                    if mat.ndim == 2 and min(mat.shape) >= 4:
                        sanitized, rep = self.diagnose_and_deflate_attractor(mat, brain_id=f"{f.stem}:{w_name}")
                        if rep.was_pathological:
                            pathological_count += 1
                            reports.append(rep)
                            if heal_pathological:
                                arrays[w_name] = sanitized.astype(mat.dtype if np.issubdtype(mat.dtype, np.floating) else np.float32)
                                file_healed = True
                if file_healed and heal_pathological:
                    np.savez_compressed(f, **arrays)
                    healed_files_count += 1
            except Exception:
                continue

        return {
            "total_files_audited": len(npz_files),
            "pathological_brains_detected": pathological_count,
            "healed_files_count": healed_files_count,
            "actions_logged": [r.__dict__ for r in reports[:10]],
        }

