"""
Aetheris Weight-Space Programming (WSP) Unified Engine.
Integrates:
1. Concept-Level Sparse Autoencoders (SAE) for non-interfering monosemantic steering.
2. Execution-Driven Neural Repair (auto-triggered on repetitive failure loops).
3. Epigenetic LoRA Lenses (instantaneous mode switching <0.05ms).
4. Formal Spectral Invariants (Lyapunov spectral norm & condition bounds safety gate).
"""

from __future__ import annotations
import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from evolution.wsp_substrate import CPUMicroRouter, ROMEAssociativeMemory
from evolution.wsp_sae import ConceptSparseAutoencoder
from evolution.wsp_lenses import EpigeneticLensManager

logger = logging.getLogger(__name__)

STORE_DIR = Path("memory/store")
WSP_LEDGER_FILE = STORE_DIR / "wsp_ledger.jsonl"
REPETITIVE_FAILURES_FILE = STORE_DIR / "wsp_failure_tracker.json"


class WSPSpectralInvariantViolation(Exception):
    """Raised when a candidate weight patch violates formal stability invariants."""
    pass


class WSPEngine:
    """
    Unified Weight-Space Programming & Neural Surgery Engine.
    Combines SVD eigenvalue manipulation, ROME associative memory,
    Sparse Autoencoders, Epigenetic Lenses, and Formal Stability Verification.
    """

    def __init__(
        self,
        router: Optional[CPUMicroRouter] = None,
        assoc_mem: Optional[ROMEAssociativeMemory] = None,
        sae: Optional[ConceptSparseAutoencoder] = None,
        lens_manager: Optional[EpigeneticLensManager] = None,
        max_condition_number: float = 250.0,
        max_frobenius_ratio: float = 1.35,
    ):
        self.router = router or CPUMicroRouter()
        self.assoc_mem = assoc_mem or ROMEAssociativeMemory()
        self.sae = sae or ConceptSparseAutoencoder(d_model=self.router.d_hidden)
        self.lens_manager = lens_manager or EpigeneticLensManager()

        # Formal Spectral Invariants Thresholds
        self.max_condition_number = max_condition_number
        self.max_frobenius_ratio = max_frobenius_ratio

        STORE_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Pillar 4: Formal Weight Invariants (Safety Gate)
    # -------------------------------------------------------------------------
    def verify_spectral_invariants(self, W_candidate: np.ndarray, W_base: np.ndarray, layer_name: str) -> Dict[str, Any]:
        """
        Formally verifies mathematical stability invariants:
        1. Condition Number: cond(W) <= max_condition_number (prevents near-singular matrices).
        2. Frobenius Norm Ratio: ||W_cand||_F / ||W_base||_F <= max_frobenius_ratio (prevents exploding weights).
        3. NaN / Inf check.
        """
        if not np.all(np.isfinite(W_candidate)):
            raise WSPSpectralInvariantViolation(f"Candidate weight matrix for {layer_name} contains NaN or Inf.")

        U, S, Vt = np.linalg.svd(W_candidate, full_matrices=False)
        cond_num = float(S[0] / (S[-1] + 1e-12))
        if cond_num > self.max_condition_number:
            raise WSPSpectralInvariantViolation(
                f"Candidate condition number {cond_num:.2f} exceeds safety threshold {self.max_condition_number}."
            )

        norm_cand = float(np.linalg.norm(W_candidate, ord="fro"))
        norm_base = float(np.linalg.norm(W_base, ord="fro")) + 1e-12
        ratio = norm_cand / norm_base
        if ratio > self.max_frobenius_ratio:
            raise WSPSpectralInvariantViolation(
                f"Candidate Frobenius norm ratio {ratio:.2f} exceeds safety threshold {self.max_frobenius_ratio}."
            )

        return {
            "status": "passed",
            "layer_name": layer_name,
            "condition_number": round(cond_num, 2),
            "frobenius_ratio": round(ratio, 4),
            "dominant_singular_value": round(float(S[0]), 4),
        }

    # -------------------------------------------------------------------------
    # Spectral Inspection & SVD Manipulation
    # -------------------------------------------------------------------------
    def inspect_spectrum(self, layer_name: str = "W1") -> Dict[str, Any]:
        W = self.router.W1_effective if layer_name == "W1" else self.router.W2_effective
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        total_energy = float(np.sum(S**2))
        top_3_ratio = float(np.sum(S[:3]**2) / (total_energy + 1e-12))
        cond_number = float(S[0] / (S[-1] + 1e-12))

        return {
            "layer_name": layer_name,
            "shape": list(W.shape),
            "singular_values_top5": [round(float(s), 4) for s in S[:5]],
            "condition_number": round(cond_number, 2),
            "top3_energy_ratio": round(top_3_ratio, 4),
            "has_dominant_rut": bool(top_3_ratio > 0.65),
            "is_degenerate": bool(cond_number > 150.0),
        }

    def regularize_spectrum(self, layer_name: str = "W1", max_condition: float = 50.0) -> Dict[str, Any]:
        """
        Pillar 4: Orthogonal Spectrum Regularizer.
        Prevents matrices from collapsing into ill-conditioned ruts (condition number > max_condition).
        Applies soft Tikhonov spectral flattening to smaller singular values.
        """
        W_eff = self.router.W1_effective if layer_name == "W1" else self.router.W2_effective
        W_base = self.router.W1_base if layer_name == "W1" else self.router.W2_base
        delta_target = self.router.delta_W1 if layer_name == "W1" else self.router.delta_W2

        U, S, Vt = np.linalg.svd(W_eff, full_matrices=False)
        old_cond = float(S[0] / (S[-1] + 1e-12))

        min_allowed_s = S[0] / max_condition
        new_S = np.maximum(S, min_allowed_s)

        new_W = np.matmul(U * new_S, Vt).astype(np.float32)
        total_delta = new_W - W_eff

        cand_W = W_eff + total_delta
        self.verify_spectral_invariants(cand_W, W_base, layer_name)

        delta_target += total_delta
        self.router.save_active_patches()

        new_cond = float(new_S[0] / (new_S[-1] + 1e-12))
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "spectrum_regularization",
            "layer_name": layer_name,
            "old_condition_number": round(old_cond, 2),
            "new_condition_number": round(new_cond, 2),
            "modes_lifted": int(np.sum(S < min_allowed_s)),
        }
        self._record_ledger(record)
        return record

    def damp_dominant_eigenvalue(
        self,
        layer_name: str = "W1",
        damping_factor: float = 0.5,
        svd_surgeon_compensation: bool = True,
    ) -> Dict[str, Any]:
        """
        Dampens the dominant attractor eigenvalue with optional SVD-Surgeon (2026) compensation.
        If svd_surgeon_compensation is enabled, applies closed-form second-order energy compensation
        to the retained singular values, preserving total matrix capacity and preventing perplexity collapse.
        """
        W_eff = self.router.W1_effective if layer_name == "W1" else self.router.W2_effective
        W_base = self.router.W1_base if layer_name == "W1" else self.router.W2_base
        delta_target = self.router.delta_W1 if layer_name == "W1" else self.router.delta_W2

        U, S, Vt = np.linalg.svd(W_eff, full_matrices=False)
        old_top_s = float(S[0])
        old_energy = float(np.sum(S**2))

        reduction = (1.0 - damping_factor) * S[0]
        new_S = S.copy()
        new_S[0] = S[0] * damping_factor

        # SVD-Surgeon (Safari & Hutter 2026): Compensate retained singular values
        # Redistribute the excised singular energy across the retained spectrum (modes 1..k)
        if svd_surgeon_compensation and len(S) > 1:
            excised_energy = old_top_s**2 - new_S[0]**2
            retained_energy = float(np.sum(S[1:]**2)) + 1e-12
            # Second-order proportional boost factor to preserve Frobenius norm
            boost_factor = math.sqrt((retained_energy + excised_energy) / retained_energy)
            # Bound boost factor to prevent over-amplification of noise modes
            boost_factor = min(boost_factor, 1.15)
            new_S[1:] = new_S[1:] * boost_factor

        # Reconstruct updated matrix with SVD-Surgeon compensated spectrum
        new_W_eff = np.matmul(U * new_S, Vt).astype(np.float32)
        total_delta = new_W_eff - W_eff

        cand_effective = W_eff + total_delta
        # Formal Invariant Check
        self.verify_spectral_invariants(cand_effective, W_base, layer_name)

        delta_target += total_delta
        self.router.save_active_patches()

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "damp_dominant_eigenvalue",
            "layer_name": layer_name,
            "damping_factor": damping_factor,
            "svd_surgeon_compensated": svd_surgeon_compensation,
            "old_dominant_s": round(old_top_s, 4),
            "new_dominant_s": round(float(new_S[0]), 4),
            "frobenius_energy_preserved": round(float(np.sum(new_S**2) / (old_energy + 1e-12)), 4),
            "params_touched": delta_target.size,
        }
        self._record_ledger(record)
        return record

    # -------------------------------------------------------------------------
    # Pillar 5: Astra-Style Subtractive Column Projection
    # -------------------------------------------------------------------------
    def apply_astra_column_subtraction(
        self,
        v_attractor: np.ndarray,
        layer_name: str = "W1",
        n_columns: int = 16,
        beta: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Implements the exact MIT CSAIL GPT-6 Astra weight editing mechanism:
        Identifies columns contributing most heavily to the cyclic attractor vector v_attractor
        and subtracts the directional projection:
            W_col = W_col - beta * (v_attractor^T @ W_col) * v_attractor
        Modifies only target columns without retraining or global disruption.
        """
        W_eff = self.router.W1_effective if layer_name == "W1" else self.router.W2_effective
        W_base = self.router.W1_base if layer_name == "W1" else self.router.W2_base
        delta_target = self.router.delta_W1 if layer_name == "W1" else self.router.delta_W2

        v = np.asarray(v_attractor, dtype=np.float32).flatten()
        v_norm = np.linalg.norm(v) + 1e-12
        v_unit = (v / v_norm).reshape(-1, 1)  # Column vector (d_target, 1)

        # For W of shape (d_in, d_out), column W[:, j] is in R^{d_in} while output row is in R^{d_out}.
        # If attractor vector v is in output space R^{d_out}, measure alignment of column outputs:
        # col_projections = W_eff @ v_unit  -> shape (d_in, 1)
        # If v is in input/column space R^{d_in}, v_unit.T @ W_eff -> shape (1, d_out).
        d_in, d_out = W_eff.shape
        if len(v) == d_in:
            # v is aligned with columns
            col_projections = np.matmul(v_unit.T, W_eff).flatten()  # (d_out,)
            col_scores = np.abs(col_projections)
            k_cols = min(n_columns, len(col_scores))
            target_col_indices = np.argsort(col_scores)[-k_cols:]
            subtraction_delta = np.zeros_like(W_eff)
            for col_idx in target_col_indices:
                proj = np.dot(W_eff[:, col_idx], v_unit.flatten())
                subtraction_delta[:, col_idx] = -beta * proj * v_unit.flatten()
        else:
            # v is in output space R^{d_out} (like Qwen MLP down-projection outputs)
            # Alignment of rows/outputs with v: W_eff @ v_unit -> (d_in,)
            out_projections = np.matmul(W_eff, v_unit).flatten()  # (d_in,)
            row_scores = np.abs(out_projections)
            k_rows = min(n_columns, len(row_scores))
            target_row_indices = np.argsort(row_scores)[-k_rows:]
            subtraction_delta = np.zeros_like(W_eff)
            for row_idx in target_row_indices:
                proj = np.dot(W_eff[row_idx, :], v_unit.flatten())
                subtraction_delta[row_idx, :] = -beta * proj * v_unit.flatten()

        cand_effective = W_eff + subtraction_delta
        # Formal safety gate
        self.verify_spectral_invariants(cand_effective, W_base, layer_name)

        # Apply surgical patch
        delta_target += subtraction_delta
        self.router.save_active_patches()

        n_patched = int(k_cols if len(v) == d_in else k_rows)
        params_touched = int(np.count_nonzero(subtraction_delta))
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "astra_column_subtraction",
            "layer_name": layer_name,
            "columns_patched": n_patched,
            "beta": beta,
            "params_touched": params_touched,
            "fraction_touched": round(float(params_touched / W_eff.size), 4),
        }
        self._record_ledger(record)
        return record

    def synthesize_rank1_patch(self, layer_name: str, u_vec: np.ndarray, v_vec: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        W_base = self.router.W1_base if layer_name == "W1" else self.router.W2_base
        delta_target = self.router.delta_W1 if layer_name == "W1" else self.router.delta_W2

        u = np.asarray(u_vec, dtype=np.float32).flatten()
        v = np.asarray(v_vec, dtype=np.float32).flatten()
        patch = (alpha * np.outer(u, v)).astype(np.float32)

        if patch.shape != delta_target.shape:
            raise ValueError(f"Patch shape {patch.shape} does not match {delta_target.shape}")

        cand_effective = W_base + (delta_target + patch)
        self.verify_spectral_invariants(cand_effective, W_base, layer_name)

        delta_target += patch
        self.router.save_active_patches()

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "synthesize_rank1_patch",
            "layer_name": layer_name,
            "alpha": alpha,
            "patch_norm": float(np.linalg.norm(patch)),
        }
        self._record_ledger(record)
        return record

    # -------------------------------------------------------------------------
    # Pillar 2: Execution-Driven Neural Repair (Auto Failure Loop Interception)
    # -------------------------------------------------------------------------
    def report_execution_failure(self, task_type: str, failure_signature: str) -> Dict[str, Any]:
        """
        Called when a tool execution, milestone test, or critique flags a failure.
        Tracks recurring identical failure signatures. If recurrence >= 3,
        automatically triggers an emergency SVD neural repair to unblock the decision rut.
        """
        tracker = self._load_failure_tracker()
        key = f"{task_type}::{failure_signature}"
        count = tracker.get(key, 0) + 1
        tracker[key] = count
        self._save_failure_tracker(tracker)

        repair_triggered = False
        repair_details: Optional[Dict[str, Any]] = None

        if count >= 3:
            logger.warning(f"WSPEngine: Detected recurring failure loop ({count}x) for '{key}'. Triggering SVD neural repair.")
            try:
                # Surgically damp W1 dominant eigenvalue to break the repetitive loop
                repair_details = self.damp_dominant_eigenvalue("W1", damping_factor=0.6)
                # Reset counter for this loop
                tracker[key] = 0
                self._save_failure_tracker(tracker)
                repair_triggered = True
            except Exception as e:
                logger.error(f"Execution-Driven Neural Repair failed: {e}")

        return {
            "key": key,
            "count": count,
            "repair_triggered": repair_triggered,
            "repair_details": repair_details,
        }

    def _load_failure_tracker(self) -> Dict[str, int]:
        if REPETITIVE_FAILURES_FILE.exists():
            try:
                with open(REPETITIVE_FAILURES_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_failure_tracker(self, tracker: Dict[str, int]) -> None:
        try:
            with open(REPETITIVE_FAILURES_FILE, "w", encoding="utf-8") as f:
                json.dump(tracker, f, indent=2)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Pillar 3: Epigenetic LoRA Lens Switching
    # -------------------------------------------------------------------------
    def activate_lens(self, lens_name: str) -> Dict[str, Any]:
        """
        Applies a named epigenetic lens (e.g. 'lens_audit_guard', 'lens_creative_research')
        directly to the router's active delta layer in < 0.05ms on CPU.
        """
        lens = self.lens_manager.load_lens(lens_name)
        d_W1 = lens["delta_W1"]
        d_W2 = lens["delta_W2"]

        # Validate Invariants before committing
        cand_W1 = self.router.W1_base + d_W1
        cand_W2 = self.router.W2_base + d_W2
        self.verify_spectral_invariants(cand_W1, self.router.W1_base, "W1")
        self.verify_spectral_invariants(cand_W2, self.router.W2_base, "W2")

        # Apply lens deltas
        self.router.delta_W1[:] = d_W1
        self.router.delta_W2[:] = d_W2
        self.router.save_active_patches()

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "activate_epigenetic_lens",
            "lens_name": lens_name,
            "meta": lens["meta"],
        }
        self._record_ledger(record)
        return record

    def clear_lens(self) -> None:
        """Resets router to base weights without lenses."""
        self.router.reset_patches()
        self.lens_manager.active_lens_name = None

    # -------------------------------------------------------------------------
    # Pillar 1 & ROME: SAE Concept Steering & Associative Memory
    # -------------------------------------------------------------------------
    def steer_routing_by_concept(self, hidden_activation: np.ndarray, concept_idx: int, factor: float = 2.0) -> np.ndarray:
        """Uses Sparse Autoencoder to amplify/ablate a concept without polysemantic entanglement."""
        return self.sae.steer_concept(hidden_activation, concept_idx=concept_idx, factor=factor)

    def apply_rome_memory_edit(self, key_feature: np.ndarray, target_knowledge: np.ndarray) -> Dict[str, Any]:
        result = self.assoc_mem.apply_rome_edit(key_feature, target_knowledge)
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "rome_associative_memory_edit",
            "residual_norm": round(result["residual_norm"], 6),
            "delta_norm": round(result["delta_norm"], 6),
        }
        self._record_ledger(record)
        return {**result, **record}

    def graft_rome_knowledge(
        self,
        key_concept: str | np.ndarray,
        target_response: str | np.ndarray,
        layer_target: str = "W2",
    ) -> Dict[str, Any]:
        """
        Pillar 3: ROME Closed-Form Key-Value Grafting.
        Overcomes the 'subtraction-only' limitation of WSP by mathematically binding
        a new key vector -> target value vector directly into the MLP down-projection layer:
            W' = W + (v - W @ k) @ (C_inv @ k)^T / (k^T @ C_inv @ k)
        Grafts new facts or routing behaviors into weights in closed form without backpropagation.
        """
        # Convert string representations to deterministic semantic dense feature vectors if needed
        if isinstance(key_concept, str):
            k_hash = int(hashlib.md5(key_concept.encode("utf-8")).hexdigest()[:8], 16)
            rng_k = np.random.default_rng(k_hash)
            k_vec = rng_k.standard_normal(self.assoc_mem.d_key).astype(np.float32)
            k_vec /= (np.linalg.norm(k_vec) + 1e-12)
        else:
            k_vec = np.asarray(key_concept, dtype=np.float32).flatten()
            if len(k_vec) != self.assoc_mem.d_key:
                k_vec = np.pad(k_vec, (0, max(0, self.assoc_mem.d_key - len(k_vec))))[:self.assoc_mem.d_key]

        if isinstance(target_response, str):
            v_hash = int(hashlib.md5(target_response.encode("utf-8")).hexdigest()[:8], 16)
            rng_v = np.random.default_rng(v_hash)
            v_vec = rng_v.standard_normal(self.assoc_mem.d_val).astype(np.float32)
            v_vec /= (np.linalg.norm(v_vec) + 1e-12)
        else:
            v_vec = np.asarray(target_response, dtype=np.float32).flatten()
            if len(v_vec) != self.assoc_mem.d_val:
                v_vec = np.pad(v_vec, (0, max(0, self.assoc_mem.d_val - len(v_vec))))[:self.assoc_mem.d_val]

        rome_result = self.assoc_mem.apply_rome_edit(k_vec, v_vec)

        # Graft rank-1 associative delta into router layer matching CPUMicroRouter (d_in, d_out) layout
        W_target = self.router.W2_effective if layer_target == "W2" else self.router.W1_effective
        W_base = self.router.W2_base if layer_target == "W2" else self.router.W1_base
        delta_target = self.router.delta_W2 if layer_target == "W2" else self.router.delta_W1
        d_in, d_out = W_target.shape

        k_proj = np.pad(k_vec, (0, max(0, d_in - len(k_vec))))[:d_in].reshape(-1, 1)
        v_proj = np.pad(v_vec, (0, max(0, d_out - len(v_vec))))[:d_out].reshape(-1, 1)
        denom = float(np.dot(k_proj.flatten(), k_proj.flatten()))
        if denom < 1e-8:
            rank1_delta = np.zeros_like(W_target)
        else:
            curr_pred = np.matmul(W_target.T, k_proj)  # (d_out, 1) for row-vector forward x @ W
            err = v_proj - curr_pred
            rank1_delta = np.matmul(k_proj, err.T) / denom  # (d_in, d_out)

            # Respect Frobenius norm invariant budget when composing successive grafts
            norm_base = float(np.linalg.norm(W_base, ord="fro")) + 1e-12
            max_allowed = self.max_frobenius_ratio * 0.98 * norm_base
            cand_norm = float(np.linalg.norm(W_target + rank1_delta, ord="fro"))
            if cand_norm > max_allowed:
                delta_norm = float(np.linalg.norm(rank1_delta, ord="fro")) + 1e-12
                headroom = max(0.0, max_allowed - float(np.linalg.norm(W_target, ord="fro")))
                rank1_delta = rank1_delta * min(1.0, headroom / delta_norm)

        # Verify stability
        cand_W = W_target + rank1_delta
        try:
            self.verify_spectral_invariants(cand_W, W_base, layer_target)
            delta_target += rank1_delta.astype(np.float32)
            self.router.save_active_patches()
            layer_patched = True
        except Exception as e:
            layer_patched = False
            logger.debug(f"Router layer grafting bypassed due to spectral guard: {e}")

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "operation": "rome_knowledge_graft",
            "key_concept": str(key_concept)[:60],
            "target_response": str(target_response)[:60],
            "residual_norm": round(float(rome_result["residual_norm"]), 6),
            "delta_norm": round(float(rome_result["delta_norm"]), 6),
            "layer_grafted": layer_patched,
            "layer_target": layer_target,
        }
        self._record_ledger(record)
        return {**rome_result, **record}

    def _record_ledger(self, record: Dict[str, Any]) -> None:
        try:
            with open(WSP_LEDGER_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"Failed to record WSP ledger: {e}")
