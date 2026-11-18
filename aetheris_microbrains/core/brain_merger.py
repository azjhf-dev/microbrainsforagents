"""
Pillar 4: TIES-Merging & SLERP Neuro-Brain Fusion Engine
========================================================
Inspired by 2025-2026 SOTA research:
- TIES-Merging (Tris, Sign Election, and Disjoint Merge, ICLR 2025/2026)
- SLERP (Spherical Linear Interpolation in Weight Space)
- Task Vector Arithmetic for Compact Language Models

Merges two or more specialized analytical micro-transformers into a unified
multi-task neuro-brain without gradient descent and without parameter interference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer
from .brain_nursery import AutonomousBrainNursery, STORE_DIR
from .topological_persistence_merger import TopologicalPersistenceMerger

logger = logging.getLogger(__name__)


class NeuroBrainMerger:
    """Fuses distinct specialized micro-transformers using SOTA weight-space interpolation."""

    _instance: Optional[NeuroBrainMerger] = None

    def __init__(self):
        self.nursery = AutonomousBrainNursery.get_instance()

    @classmethod
    def get_instance(cls) -> NeuroBrainMerger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _pad_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pads two arrays with zeros along their trailing dimensions so their shapes match."""
        if a.shape == b.shape:
            return a, b
        max_shape = tuple(max(s_a, s_b) for s_a, s_b in zip(a.shape, b.shape))
        pad_a = [(0, max_shape[i] - a.shape[i]) for i in range(len(a.shape))]
        pad_b = [(0, max_shape[i] - b.shape[i]) for i in range(len(b.shape))]
        return np.pad(a, pad_a), np.pad(b, pad_b)

    @staticmethod
    def _extract_symbolic_rules(brain: AnalyticalMicroTransformer) -> List[Tuple[str, str, float]]:
        """Decode (trigger_token, target_token, binding_gain) triples using parent brain's embedding basis."""
        rules: List[Tuple[str, str, float]] = []
        if not brain.hopfield_keys or not brain.hopfield_values:
            return rules
        n_v = min(len(brain.vocab), len(brain.W_embed))
        if n_v == 0:
            return rules
        for hk, hv in zip(brain.hopfield_keys, brain.hopfield_values):
            min_d = min(brain.d_model, len(hk), len(hv), brain.W_embed.shape[1])
            emb_slice = brain.W_embed[:n_v, :min_d]
            emb_norms = np.linalg.norm(emb_slice, axis=-1, keepdims=True) + 1e-12
            emb_unit = emb_slice / emb_norms
            k_idx = int(np.argmax(np.dot(emb_unit, hk[:min_d])))
            v_idx = int(np.argmax(np.dot(emb_unit, hv[:min_d])))
            k_tok = brain.inv_vocab.get(k_idx, "<unk>")
            v_tok = brain.inv_vocab.get(v_idx, "<unk>")
            gain = max(1e-3, float(np.linalg.norm(hv)))
            if k_tok not in ("<pad>", "<unk>", "<eos>") and v_tok not in ("<pad>", "<unk>", "<eos>"):
                rules.append((k_tok, v_tok, gain))
        return rules

    @staticmethod
    def _slerp(t: float, v0: np.ndarray, v1: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Spherical Linear Interpolation between two weight vectors."""
        v0, v1 = NeuroBrainMerger._pad_to_match(v0, v1)
        v0_flat = v0.flatten()
        v1_flat = v1.flatten()

        v0_norm = np.linalg.norm(v0_flat) + eps
        v1_norm = np.linalg.norm(v1_flat) + eps

        u0 = v0_flat / v0_norm
        u1 = v1_flat / v1_norm

        dot = np.clip(np.dot(u0, u1), -1.0, 1.0)
        omega = np.arccos(dot)

        if np.abs(omega) < eps:
            return ((1.0 - t) * v0 + t * v1).astype(v0.dtype)

        sin_omega = np.sin(omega)
        s0 = np.sin((1.0 - t) * omega) / sin_omega
        s1 = np.sin(t * omega) / sin_omega

        res_flat = (s0 * v0_norm * u0 + s1 * v1_norm * u1)
        return res_flat.reshape(v0.shape).astype(v0.dtype)

    def merge_two_brains_slerp(
        self,
        brain_name_a: str,
        brain_name_b: str,
        fused_brain_name: str,
        interpolation_weight: float = 0.5,
    ) -> Optional[AnalyticalMicroTransformer]:
        """
        Merges two micro-transformers along the geodesic of high-dimensional weight space using SLERP.
        Preserves vector magnitudes and directional knowledge of both models.
        """
        clean_a = brain_name_a.replace("neuro_", "")
        clean_b = brain_name_b.replace("neuro_", "")

        b_a = self.nursery.active_brains.get(f"neuro_{clean_a}")
        b_b = self.nursery.active_brains.get(f"neuro_{clean_b}")

        if not b_a or not b_b:
            logger.warning(f"Could not find brains {brain_name_a} or {brain_name_b} for SLERP merge")
            return None

        # Instantiate target fused brain with max dimensions
        max_d_model = max(b_a.d_model, b_b.d_model)
        max_d_mlp = max(b_a.d_mlp, b_b.d_mlp)
        max_n_heads = max(b_a.n_heads, b_b.n_heads)

        fused = AnalyticalMicroTransformer(
            name=f"neuro_{fused_brain_name.replace('neuro_', '')}",
            d_model=max_d_model,
            n_heads=max_n_heads,
            d_mlp=max_d_mlp,
        )

        # Merge Vocabularies first so all tokens exist in fused coordinate basis
        for k in b_a.vocab:
            fused.register_token(k)
        for k in b_b.vocab:
            fused.register_token(k)

        t = interpolation_weight
        # Merge Attention weights
        fused.W_Q = self._slerp(t, b_a.W_Q, b_b.W_Q)
        fused.W_K = self._slerp(t, b_a.W_K, b_b.W_K)
        fused.W_V = self._slerp(t, b_a.W_V, b_b.W_V)
        fused.W_O = self._slerp(t, b_a.W_O, b_b.W_O)

        # Merge MLP weights & biases
        fused.W_up = self._slerp(t, b_a.W_up, b_b.W_up)
        fused.W_down = self._slerp(t, b_a.W_down, b_b.W_down)
        fused.b_up = ((1.0 - t) * b_a.b_up[:max_d_mlp] if len(b_a.b_up) >= max_d_mlp else np.pad(b_a.b_up, (0, max_d_mlp - len(b_a.b_up)))) + (
            t * b_b.b_up[:max_d_mlp] if len(b_b.b_up) >= max_d_mlp else np.pad(b_b.b_up, (0, max_d_mlp - len(b_b.b_up)))
        )
        fused.b_down = ((1.0 - t) * b_a.b_down[:max_d_model] if len(b_a.b_down) >= max_d_model else np.pad(b_a.b_down, (0, max_d_model - len(b_a.b_down)))) + (
            t * b_b.b_down[:max_d_model] if len(b_b.b_down) >= max_d_model else np.pad(b_b.b_down, (0, max_d_model - len(b_b.b_down)))
        )

        # Merge W_mlp under Topological Persistence Betti-1 Homology Guard
        w_mlp_a = getattr(b_a, "W_mlp", None)
        w_mlp_b = getattr(b_b, "W_mlp", None)
        if isinstance(w_mlp_a, np.ndarray) and isinstance(w_mlp_b, np.ndarray):
            w_mlp_a_pad, w_mlp_b_pad = self._pad_to_match(w_mlp_a, w_mlp_b)
            topo_guard = TopologicalPersistenceMerger.get_instance().safe_slerp_merge(
                w_mlp_a_pad, w_mlp_b_pad, interpolation_t=t
            )
            fused.W_mlp = np.asarray(topo_guard["fused_weights"], dtype=np.float32).reshape(max_d_model, max_d_model)
            fused.last_merge_topology = {
                k: topo_guard[k] for k in ("status", "is_topologically_conserved", "parent_a_betti1", "parent_b_betti1", "fused_betti1")
            }

        # Re-compile extracted symbolic rules into fused brain's unified W_embed coordinate basis
        for src_b in (b_a, b_b):
            extracted = self._extract_symbolic_rules(src_b)
            if extracted:
                for k_tok, v_tok, gain in extracted:
                    fused.compile_logic_association(k_tok, v_tok, binding_gain=gain)
            else:
                for hk, hv in zip(src_b.hopfield_keys, src_b.hopfield_values):
                    hk_pad = hk[:max_d_model] if len(hk) >= max_d_model else np.pad(hk, (0, max_d_model - len(hk)))
                    hv_pad = hv[:max_d_model] if len(hv) >= max_d_model else np.pad(hv, (0, max_d_model - len(hv)))
                    fused.hopfield_keys.append(hk_pad.astype(np.float32))
                    fused.hopfield_values.append(hv_pad.astype(np.float32))

        fused.save()
        self.nursery.active_brains[fused.name] = fused
        return fused

    def merge_brains_ties(
        self,
        brain_names: List[str],
        fused_brain_name: str,
        density_threshold: float = 0.25,
    ) -> Optional[AnalyticalMicroTransformer]:
        """
        Applies TIES-Merging (Trim, Elect Sign, and Disjoint Merge) across multiple neuro-brains.
        1. Trim: Drops low-magnitude noise updates below density threshold.
        2. Elect: Elects dominant sign direction across model parameters.
        3. Disjoint Merge: Averages only parameters that agree with the elected sign.
        """
        brains = []
        for name in brain_names:
            clean = name.replace("neuro_", "")
            b = self.nursery.active_brains.get(f"neuro_{clean}")
            if b:
                brains.append(b)

        if len(brains) < 2:
            logger.warning("TIES-Merging requires at least 2 valid neuro-brains")
            return None

        max_d_model = max(b.d_model for b in brains)
        max_d_mlp = max(b.d_mlp for b in brains)
        max_n_heads = max(b.n_heads for b in brains)

        fused = AnalyticalMicroTransformer(
            name=f"neuro_{fused_brain_name.replace('neuro_', '')}",
            d_model=max_d_model,
            n_heads=max_n_heads,
            d_mlp=max_d_mlp,
        )

        for b in brains:
            for k in b.vocab:
                fused.register_token(k)

        def _ties_tensor_merge(tensors: List[np.ndarray]) -> np.ndarray:
            max_shape = tuple(max(t.shape[i] for t in tensors) for i in range(len(tensors[0].shape)))
            padded_tensors = []
            for t in tensors:
                pad = [(0, max_shape[i] - t.shape[i]) for i in range(len(t.shape))]
                padded_tensors.append(np.pad(t, pad))

            stacked = np.stack(padded_tensors, axis=0)  # (N, ...)
            # 1. Trim: retain top density_threshold parameters by magnitude
            trimmed = np.zeros_like(stacked)
            for i in range(len(padded_tensors)):
                t = stacked[i]
                abs_t = np.abs(t)
                cutoff = np.quantile(abs_t, 1.0 - density_threshold)
                trimmed[i] = np.where(abs_t >= cutoff, t, 0.0)

            # 2. Elect Sign: Majority sign voting across models
            sign_consensus = np.sign(np.sum(np.sign(trimmed), axis=0))
            sign_consensus = np.where(sign_consensus == 0, 1.0, sign_consensus)

            # 3. Disjoint Merge: Average only tensors aligning with consensus
            aligned_mask = (np.sign(trimmed) == sign_consensus) & (trimmed != 0.0)
            counts = np.sum(aligned_mask, axis=0)
            sums = np.sum(np.where(aligned_mask, trimmed, 0.0), axis=0)

            merged = np.divide(sums, np.maximum(counts, 1), out=np.zeros_like(sums))
            return merged.astype(tensors[0].dtype)

        fused.W_Q = _ties_tensor_merge([b.W_Q for b in brains])
        fused.W_K = _ties_tensor_merge([b.W_K for b in brains])
        fused.W_V = _ties_tensor_merge([b.W_V for b in brains])
        fused.W_O = _ties_tensor_merge([b.W_O for b in brains])
        fused.W_up = _ties_tensor_merge([b.W_up for b in brains])
        fused.W_down = _ties_tensor_merge([b.W_down for b in brains])
        w_mlps = [b.W_mlp for b in brains if isinstance(getattr(b, "W_mlp", None), np.ndarray)]
        if len(w_mlps) == len(brains):
            fused.W_mlp = _ties_tensor_merge(w_mlps)

        for b in brains:
            extracted = self._extract_symbolic_rules(b)
            if extracted:
                for k_tok, v_tok, gain in extracted:
                    fused.compile_logic_association(k_tok, v_tok, binding_gain=gain)
            else:
                for hk, hv in zip(b.hopfield_keys, b.hopfield_values):
                    hk_pad = hk[:max_d_model] if len(hk) >= max_d_model else np.pad(hk, (0, max_d_model - len(hk)))
                    hv_pad = hv[:max_d_model] if len(hv) >= max_d_model else np.pad(hv, (0, max_d_model - len(hv)))
                    fused.hopfield_keys.append(hk_pad.astype(np.float32))
                    fused.hopfield_values.append(hv_pad.astype(np.float32))

        fused.save()
        self.nursery.active_brains[fused.name] = fused
        return fused
