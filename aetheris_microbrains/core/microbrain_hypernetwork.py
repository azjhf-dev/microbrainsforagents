"""
Aetheris Zero-Shot Fast-Weight Hypernetwork Engine.
Inspired by Cambridge / Sakana AI 2025-2026 Doc-to-LoRA and fast-weight architectures.
Generates normalized d=64 micro-brain weight tensors in a single forward pass from
task/tool descriptions on CPU.

Honesty note: the generator projection (W_gen1/W_gen2) is FIXED and never trained -
synthesized weights are a deterministic hash-keyed random function of the description,
NOT semantically specialized (unrelated task descriptions produce ~0.5-correlated
weight matrices).  Only W_ttt_fast, updated by the TTT / Titans-Muon adaptation
paths below, carries actual task adaptation.
"""

from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class MicroBrainHypernetwork:
    """Generates dynamic neural weights for specialized micro-brains in one pass without backprop."""

    _instance: Optional[MicroBrainHypernetwork] = None

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 32):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        # Deterministic projection basis for the hypernetwork generator
        rng = np.random.RandomState(42)
        self.W_gen1 = rng.randn(latent_dim, hidden_dim).astype(np.float32) / math.sqrt(latent_dim)
        self.W_gen2 = rng.randn(hidden_dim, latent_dim * latent_dim).astype(np.float32) / math.sqrt(hidden_dim)
        # TTT-Linear fast-weight adaptation matrix (updated online at test time on GPU/CPU)
        self.W_ttt_fast = np.eye(latent_dim, dtype=np.float32)
        self.titans_momentum = np.zeros_like(self.W_ttt_fast)
        self.surprise_ema: float = 0.0
        self.ttt_steps: int = 0
        self.last_ttt_loss: Optional[float] = None

    def adapt_titans_muon_ttt(
        self,
        context_batch: Union[List[str], List[List[float]], np.ndarray],
        target_batch: Union[List[str], List[List[float]], np.ndarray],
        lr: float = 0.08,
        momentum_beta: float = 0.90,
        weight_decay: float = 0.005,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Titans / MIRAS Surprise-Gated & Muon-Orthogonalized Test-Time Memory Adaptation:
        1. Computes surprise S_t = mean(||W_ttt @ k - v||_2) for incoming batch (K, V).
        2. Derives surprise gate alpha_t = sigmoid(S_t - surprise_ema) and retention decay.
        3. Updates W_ttt_fast via Newton-Schulz orthogonalized Muon momentum step.
        """
        def _to_latent_matrix(batch_in: Union[List[str], List[List[float]], np.ndarray]) -> np.ndarray:
            if isinstance(batch_in, (list, tuple)) and len(batch_in) > 0 and isinstance(batch_in[0], str):
                return np.stack([self._embed_task_description(s) for s in batch_in], axis=0).astype(np.float32)
            if isinstance(batch_in, str):
                return self._embed_task_description(batch_in)[None, :].astype(np.float32)
            arr = np.asarray(batch_in, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            else:
                arr = arr.reshape(arr.shape[0], -1)
            if arr.shape[1] < self.latent_dim:
                arr = np.pad(arr, ((0, 0), (0, self.latent_dim - arr.shape[1])))
            else:
                arr = arr[:, : self.latent_dim]
            return arr.astype(np.float32)

        k_mat = _to_latent_matrix(context_batch)
        v_mat = _to_latent_matrix(target_batch)
        n = min(len(k_mat), len(v_mat))
        if n == 0:
            return {"status": "skipped_empty_batch", "ttt_steps": self.ttt_steps}
        k_mat = k_mat[:n]
        v_mat = v_mat[:n]

        preds_pre = np.dot(k_mat, self.W_ttt_fast.T)
        residuals = preds_pre - v_mat
        surprise_metric = float(np.mean(np.sqrt(np.mean(residuals ** 2, axis=-1))))
        alpha_t = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, surprise_metric - self.surprise_ema))))

        # Apply Titans forgetting/retention gate prior to step only on previously adapted memory
        if self.ttt_steps > 0 and weight_decay > 0.0:
            retention = max(0.90, 1.0 - float(weight_decay) * alpha_t)
            w_start = (self.W_ttt_fast * retention).astype(np.float32)
        else:
            w_start = self.W_ttt_fast

        from core.opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        muon_res = acc.muon_fast_weight_step(
            w_fast=w_start,
            k_batch=k_mat,
            v_batch=v_mat,
            momentum_buf=self.titans_momentum,
            lr=float(lr) * float(0.5 + alpha_t),
            momentum=momentum_beta,
            force_gpu=force_gpu,
        )
        self.W_ttt_fast = muon_res["updated_weights"]
        self.titans_momentum = muon_res["momentum_buf"]
        self.surprise_ema = 0.8 * self.surprise_ema + 0.2 * surprise_metric
        self.ttt_steps += 1
        self.last_ttt_loss = float(muon_res["post_update_loss"])

        return {
            "status": "titans_muon_adapted",
            "backend": muon_res["backend"],
            "ttt_steps": self.ttt_steps,
            "surprise_metric": round(surprise_metric, 6),
            "adaptive_gate_alpha": round(alpha_t, 4),
            "pre_loss": muon_res["pre_update_loss"],
            "post_loss": muon_res["post_update_loss"],
            "loss_reduced": bool(muon_res["loss_reduced"]),
            "spectral_isotropy_ratio": muon_res.get("spectral_isotropy_ratio", 1.0),
        }

    def adapt_test_time_ttt(
        self,
        context_spec: str,
        outcome_spec: str,
        lr: float = 0.05,
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Test-Time Training (TTT-Linear) Online Fast-Weight Adaptation:
        Updates `self.W_ttt_fast` in-place via self-supervised reconstruction loss
        ||W_ttt_fast @ k - v||^2 on the OpenCL GPU co-processor (or CPU fallback).

        Currently has no production caller - the wired adaptation path is
        `adapt_titans_muon_ttt` (microbrain_lab action hypernet_titans_muon).
        """
        k_vec = self._embed_task_description(context_spec)
        v_vec = self._embed_task_description(outcome_spec)
        from core.opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        self.W_ttt_fast, loss, backend = acc.ttt_linear_step(
            self.W_ttt_fast, k_vec, v_vec, lr=lr, force_gpu=force_gpu
        )
        self.ttt_steps += 1
        self.last_ttt_loss = round(float(loss), 6)
        return {
            "status": "ttt_adapted",
            "backend": backend,
            "ttt_steps": self.ttt_steps,
            "reconstruction_loss": self.last_ttt_loss,
        }

    @classmethod
    def get_instance(cls) -> MicroBrainHypernetwork:
        if cls._instance is None:
            cls._instance = MicroBrainHypernetwork()
        return cls._instance

    def _embed_task_description(self, task_spec: str) -> np.ndarray:
        """Embeds task text into a normalized d=64 conditioning vector."""
        vec = np.zeros(self.latent_dim, dtype=np.float32)
        tokens = task_spec.lower().split()
        for idx, tok in enumerate(tokens):
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            pos = h % self.latent_dim
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vec[pos] += sign / (1.0 + math.sqrt(idx + 1))
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return vec

    def synthesize_weights(self, task_description: str, out_dim: int = 64, in_dim: int = 64) -> Dict[str, Any]:
        """
        Hypernetwork Forward Pass:
        z = TaskEmbed(task_description)
        h = ReLU(z @ W_gen1)
        flat_weights = h @ W_gen2
        W = Reshape(flat_weights, (out_dim, in_dim))

        The generator is an UNTRAINED fixed random projection: the output is a
        deterministic function of the task text's hash embedding, disclosed via
        `generator_trained: False` in the result.
        """
        t0 = time.perf_counter()
        z_raw = self._embed_task_description(task_description)
        z = np.dot(self.W_ttt_fast, z_raw) if self.ttt_steps > 0 else z_raw
        h = np.maximum(0.0, np.dot(z, self.W_gen1))
        flat_w = np.dot(h, self.W_gen2)

        # Reshape to requested dimensions
        needed_size = out_dim * in_dim
        if len(flat_w) >= needed_size:
            w_matrix = flat_w[:needed_size].reshape((out_dim, in_dim))
        else:
            w_matrix = np.pad(flat_w, (0, needed_size - len(flat_w))).reshape((out_dim, in_dim))

        # Spectral normalization to guarantee numerical stability
        u, s, vt = np.linalg.svd(w_matrix, full_matrices=False)
        spectral_norm = float(s[0]) if len(s) > 0 and s[0] > 1e-9 else 1.0
        w_stable = (w_matrix / spectral_norm) * 0.95

        # Also generate ternary BitNet b1.58 representation
        gamma = float(np.mean(np.abs(w_stable))) or 1e-9
        w_ternary = np.clip(np.round(w_stable / gamma), -1, 1).astype(np.int8)

        return {
            "status": "synthesized_success",
            "generator_trained": False,
            "generator_note": (
                "fixed random projection over a hash embedding (no semantic "
                "specialization); only the TTT-updated W_ttt_fast adapts"
            ),
            "task_snippet": task_description[:80],
            "weights_shape": [out_dim, in_dim],
            "spectral_norm": round(spectral_norm, 4),
            "stable_weights": w_stable.tolist(),
            "ternary_weights": w_ternary.tolist(),
            "generation_latency_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }

