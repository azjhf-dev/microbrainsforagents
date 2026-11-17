"""
Aetheris Weight-Space Programming (WSP) Substrate.
Lightweight, CPU-native neural network substrate for rapid triage,
tool routing, and direct zero-gradient weight-space surgical manipulation.
Runs purely on NumPy / PyTorch CPU with zero GPU requirement (< 10 MB RAM, < 1ms inference).
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

STORE_DIR = Path("memory/store")
BASE_WEIGHTS_FILE = STORE_DIR / "wsp_base_weights.npz"
ACTIVE_PATCHES_FILE = STORE_DIR / "wsp_active_patches.npz"
WSP_LEDGER_FILE = STORE_DIR / "wsp_ledger.jsonl"


class CPUMicroRouter:
    """
    Lightweight 2-layer perceptron running on CPU.
    Input: d_in dimensional feature vector (representing task/intent embedding)
    Hidden: d_hidden dimensional representation with ReLU
    Output: d_out dimensional logits (representing tool/decision routing)
    """

    def __init__(self, d_in: int = 128, d_hidden: int = 64, d_out: int = 16, seed: int = 42):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.rng = np.random.default_rng(seed)

        # Base weights (immutable reference)
        self.W1_base: np.ndarray = self._init_weights(d_in, d_hidden)
        self.b1_base: np.ndarray = np.zeros(d_hidden, dtype=np.float32)
        self.W2_base: np.ndarray = self._init_weights(d_hidden, d_out)
        self.b2_base: np.ndarray = np.zeros(d_out, dtype=np.float32)

        # Active delta patches (W_effective = W_base + delta)
        self.delta_W1: np.ndarray = np.zeros_like(self.W1_base)
        self.delta_b1: np.ndarray = np.zeros_like(self.b1_base)
        self.delta_W2: np.ndarray = np.zeros_like(self.W2_base)
        self.delta_b2: np.ndarray = np.zeros_like(self.b2_base)

        self._ensure_storage()
        self.load_or_initialize()

    def _init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

    def _ensure_storage(self) -> None:
        STORE_DIR.mkdir(parents=True, exist_ok=True)

    def load_or_initialize(self) -> None:
        """Loads base weights and active patches if they exist, or saves initial base state."""
        if BASE_WEIGHTS_FILE.exists():
            try:
                data = np.load(BASE_WEIGHTS_FILE)
                # Ensure saved weights match current dimension configuration
                if data["W1"].shape == (self.d_in, self.d_hidden):
                    self.W1_base = data["W1"]
                    self.b1_base = data["b1"]
                    self.W2_base = data["W2"]
                    self.b2_base = data["b2"]
                else:
                    self.save_base_weights()
            except Exception as e:
                logger.warning(f"Failed to load existing base weights, reinitializing: {e}")
                self.save_base_weights()
        else:
            self.save_base_weights()

        if ACTIVE_PATCHES_FILE.exists():
            try:
                pdata = np.load(ACTIVE_PATCHES_FILE)
                if pdata["delta_W1"].shape == self.W1_base.shape:
                    self.delta_W1 = pdata["delta_W1"]
                    self.delta_b1 = pdata["delta_b1"]
                    self.delta_W2 = pdata["delta_W2"]
                    self.delta_b2 = pdata["delta_b2"]
                else:
                    self.reset_patches()
            except Exception as e:
                logger.warning(f"Failed to load active patches: {e}")
                self.reset_patches()

    def save_base_weights(self) -> None:
        """Saves base weights to disk."""
        np.savez_compressed(
            BASE_WEIGHTS_FILE,
            W1=self.W1_base,
            b1=self.b1_base,
            W2=self.W2_base,
            b2=self.b2_base,
        )

    def save_active_patches(self) -> None:
        """Saves active delta patches to disk."""
        np.savez_compressed(
            ACTIVE_PATCHES_FILE,
            delta_W1=self.delta_W1,
            delta_b1=self.delta_b1,
            delta_W2=self.delta_W2,
            delta_b2=self.delta_b2,
        )

    def reset_patches(self) -> None:
        """Rolls back all active patches to base state."""
        self.delta_W1.fill(0)
        self.delta_b1.fill(0)
        self.delta_W2.fill(0)
        self.delta_b2.fill(0)
        self.save_active_patches()

    @property
    def W1_effective(self) -> np.ndarray:
        return self.W1_base + self.delta_W1

    @property
    def b1_effective(self) -> np.ndarray:
        return self.b1_base + self.delta_b1

    @property
    def W2_effective(self) -> np.ndarray:
        return self.W2_base + self.delta_W2

    @property
    def b2_effective(self) -> np.ndarray:
        return self.b2_base + self.delta_b2

    def forward(self, x: np.ndarray, use_patches: bool = True) -> np.ndarray:
        """
        Forward inference on CPU (< 0.1ms).
        x: (batch_size, d_in) or (d_in,)
        """
        x_in = np.asarray(x, dtype=np.float32)
        if x_in.ndim == 1:
            x_in = x_in[np.newaxis, :]

        if use_patches:
            W1, b1 = self.W1_effective, self.b1_effective
            W2, b2 = self.W2_effective, self.b2_effective
        else:
            W1, b1 = self.W1_base, self.b1_base
            W2, b2 = self.W2_base, self.b2_base

        # Layer 1: ReLU(x @ W1 + b1)
        h = np.maximum(0.0, np.matmul(x_in, W1) + b1)
        # Layer 2: logits = h @ W2 + b2
        logits = np.matmul(h, W2) + b2
        return logits.squeeze(0) if x.ndim == 1 else logits

    def route_decision(self, feature_vec: np.ndarray) -> int:
        """Returns the top predicted routing category index."""
        logits = self.forward(feature_vec, use_patches=True)
        return int(np.argmax(logits))


class ROMEAssociativeMemory:
    """
    Linear Associative Key-Value Memory Matrix.
    W_mem in R^{d_out x d_in} where Memory output y = W_mem @ k
    Supports Rank-One Model Editing (ROME) closed-form zero-gradient updates:
        Delta W = (v - W @ k) @ (C^{-1} @ k)^T / (k^T @ C^{-1} @ k)
    """

    def __init__(self, d_key: int = 64, d_val: int = 64, seed: int = 42):
        self.d_key = d_key
        self.d_val = d_val
        rng = np.random.default_rng(seed)
        self.W_mem = rng.standard_normal((d_val, d_key), dtype=np.float32) * 0.05
        # Covariance matrix C initialized to Identity * lambda
        self.C_inv = np.eye(d_key, dtype=np.float32)

    def retrieve(self, key_vector: np.ndarray) -> np.ndarray:
        """Associative recall: y = W_mem @ k."""
        k = np.asarray(key_vector, dtype=np.float32)
        return np.matmul(self.W_mem, k)

    def apply_rome_edit(self, key: np.ndarray, target_val: np.ndarray) -> Dict[str, Any]:
        """
        Closed-form Rank-One Model Edit (ROME).
        Surgically binds `key` -> `target_val` exactly without modifying other orthogonal concepts.
        """
        k = np.asarray(key, dtype=np.float32).reshape(-1, 1)  # (d_key, 1)
        v = np.asarray(target_val, dtype=np.float32).reshape(-1, 1)  # (d_val, 1)

        # Current prediction
        curr_v = np.matmul(self.W_mem, k)
        err = v - curr_v  # (d_val, 1)

        # Closed form update: Delta W = err @ (C_inv @ k)^T / (k^T @ C_inv @ k)
        c_k = np.matmul(self.C_inv, k)  # (d_key, 1)
        denom = max(float(np.matmul(k.T, c_k)[0, 0]), 1e-6)

        delta_W = np.matmul(err, c_k.T) / denom

        # Apply surgical update
        self.W_mem += delta_W

        # Update covariance inverse via Sherman-Morrison and maintain symmetry / PD floor
        self.C_inv -= np.matmul(c_k, c_k.T) / (1.0 + denom)
        self.C_inv = (
            0.5 * (self.C_inv + self.C_inv.T)
            + 1e-6 * np.eye(self.d_key, dtype=np.float32)
        ).astype(np.float32)

        # Verify recall error
        new_v = np.matmul(self.W_mem, k)
        residual_norm = float(np.linalg.norm(v - new_v))

        return {
            "status": "success",
            "residual_norm": residual_norm,
            "delta_norm": float(np.linalg.norm(delta_W)),
            "delta_max": float(np.max(np.abs(delta_W))),
        }
