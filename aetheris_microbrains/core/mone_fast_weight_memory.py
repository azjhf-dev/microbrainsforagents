"""
MoNe: Modular Fast-Weight Neural Memory for Aetheris.
Ported and evolved from Qualcomm AI Research SOTA (Yun et al., August 2026).

Replaces flat context window tokens with layer-localized fast-weight outer products:
1. Fast Weights Matrix W_fast: Stores sequential streaming context at O(1) query time.
2. Dynamic Outer-Product Ingestion: W_fast = lambda * W_fast + eta * (x outer y).
3. Associative Retrieval: Recalls bound context without quadratic O(N^2) Transformer attention.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

log = logging.getLogger("aetheris.mone_fast_weights")


@dataclass
class MoNeRetrievalResult:
    recalled_vector: np.ndarray
    confidence: float
    memory_norm: float
    step_count: int


class MoNeFastWeightMemory:
    """
    Lightweight modular neural memory with test-time fast-weight updates.
    """

    _instance: Optional["MoNeFastWeightMemory"] = None

    def __init__(self, dim: int = 128, decay_lambda: float = 0.95, learning_rate: float = 0.15, momentum_alpha: float = 0.85):
        self.dim = dim
        self.decay_lambda = decay_lambda
        self.learning_rate = learning_rate
        self.momentum_alpha = momentum_alpha
        self.W_fast = np.zeros((dim, dim), dtype=np.float64)
        self.S_momentum = np.zeros((dim, dim), dtype=np.float64)
        self.step_counter = 0

    @classmethod
    def get_instance(cls) -> "MoNeFastWeightMemory":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def clear(self):
        """Reset fast-weight state."""
        self.W_fast.fill(0.0)
        self.S_momentum.fill(0.0)
        self.step_counter = 0

    def ingest_context_pair(self, key_vector: np.ndarray, value_vector: np.ndarray) -> float:
        """
        Updates fast-weight memory with key-value association:
        W_fast <- lambda * W_fast + eta * (v_val outer v_key)
        Normalized to prevent runaway norms.
        """
        k = np.asarray(key_vector, dtype=np.float64).flatten()
        v = np.asarray(value_vector, dtype=np.float64).flatten()

        # Ensure correct dimensionality via truncation or zero-padding
        k_padded = np.zeros(self.dim, dtype=np.float64)
        v_padded = np.zeros(self.dim, dtype=np.float64)
        k_len = min(len(k), self.dim)
        v_len = min(len(v), self.dim)
        k_padded[:k_len] = k[:k_len]
        v_padded[:v_len] = v[:v_len]

        # Normalize key vector
        norm_k = np.linalg.norm(k_padded)
        if norm_k > 1e-9:
            k_padded /= norm_k

        # Outer product update
        outer_prod = np.outer(v_padded, k_padded)
        self.W_fast = self.decay_lambda * self.W_fast + self.learning_rate * outer_prod
        self.step_counter += 1

        current_norm = float(np.linalg.norm(self.W_fast))
        return current_norm

    def ingest_delta_rule(
        self,
        key_vector: Union[List[float], np.ndarray],
        value_vector: Union[List[float], np.ndarray],
        beta_gate: float = 1.0,
        momentum_beta: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Gated Delta-Rule Fast-Weight Memory Update with Surprise-Driven Momentum
        (Titans / DeltaNet SOTA 2025-2026):
        1. Decay existing memory: W_decayed = lambda * W_fast
        2. Predict current retrieval error (surprise gradient): error = v - W_decayed @ k_hat
        3. Rank-1 error correction: outer_grad = error outer k_hat
        4. Optional momentum accumulation (when momentum_beta > 0):
           S_momentum = momentum_beta * S_momentum + outer_grad
        5. Apply update: W_fast = W_decayed + beta_gate * (outer_grad + momentum_beta * S_prev)
        Guarantees exact post-recall cosine > 0.99 for one-shot overwriting when momentum_beta=0.
        """
        k = np.asarray(key_vector, dtype=np.float64).flatten()
        v = np.asarray(value_vector, dtype=np.float64).flatten()

        k_padded = np.zeros(self.dim, dtype=np.float64)
        v_padded = np.zeros(self.dim, dtype=np.float64)
        k_padded[: min(len(k), self.dim)] = k[: min(len(k), self.dim)]
        v_padded[: min(len(v), self.dim)] = v[: min(len(v), self.dim)]

        norm_k = float(np.linalg.norm(k_padded))
        if norm_k > 1e-9:
            k_padded /= norm_k

        w_decayed = self.decay_lambda * self.W_fast
        v_pred = np.dot(w_decayed, k_padded)
        error = v_padded - v_pred
        error_norm = float(np.linalg.norm(error))

        outer_grad = np.outer(error, k_padded)
        if momentum_beta > 0.0:
            self.S_momentum = float(momentum_beta) * self.S_momentum + outer_grad
            update_matrix = outer_grad + float(momentum_beta) * self.S_momentum
        else:
            self.S_momentum = outer_grad.copy()
            update_matrix = outer_grad

        self.W_fast = w_decayed + float(beta_gate) * update_matrix
        self.step_counter += 1

        recalled = np.dot(self.W_fast, k_padded)
        norm_rec = float(np.linalg.norm(recalled))
        norm_v = float(np.linalg.norm(v_padded))
        if norm_rec > 1e-9 and norm_v > 1e-9:
            cos_sim = float(np.dot(recalled, v_padded) / (norm_rec * norm_v))
        else:
            cos_sim = 1.0 if norm_rec <= 1e-9 and norm_v <= 1e-9 else 0.0

        return {
            "status": "mone_delta_updated",
            "prediction_error_norm": round(error_norm, 6),
            "post_recall_cosine": round(cos_sim, 4),
            "memory_norm": round(float(np.linalg.norm(self.W_fast)), 4),
            "momentum_norm": round(float(np.linalg.norm(self.S_momentum)), 4),
            "step_count": self.step_counter,
        }

    def retrieve(self, query_key: np.ndarray) -> MoNeRetrievalResult:
        """
        Retrieves associated value at O(1) time complexity:
        y = W_fast * query_key
        """
        q = np.asarray(query_key, dtype=np.float64).flatten()
        q_padded = np.zeros(self.dim, dtype=np.float64)
        q_len = min(len(q), self.dim)
        q_padded[:q_len] = q[:q_len]

        norm_q = np.linalg.norm(q_padded)
        if norm_q > 1e-9:
            q_padded /= norm_q

        recalled = np.dot(self.W_fast, q_padded)
        norm_out = float(np.linalg.norm(recalled))
        norm_w = float(np.linalg.norm(self.W_fast))

        # Confidence metric derived from output SNR
        confidence = min(1.0, norm_out / max(1e-9, norm_w * 0.5)) if norm_w > 1e-9 else 0.0

        return MoNeRetrievalResult(
            recalled_vector=recalled,
            confidence=round(confidence, 4),
            memory_norm=round(norm_w, 4),
            step_count=self.step_counter
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "step_counter": self.step_counter,
            "memory_norm": round(float(np.linalg.norm(self.W_fast)), 4),
            "decay_lambda": self.decay_lambda,
            "learning_rate": self.learning_rate,
        }

