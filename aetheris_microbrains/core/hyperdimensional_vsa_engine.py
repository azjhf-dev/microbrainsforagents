"""
Aetheris Hyperdimensional Vector Symbolic Architecture (HD-VSA) Engine.
Inspired by Kanerva (Hyperdimensional Computing) and holographic reduced representations (2025-2026).
Enables zero-shot tool composition and memory binding via algebraic vector operations:
- Binding: circular convolution or elementwise multiply (x * y)
- Bundling: normalized superposition (x + y)
- Clean-up: nearest neighbor cosine resonance in < 1 microsecond.
"""

from __future__ import annotations
import math
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class HyperdimensionalVSAEngine:
    """Performs holographic composition and algebraic binding of symbolic tool vectors."""

    _instance: Optional[HyperdimensionalVSAEngine] = None

    def __init__(self, hd_dim: int = 128):
        self.hd_dim = hd_dim
        self.symbol_memory: Dict[str, np.ndarray] = {}

    @classmethod
    def get_instance(cls) -> HyperdimensionalVSAEngine:
        if cls._instance is None:
            cls._instance = HyperdimensionalVSAEngine()
        return cls._instance

    def get_or_create_symbol(self, name: str) -> np.ndarray:
        """Generates or retrieves a deterministic bipolar hypervector (-1 or +1) for a symbol."""
        if name in self.symbol_memory:
            return self.symbol_memory[name]

        # Generate deterministic pseudo-random bipolar vector
        vec = np.zeros(self.hd_dim, dtype=np.float32)
        for i in range(self.hd_dim):
            h = int(hashlib.sha256(f"{name}_{i}".encode()).hexdigest(), 16)
            vec[i] = 1.0 if (h % 2 == 0) else -1.0

        # Normalize
        norm = np.linalg.norm(vec)
        vec = vec / norm
        self.symbol_memory[name] = vec
        return vec

    def bind_symbols(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """
        Binding operator (association / role-filler binding):
        Uses circular convolution in frequency domain for reversible associative binding.
        """
        fft_a = np.fft.fft(vec_a)
        fft_b = np.fft.fft(vec_b)
        bound = np.fft.ifft(fft_a * fft_b).real.astype(np.float32)
        norm = np.linalg.norm(bound)
        return bound / (norm if norm > 1e-9 else 1.0)

    def unbind_symbols(self, composite: np.ndarray, key_vec: np.ndarray) -> np.ndarray:
        """
        Approximate inverse of circular convolution via circular correlation:
        unbind(c, b) = IFFT(FFT(c) * conj(FFT(b)))
        """
        fft_c = np.fft.fft(composite)
        fft_k = np.fft.fft(key_vec)
        unbound = np.fft.ifft(fft_c * np.conj(fft_k)).real.astype(np.float32)
        norm = float(np.linalg.norm(unbound))
        return unbound / (norm if norm > 1e-9 else 1.0)

    def bundle_symbols(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundling operator (set superposition):
        Superimposes multiple vectors: V_bundled = norm(sum(v_i)).
        """
        if not vectors:
            return np.zeros(self.hd_dim, dtype=np.float32)
        total = np.sum(vectors, axis=0)
        norm = np.linalg.norm(total)
        return total / (norm if norm > 1e-9 else 1.0)

    def compose_tool_pipeline(self, tool_names: List[str], role_tags: List[str]) -> Dict[str, Any]:
        """Composes a multi-step tool pipeline into a single holographic hypervector."""
        t0 = time.perf_counter()
        bound_pairs = []
        for t, r in zip(tool_names, role_tags):
            v_t = self.get_or_create_symbol(t)
            v_r = self.get_or_create_symbol(r)
            bound = self.bind_symbols(v_t, v_r)
            bound_pairs.append(bound)

        holographic_representation = self.bundle_symbols(bound_pairs)

        # Unbind each role tag before checking tool resonance
        resonances = {}
        for t, r in zip(tool_names, role_tags):
            vt = self.get_or_create_symbol(t)
            vr = self.get_or_create_symbol(r)
            unbound = self.unbind_symbols(holographic_representation, vr)
            cos = float(np.dot(unbound, vt))
            resonances[t] = round(cos, 4)

        return {
            "status": "composed_success",
            "tools_bundled": tool_names,
            "roles_bound": role_tags,
            "hd_dim": self.hd_dim,
            "holographic_vector": holographic_representation.tolist(),
            "constituent_resonances": resonances,
            "composition_latency_us": round((time.perf_counter() - t0) * 1e6, 2),
        }

    def factorize_resonator(
        self,
        composite_vec: Union[List[float], np.ndarray],
        candidate_tools: List[str],
        candidate_roles: List[str],
        max_iters: int = 15,
        noise_std: float = 0.03,
        beta: float = 12.0,
        top_k_factors: int = 1,
    ) -> Dict[str, Any]:
        """
        2025-2026 SOTA Resonator Network Factorization with Explaining-Away:
        Decodes up to `top_k_factors` bound factor pairs (tool x role) from a superposed
        composite hypervector without combinatorial search, subtracting each converged
        constituent ("explaining away") before isolating subsequent factors.
        """
        t0 = time.perf_counter()
        if not candidate_tools or not candidate_roles:
            return {"status": "skipped_empty_codebook"}

        s = np.asarray(composite_vec, dtype=np.float32).flatten()
        if len(s) < self.hd_dim:
            s = np.pad(s, (0, self.hd_dim - len(s)))
        else:
            s = s[: self.hd_dim]
        s_norm = float(np.linalg.norm(s))
        if s_norm > 1e-9:
            s = s / s_norm

        t_mat = np.stack([self.get_or_create_symbol(t) for t in candidate_tools], axis=0)
        r_mat = np.stack([self.get_or_create_symbol(r) for r in candidate_roles], axis=0)

        s_residual = s.copy()
        decoded_factors: List[Dict[str, Any]] = []
        n_factors = max(1, int(top_k_factors))

        for factor_idx in range(n_factors):
            rng = np.random.RandomState(42 + factor_idx)
            t_hat = np.mean(t_mat, axis=0) + float(noise_std) * rng.randn(self.hd_dim).astype(np.float32)
            r_hat = np.mean(r_mat, axis=0) + float(noise_std) * rng.randn(self.hd_dim).astype(np.float32)
            t_hat /= (np.linalg.norm(t_hat) + 1e-9)
            r_hat /= (np.linalg.norm(r_hat) + 1e-9)

            converged = False
            iters = 0
            sims_t = np.zeros(len(candidate_tools), dtype=np.float32)
            sims_r = np.zeros(len(candidate_roles), dtype=np.float32)

            for k in range(max(1, int(max_iters))):
                iters = k + 1
                q_t = self.unbind_symbols(s_residual, r_hat)
                sims_t = np.dot(t_mat, q_t)
                exp_t = np.exp(beta * (sims_t - np.max(sims_t)))
                w_t = exp_t / (np.sum(exp_t) + 1e-9)
                t_next = np.dot(w_t, t_mat)
                t_next /= (np.linalg.norm(t_next) + 1e-9)

                q_r = self.unbind_symbols(s_residual, t_next)
                sims_r = np.dot(r_mat, q_r)
                exp_r = np.exp(beta * (sims_r - np.max(sims_r)))
                w_r = exp_r / (np.sum(exp_r) + 1e-9)
                r_next = np.dot(w_r, r_mat)
                r_next /= (np.linalg.norm(r_next) + 1e-9)

                if float(np.dot(t_next, t_hat)) > 0.9995 and float(np.dot(r_next, r_hat)) > 0.9995:
                    t_hat, r_hat = t_next, r_next
                    converged = True
                    break
                t_hat, r_hat = t_next, r_next

            best_t_idx = int(np.argmax(sims_t))
            best_r_idx = int(np.argmax(sims_r))
            decoded_factors.append({
                "tool": candidate_tools[best_t_idx],
                "role": candidate_roles[best_r_idx],
                "tool_resonance": round(float(sims_t[best_t_idx]), 4),
                "role_resonance": round(float(sims_r[best_r_idx]), 4),
                "iterations": iters,
                "converged": converged,
            })

            # Explain away converged bound factor pair from composite residual
            if factor_idx + 1 < n_factors:
                bound_est = self.bind_symbols(t_mat[best_t_idx], r_mat[best_r_idx])
                proj = float(np.dot(s_residual, bound_est))
                s_residual = s_residual - proj * bound_est
                res_norm = float(np.linalg.norm(s_residual))
                if res_norm > 1e-9:
                    s_residual = s_residual / res_norm

        primary = decoded_factors[0]
        return {
            "status": "resonator_factorized",
            "best_tool": primary["tool"],
            "best_role": primary["role"],
            "tool_resonance": primary["tool_resonance"],
            "role_resonance": primary["role_resonance"],
            "iterations_used": primary["iterations"],
            "converged": primary["converged"],
            "decoded_factors": decoded_factors,
            "latency_us": round((time.perf_counter() - t0) * 1e6, 2),
        }

