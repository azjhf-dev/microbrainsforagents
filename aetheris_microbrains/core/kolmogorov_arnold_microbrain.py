"""
Aetheris Kolmogorov-Arnold Network (KAN 2.0) Micro-Brain Engine.
Inspired by ICLR 2025-2026 KANs and Symbolic-KANs.
Places learnable 1D basis functions (Chebyshev orthogonal polynomials) on edges rather than nodes.
Enables direct symbolic formula decompilation: extracting human-readable algebraic rules from neural weights.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

class KolmogorovArnoldMicroBrain:
    """A symbolic KAN layer that evaluates Chebyshev polynomial basis functions on edges."""

    _instance: Optional[KolmogorovArnoldMicroBrain] = None

    def __init__(self, in_features: int = 4, out_features: int = 2, degree: int = 3):
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree  # Chebyshev polynomial degree (T_0, T_1, ..., T_degree)
        self.trained: bool = False
        self.training_mse: Optional[float] = None
        self.grid_center = np.zeros(in_features, dtype=np.float32)
        self.grid_scale = np.ones(in_features, dtype=np.float32)
        self.grid_adapted: bool = False

        # Coefficients tensor: shape (out_features, in_features, degree + 1)
        rng = np.random.RandomState(42)
        self.coeffs = rng.randn(out_features, in_features, degree + 1).astype(np.float32) * 0.1

    @classmethod
    def get_instance(cls) -> KolmogorovArnoldMicroBrain:
        if cls._instance is None:
            cls._instance = KolmogorovArnoldMicroBrain()
        return cls._instance

    def _chebyshev_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Computes Chebyshev polynomials of the first kind:
        T_0(x) = 1
        T_1(x) = x
        T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
        Assumes x is bounded in [-1, 1].
        """
        x_clipped = np.clip(x, -1.0, 1.0)
        if self.degree <= 0:
            return np.stack([np.ones_like(x_clipped)], axis=-1)
        bases = [np.ones_like(x_clipped), x_clipped]
        for n in range(2, self.degree + 1):
            t_n = 2.0 * x_clipped * bases[-1] - bases[-2]
            bases.append(t_n)
        return np.stack(bases, axis=-1)  # Shape: (..., in_features, degree + 1)

    def _normalize_inputs(self, x: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, bool]:
        x_arr = np.asarray(x, dtype=np.float32)
        is_1d = x_arr.ndim <= 1
        if is_1d:
            x_mat = x_arr.reshape(1, -1)
        else:
            x_mat = x_arr.reshape(-1, x_arr.shape[-1])
        if x_mat.shape[1] < self.in_features:
            x_mat = np.pad(x_mat, ((0, 0), (0, self.in_features - x_mat.shape[1])))
        else:
            x_mat = x_mat[:, : self.in_features]
        return x_mat, is_1d

    def _map_to_domain(self, x_mat: np.ndarray) -> np.ndarray:
        """Maps normalized input matrix into [-1, 1] using adaptive domain grid if calibrated."""
        if self.grid_adapted:
            scaled = (x_mat - self.grid_center[None, :]) / np.maximum(self.grid_scale[None, :], 1e-6)
            return np.tanh(scaled)
        return np.tanh(x_mat)

    def adapt_grid(
        self,
        x_samples: Union[List[List[float]], np.ndarray],
        margin: float = 0.15,
    ) -> Dict[str, Any]:
        """
        AdaptKAN (2025-2026) Dynamic Domain Grid Calibration:
        Adapts per-feature center and half-range scale from empirical sample distributions
        so arbitrary-scale inputs remain in the responsive interior of [-1, 1].
        """
        X, _ = self._normalize_inputs(x_samples)
        if X.shape[0] == 0:
            return {"status": "skipped_empty_samples", "grid_adapted": self.grid_adapted}

        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        center = 0.5 * (x_min + x_max)
        half_range = np.maximum((x_max - x_min) * 0.5 * (1.0 + float(margin)), 1e-3)

        self.grid_center = center.astype(np.float32)
        self.grid_scale = half_range.astype(np.float32)
        self.grid_adapted = True

        return {
            "status": "kan_grid_adapted",
            "grid_adapted": True,
            "samples": int(X.shape[0]),
            "grid_center": [round(float(v), 4) for v in self.grid_center],
            "grid_scale": [round(float(v), 4) for v in self.grid_scale],
        }

    def forward(self, x: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Vectorized KAN Forward Pass supporting 1D vectors or 2D batches:
        y_j = sum_i phi_{i,j}(x_i) = sum_i sum_k c_{j,i,k} * T_k(map(x_i))
        """
        x_mat, is_1d = self._normalize_inputs(x)
        x_norm = self._map_to_domain(x_mat)
        bases = self._chebyshev_basis(x_norm)  # (batch, in_features, degree + 1)

        # coeffs: (out_features, in_features, degree + 1)
        y_batch = np.einsum("oik,bik->bo", self.coeffs, bases).astype(np.float32)
        return y_batch[0] if is_1d else y_batch

    def fit_symbolic(
        self,
        x_samples: Union[List[List[float]], np.ndarray],
        y_targets: Union[List[List[float]], np.ndarray],
        l2_reg: float = 1e-3,
        prune_threshold: float = 1e-3,
        adapt_domain_grid: bool = False,
    ) -> Dict[str, Any]:
        """
        Fits the Chebyshev edge polynomial coefficients via closed-form Vandermonde
        ridge regression in O(N * D^2 + D^3) CPU time, with optional dynamic domain
        grid adaptation and sparse pruning to yield clean symbolic formulas.
        """
        X, _ = self._normalize_inputs(x_samples)
        n_samples = X.shape[0]
        Y_raw = np.asarray(y_targets, dtype=np.float32).reshape(n_samples, -1)
        if Y_raw.shape[1] < self.out_features:
            Y = np.pad(Y_raw, ((0, 0), (0, self.out_features - Y_raw.shape[1])))
        else:
            Y = Y_raw[:, : self.out_features]

        if n_samples == 0:
            return {"status": "skipped_empty_samples", "trained": self.trained}

        if adapt_domain_grid:
            self.adapt_grid(X)

        y_pre = self.forward(X)
        pre_mse = float(np.mean((y_pre - Y) ** 2))

        bases = self._chebyshev_basis(self._map_to_domain(X))  # (N, in_features, degree + 1)
        n_basis_total = self.in_features * (self.degree + 1)
        phi = bases.reshape(n_samples, n_basis_total).astype(np.float64)
        y_64 = Y.astype(np.float64)

        gram = phi.T @ phi + float(l2_reg) * np.eye(n_basis_total, dtype=np.float64)
        rhs = phi.T @ y_64
        try:
            w_flat = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            w_flat, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)

        new_coeffs = w_flat.T.reshape(self.out_features, self.in_features, self.degree + 1).astype(np.float32)
        if prune_threshold > 0.0:
            new_coeffs[np.abs(new_coeffs) < float(prune_threshold)] = 0.0

        self.coeffs = new_coeffs
        self.trained = True

        y_post = self.forward(X)
        post_mse = float(np.mean((y_post - Y) ** 2))
        self.training_mse = round(post_mse, 6)

        return {
            "status": "symbolic_kan_fitted",
            "trained": True,
            "samples": int(n_samples),
            "pre_mse": round(pre_mse, 6),
            "post_mse": round(post_mse, 6),
            "mse_reduced": bool(post_mse <= pre_mse),
            "active_terms": int(np.count_nonzero(self.coeffs)),
            "total_terms": int(self.coeffs.size),
        }

    def extract_symbolic_formula(self, top_k_terms: Optional[int] = None) -> Dict[str, Any]:
        """
        Decompiles the CURRENT edge polynomial coefficients into human-readable
        algebraic formulas. Before fit_symbolic() is called, status is
        "extracted_untrained_basis" (trained=False); once fitted, status becomes
        "extracted_fitted_basis" (trained=True) with measured training_mse.
        """
        formulas = {}
        for j in range(self.out_features):
            raw_terms: List[Tuple[float, str]] = []
            for i in range(self.in_features):
                c = self.coeffs[j, i]
                for k in range(self.degree + 1):
                    val = float(c[k])
                    if abs(val) <= 0.01:
                        continue
                    if k == 0:
                        expr = f"{val:+.3f}"
                    elif k == 1:
                        expr = f"{val:+.3f}*x_{i}"
                    elif k == 2:
                        expr = f"{val:+.3f}*(2*x_{i}**2 - 1)"
                    elif k == 3:
                        expr = f"{val:+.3f}*(4*x_{i}**3 - 3*x_{i})"
                    else:
                        expr = f"{val:+.3f}*T_{k}(x_{i})"
                    raw_terms.append((abs(val), expr))

            if top_k_terms is not None and top_k_terms > 0 and len(raw_terms) > top_k_terms:
                raw_terms.sort(key=lambda pair: pair[0], reverse=True)
                raw_terms = raw_terms[:top_k_terms]

            formula_str = " + ".join(expr for _, expr in raw_terms) if raw_terms else "0.0"
            formulas[f"y_{j}"] = formula_str

        status_label = "extracted_fitted_basis" if self.trained else "extracted_untrained_basis"
        return {
            "status": status_label,
            "trained": bool(self.trained),
            "training_mse": self.training_mse,
            "model_type": "Kolmogorov-Arnold-Chebyshev",
            "symbolic_formulas": formulas,
            "degrees": self.degree,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }

