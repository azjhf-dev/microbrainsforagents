"""
Aetheris Liquid State Machine & Reservoir Computing Engine.
Inspired by NeoAura's echo_core and Edge Neuromorphic Reservoir Computing (2025-2026).
Tracks multi-step temporal workflows and telemetry dynamics across agent tool executions
using an in-memory sparse recurrent liquid without requiring backpropagation through time.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

class LiquidStateReservoir:
    """A recurrent sparse reservoir that maintains temporal memory over sequences of actions."""

    _instance: Optional[LiquidStateReservoir] = None

    def __init__(self, reservoir_dim: int = 128, input_dim: int = 64, spectral_radius: float = 0.92, leak_rate: float = 0.3):
        self.reservoir_dim = reservoir_dim
        self.input_dim = input_dim
        self.spectral_radius = spectral_radius
        self.leak_rate = leak_rate

        # Initialize internal reservoir state
        self.state = np.zeros(reservoir_dim, dtype=np.float32)

        # Build sparse deterministic recurrent matrix
        rng = np.random.RandomState(1337)
        raw_W = rng.randn(reservoir_dim, reservoir_dim).astype(np.float32)
        # 80% sparsity
        mask = rng.rand(reservoir_dim, reservoir_dim) < 0.2
        raw_W = raw_W * mask

        # Scale to designated spectral radius
        eigenvalues = np.linalg.eigvals(raw_W)
        max_eig = max(abs(e) for e in eigenvalues) if len(eigenvalues) > 0 else 1.0
        if max_eig > 1e-9:
            self.W_res = raw_W * (spectral_radius / max_eig)
        else:
            self.W_res = raw_W

        self.W_in = rng.randn(reservoir_dim, input_dim).astype(np.float32) * 0.5
        self.W_gate = rng.randn(reservoir_dim, input_dim).astype(np.float32) * 0.3
        # Multi-timescale inverse time constants tau_inv in [0.05, 5.0] for CfC dynamics
        self.tau_inv = np.linspace(0.05, 5.0, reservoir_dim, dtype=np.float32)
        self.W_readout: Optional[np.ndarray] = None
        self.readout_trained: bool = False
        self.step_counter = 0
        self._last_step_ts: Optional[float] = None

    @classmethod
    def get_instance(cls) -> LiquidStateReservoir:
        if cls._instance is None:
            cls._instance = LiquidStateReservoir()
        return cls._instance

    def step(self, input_vector: Union[List[float], np.ndarray], event_tag: str = "agent_step") -> Dict[str, Any]:
        """
        Leaky Integrate-and-Fire Reservoir Update:
        s_t = (1 - alpha) * s_{t-1} + alpha * tanh(W_res @ s_{t-1} + W_in @ u_t)
        """
        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.input_dim:
            u = np.pad(u, (0, self.input_dim - len(u)))
        else:
            u = u[:self.input_dim]

        pre_activation = np.dot(self.W_res, self.state) + np.dot(self.W_in, u)
        non_linear = np.tanh(pre_activation)
        
        # Leaky integration
        self.state = (1.0 - self.leak_rate) * self.state + self.leak_rate * non_linear
        self.step_counter += 1

        state_energy = float(np.linalg.norm(self.state))
        top_activations = [round(float(x), 4) for x in self.state[:6]]

        res = {
            "status": "stepped",
            "step_count": self.step_counter,
            "event_tag": event_tag,
            "reservoir_energy": round(state_energy, 4),
            "top_state_readout": top_activations,
            "full_state": self.state.tolist(),
        }
        if self.readout_trained and self.W_readout is not None:
            res["readout_prediction"] = np.dot(self.state, self.W_readout).tolist()
        return res

    def fit_readout(
        self,
        state_history: Union[List[List[float]], np.ndarray],
        targets: Union[List[List[float]], np.ndarray],
        l2_reg: float = 1e-2,
    ) -> Dict[str, Any]:
        """
        2025-2026 LFM / Reservoir Readout Adaptation:
        Learns linear readout weights W_readout via closed-form Tikhonov ridge regression
        over collected reservoir trajectories (without BPTT).
        """
        S = np.asarray(state_history, dtype=np.float32)
        if S.ndim == 1:
            S = S[None, :]
        n = S.shape[0]
        if n == 0:
            return {"status": "skipped_empty", "readout_trained": self.readout_trained}
        if S.shape[1] < self.reservoir_dim:
            S = np.pad(S, ((0, 0), (0, self.reservoir_dim - S.shape[1])))
        else:
            S = S[:, : self.reservoir_dim]

        Y = np.asarray(targets, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(n, -1)
        else:
            Y = Y[:n]

        if self.W_readout is not None and self.W_readout.shape == (self.reservoir_dim, Y.shape[1]):
            pre_pred = np.dot(S, self.W_readout)
            pre_mse = float(np.mean((pre_pred - Y) ** 2))
        else:
            pre_mse = float(np.mean(Y ** 2))

        S64 = S.astype(np.float64)
        Y64 = Y.astype(np.float64)
        gram = S64.T @ S64 + float(l2_reg) * np.eye(self.reservoir_dim, dtype=np.float64)
        rhs = S64.T @ Y64
        try:
            W_opt = np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            W_opt, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
            W_opt = W_opt.astype(np.float32)

        self.W_readout = W_opt
        self.readout_trained = True
        post_pred = np.dot(S, self.W_readout)
        post_mse = float(np.mean((post_pred - Y) ** 2))

        return {
            "status": "readout_fitted",
            "readout_trained": True,
            "samples": int(n),
            "out_dim": int(Y.shape[1]),
            "pre_mse": round(pre_mse, 6),
            "post_mse": round(post_mse, 6),
            "mse_reduced": bool(post_mse <= pre_mse),
        }

    def step_cfc_continuous(
        self,
        input_vector: Union[List[float], np.ndarray],
        delta_t: Optional[float] = None,
        event_tag: str = "agent_step",
        force_gpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Closed-Form Continuous-Time (CfC) Reservoir Update on OpenCL GPU / CPU:
        h(t + dt) = sigmoid(-|f(x, h)| * tau_inv * dt) * h(t) + (1 - sigmoid(...)) * tanh(g(x, h))
        Natively adapts to irregular wall-clock delays between agent tool calls.
        """
        import time as _time

        now = _time.perf_counter()
        if delta_t is None:
            dt = (now - self._last_step_ts) if self._last_step_ts is not None else 0.1
        else:
            dt = float(delta_t)
        self._last_step_ts = now

        u = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(u) < self.input_dim:
            u = np.pad(u, (0, self.input_dim - len(u)))
        else:
            u = u[: self.input_dim]

        g_drive = np.dot(self.W_res, self.state) + np.dot(self.W_in, u)
        f_drive = np.dot(self.W_gate, u) + self.state

        from .opencl_microbrain_accelerator import get_opencl_accelerator
        acc = get_opencl_accelerator()
        self.state, backend = acc.cfc_step(self.state, f_drive, g_drive, self.tau_inv, dt, force_gpu=force_gpu)
        self.step_counter += 1

        state_energy = float(np.linalg.norm(self.state))
        top_activations = [round(float(x), 4) for x in self.state[:6]]
        res = {
            "status": "stepped_cfc",
            "backend": backend,
            "delta_t": round(float(dt), 4),
            "step_count": self.step_counter,
            "event_tag": event_tag,
            "reservoir_energy": round(state_energy, 4),
            "top_state_readout": top_activations,
            "full_state": self.state.tolist(),
        }
        if self.readout_trained and self.W_readout is not None:
            res["readout_prediction"] = np.dot(self.state, self.W_readout).tolist()
        return res

    def reset_state(self):
        """Flushes the reservoir liquid to ground state."""
        self.state = np.zeros(self.reservoir_dim, dtype=np.float32)
        self.step_counter = 0
        self._last_step_ts = None

