"""
Aetheris Multi-Compartment Dendritic Tree Micro-Brain.
Inspired by MMDEND and biological dendritic computing (2025-2026).
Replaces flat point-neurons with branched dendritic arbors that compute non-linear sub-operations
(branch-specific thresholding and logical gating) before somatic integration.

No headline performance claim: at the default configuration the branch weights cost
MORE parameters than an equivalent dense MLP (~6.6x — see the module's own
parameter_ratio_vs_dense_mlp result field), and per-call latency is clock-measured
and reported as latency_ms rather than asserted here.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

class DendriticCompartment:
    """A non-linear branch compartment computing local sub-threshold receptive field logic."""

    def __init__(self, in_features: int, branch_id: int):
        self.in_features = in_features
        self.branch_id = branch_id
        rng = np.random.RandomState(42 + branch_id)
        # Synaptic weights for this specific dendritic branch
        self.W_syn = rng.randn(in_features).astype(np.float32) / math.sqrt(in_features)
        self.threshold = 0.2

    def forward_branch(self, x: np.ndarray) -> float:
        """
        Local dendritic branch computation:
        Computes non-linear branch activation: v_branch = ReLU(W_syn @ x - theta)
        """
        raw_stimulus = float(np.dot(self.W_syn, x))
        # Non-linear sub-threshold rectification
        return max(0.0, raw_stimulus - self.threshold)

class MultiCompartmentDendriticNeuron:
    """A biological neuron featuring multiple non-linear dendritic branches converging onto a soma."""

    def __init__(self, in_features: int, num_branches: int = 4, neuron_id: int = 0):
        self.in_features = in_features
        self.num_branches = num_branches
        self.branches = [
            DendriticCompartment(in_features, branch_id=neuron_id * 10 + b)
            for b in range(num_branches)
        ]
        rng = np.random.RandomState(1337 + neuron_id)
        # Somatic integration weights across branches
        self.W_soma = rng.randn(num_branches).astype(np.float32) / math.sqrt(num_branches)
        self.soma_bias = 0.1

    def forward_soma(self, x: np.ndarray) -> float:
        """
        Somatic integration:
        y = tanh(sum_b (W_soma_b * v_branch_b) + b_soma)
        """
        branch_signals = np.array([b.forward_branch(x) for b in self.branches], dtype=np.float32)
        somatic_input = float(np.dot(self.W_soma, branch_signals)) + self.soma_bias
        return math.tanh(somatic_input)

class DendriticTreeMicroBrain:
    """A full layer of multi-compartment dendritic neurons."""

    _instance: Optional[DendriticTreeMicroBrain] = None

    def __init__(self, in_features: int = 32, out_features: int = 8, num_branches_per_neuron: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches_per_neuron
        self.neurons = [
            MultiCompartmentDendriticNeuron(in_features, num_branches=num_branches_per_neuron, neuron_id=i)
            for i in range(out_features)
        ]

    @classmethod
    def get_instance(cls) -> DendriticTreeMicroBrain:
        if cls._instance is None:
            cls._instance = DendriticTreeMicroBrain()
        return cls._instance

    def forward(self, input_vector: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Runs dendritic tree forward pass with sub-millisecond latency."""
        t0 = time.perf_counter()
        x = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(x) < self.in_features:
            x = np.pad(x, (0, self.in_features - len(x)))
        else:
            x = x[:self.in_features]

        outputs = [n.forward_soma(x) for n in self.neurons]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Honest parameter accounting: dendritic (synapses + somatic) vs a
        # dense single-hidden-layer MLP with the same branch-wide hidden layer.
        dendritic_params = self.out_features * (self.num_branches * self.in_features + self.num_branches)
        dense_mlp_params = self.in_features * self.num_branches + self.num_branches * self.out_features
        return {
            "status": "dendritic_evaluated",
            "out_dim": self.out_features,
            "output_vector": outputs,
            "dendritic_branches_evaluated": self.out_features * self.num_branches,
            "dendritic_parameters_count": dendritic_params,
            "dense_mlp_parameters_count": dense_mlp_params,
            "parameter_ratio_vs_dense_mlp": round(dendritic_params / max(1, dense_mlp_params), 3),
            "latency_ms": round(latency_ms, 4),
        }

    def adapt_dendritic_plateau(
        self,
        input_vector: Union[List[float], np.ndarray],
        target_vector: Union[List[float], np.ndarray],
        reward: float = 1.0,
        lr: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Biological Three-Factor Dendritic Plateau Plasticity:
        Modulates local branch synaptic weights (W_syn), branch gating thresholds,
        and somatic integration weights (W_soma, soma_bias) using somatic error
        feedback without global backpropagation.
        """
        t0 = time.perf_counter()
        x = np.asarray(input_vector, dtype=np.float32).flatten()
        if len(x) < self.in_features:
            x = np.pad(x, (0, self.in_features - len(x)))
        else:
            x = x[: self.in_features]

        target = np.asarray(target_vector, dtype=np.float32).flatten()
        if len(target) < self.out_features:
            target = np.pad(target, (0, self.out_features - len(target)))
        else:
            target = target[: self.out_features]

        y_pre = np.array([n.forward_soma(x) for n in self.neurons], dtype=np.float32)
        pre_mse = float(np.mean((y_pre - target) ** 2))

        active_branches = 0
        eff_lr = float(lr) * float(reward)
        for j, neuron in enumerate(self.neurons):
            y_j = float(y_pre[j])
            tanh_deriv = max(1e-3, 1.0 - y_j * y_j)
            delta_soma = (y_j - float(target[j])) * tanh_deriv

            branch_signals = np.array([b.forward_branch(x) for b in neuron.branches], dtype=np.float32)
            # Local three-factor branch synaptic update only for depolarized (active) branches
            w_soma_snapshot = neuron.W_soma.copy()
            neuron_active = 0
            for b_idx, branch in enumerate(neuron.branches):
                if branch_signals[b_idx] > 0.0:
                    active_branches += 1
                    neuron_active += 1
                    delta_b = delta_soma * float(w_soma_snapshot[b_idx])
                    branch.W_syn = (branch.W_syn - eff_lr * delta_b * x).astype(np.float32)
                    branch.threshold = float(branch.threshold + eff_lr * delta_b)

            # Homeostatic Dead-Dendrite Recruitment when all branches were quiescent
            if neuron_active == 0 and abs(delta_soma) > 1e-6 and neuron.branches:
                stimuli = [float(np.dot(b.W_syn, x)) - b.threshold for b in neuron.branches]
                rescue_idx = int(np.argmax(stimuli))
                branch = neuron.branches[rescue_idx]
                branch.threshold = float(np.dot(branch.W_syn, x)) - 0.05
                if abs(float(neuron.W_soma[rescue_idx])) < 0.1:
                    neuron.W_soma[rescue_idx] = np.float32(-0.25 * (1.0 if delta_soma >= 0 else -1.0))
                delta_b = delta_soma * float(neuron.W_soma[rescue_idx])
                branch.W_syn = (branch.W_syn - eff_lr * delta_b * x).astype(np.float32)
                active_branches += 1
                branch_signals[rescue_idx] = branch.forward_branch(x)

            neuron.W_soma = (neuron.W_soma - eff_lr * delta_soma * branch_signals).astype(np.float32)
            neuron.soma_bias = float(neuron.soma_bias - eff_lr * delta_soma)

        y_post = np.array([n.forward_soma(x) for n in self.neurons], dtype=np.float32)
        post_mse = float(np.mean((y_post - target) ** 2))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "status": "dendritic_plateau_adapted",
            "pre_mse": round(pre_mse, 6),
            "post_mse": round(post_mse, 6),
            "mse_improved": bool(post_mse <= pre_mse),
            "active_branches_updated": active_branches,
            "latency_ms": round(latency_ms, 4),
        }

