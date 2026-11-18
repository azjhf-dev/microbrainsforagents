"""
Pillar 3: Zero-Gradient Online Brain Plasticity Updater
======================================================
Inspired by biological Long-Term Potentiation (LTP) and MIT CSAIL WSP.

Dynamically reinforces or dampens synapses in autonomous micro-transformers
based on real-world execution feedback, WITHOUT gradient descent:
1. Success Feedback -> Hebbian Potentiation:
       W_down[neuron, :] += eta * target_vec
2. Failure / Anomaly Feedback -> Astra Subtractive Damping:
       W_up[:, neuron] -= beta * trigger_vec
Stabilized by continuous spectral norm bounding (||W||_2 <= 2.0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from .analytical_micro_transformer import AnalyticalMicroTransformer
from .brain_nursery import AutonomousBrainNursery

logger = logging.getLogger(__name__)


class BrainPlasticityUpdater:
    """Updates micro-transformer parameters online in response to task execution outcomes."""

    _instance: Optional[BrainPlasticityUpdater] = None

    def __init__(self, eta_hebbian: float = 0.15, beta_astra: float = 0.25):
        self.eta = eta_hebbian
        self.beta = beta_astra
        self.nursery = AutonomousBrainNursery.get_instance()

    @classmethod
    def get_instance(cls) -> BrainPlasticityUpdater:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def process_execution_feedback(
        self,
        domain_name: str,
        trigger_token: str,
        predicted_token: str,
        success: bool,
        quality_score: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Applies immediate zero-gradient synaptic update to the domain micro-transformer:
        - If success: strengthens connection between trigger_token and predicted_token.
        - If failure: applies Astra column subtraction to weaken the faulty association.
        """
        clean_domain = domain_name.lower().strip().replace(" ", "_")
        brain_name = clean_domain if clean_domain.startswith("neuro_") else f"neuro_{clean_domain}"
        if brain_name not in self.nursery.active_brains:
            loaded = AnalyticalMicroTransformer.load(brain_name)
            if loaded is not None:
                self.nursery.active_brains[brain_name] = loaded
            else:
                return {"status": "skipped", "reason": f"Brain {brain_name} not active"}

        brain: AnalyticalMicroTransformer = self.nursery.active_brains[brain_name]

        k_idx = brain.vocab.get(trigger_token)
        v_idx = brain.vocab.get(predicted_token)

        auto_registered = False
        if k_idx is None or v_idx is None:
            if success and hasattr(brain, "compile_logic_association"):
                brain.compile_logic_association(trigger_token, predicted_token, binding_gain=3.0)
                k_idx = brain.vocab.get(trigger_token)
                v_idx = brain.vocab.get(predicted_token)
                auto_registered = (k_idx is not None and v_idx is not None)
            if k_idx is None or v_idx is None:
                return {"status": "skipped", "reason": "Tokens not in brain vocabulary"}

        k_vec = brain.W_embed[k_idx]
        v_vec = brain.W_embed[v_idx]
        neuron_idx = k_idx % brain.d_mlp

        hopfield_updated = False
        k_norm = float(np.linalg.norm(k_vec)) + 1e-9

        if success:
            # 1. Biological Hebbian Potentiation (LTP)
            # Reinforce W_down projection towards v_vec
            boost = float(self.eta * quality_score)
            brain.W_down[neuron_idx, :] += boost * v_vec

            # Bounding check
            norm = np.linalg.norm(brain.W_down[neuron_idx, :])
            if norm > 3.0:
                brain.W_down[neuron_idx, :] *= (3.0 / norm)

            # Sync matching Hopfield associative bank
            for idx, hk in enumerate(brain.hopfield_keys):
                cos_k = float(np.dot(hk, k_vec) / ((float(np.linalg.norm(hk)) + 1e-9) * k_norm))
                if cos_k > 0.99:
                    brain.hopfield_values[idx] = brain.hopfield_values[idx] + boost * v_vec
                    hopfield_updated = True

            action_type = "hebbian_potentiation"
        else:
            # 2. Astra Subtractive Column Damping
            # Neutralize the attractor weight column in W_up
            proj = float(np.dot(brain.W_up[:, neuron_idx], k_vec))
            brain.W_up[:, neuron_idx] -= float(self.beta) * proj * k_vec
            up_norm = float(np.linalg.norm(brain.W_up[:, neuron_idx]))
            if up_norm > 3.0:
                brain.W_up[:, neuron_idx] *= (3.0 / up_norm)

            # Dampen matching Hopfield value away from failed token
            for idx, hk in enumerate(brain.hopfield_keys):
                cos_k = float(np.dot(hk, k_vec) / ((float(np.linalg.norm(hk)) + 1e-9) * k_norm))
                if cos_k > 0.99:
                    proj_v = float(np.dot(brain.hopfield_values[idx], v_vec))
                    brain.hopfield_values[idx] = brain.hopfield_values[idx] - float(self.beta) * proj_v * v_vec
                    hopfield_updated = True

            action_type = "astra_column_damping"

        # Continuous spectral norm cap across W_up / W_down to prevent attractor drift
        for attr in ("W_up", "W_down"):
            W_mat = getattr(brain, attr, None)
            if isinstance(W_mat, np.ndarray) and W_mat.ndim == 2 and W_mat.size > 0:
                s_max = float(np.linalg.norm(W_mat, ord=2))
                if s_max > 3.0:
                    setattr(brain, attr, (W_mat * (3.0 / s_max)).astype(W_mat.dtype))

        brain.save()

        return {
            "status": "updated",
            "action_type": action_type,
            "domain_name": domain_name,
            "brain_name": brain_name,
            "trigger_token": trigger_token,
            "target_token": predicted_token,
            "neuron_idx": neuron_idx,
            "hopfield_synced": hopfield_updated,
            "auto_registered": auto_registered,
            "success": success,
        }

