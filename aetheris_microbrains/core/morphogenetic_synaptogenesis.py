"""
Morphogenetic Synaptogenesis & White-Matter Routing Engine (BraiNCA).
Ported and evolved from 2026 SOTA (BraiNCA: Pio-Lopez, Hartl, & Levin; Developmental Graph Automata).

Enables continuous organic synaptogenesis between micro-brains:
1. Dynamic Axon Sprouting: When task complexity / prediction error VFE exceeds a threshold (e.g. > 0.40),
   the micro-brain sprouts long-range axonal connections to neighboring domain brains.
2. Synaptic Pruning: Inactive connections decay with an activity-dependent penalty.
3. White-Matter Hyper-Graph: Transforms static weight files into a dynamically reconfigurable
   neural connectome that expands network capacity organically where cognitive demand spikes.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

log = logging.getLogger("aetheris.morphogenetic_synaptogenesis")


@dataclass
class MorphogeneticAxon:
    source_brain: str
    target_brain: str
    weight: float
    conductivity: float
    last_activation_ts: float
    sprout_count: int = 1

    @property
    def conductance(self) -> float:
        return self.conductivity


@dataclass
class SynaptogenesisReport:
    active_brains_count: int
    total_axons_count: int
    sprouted_new_axons: int
    pruned_dead_axons: int
    network_diameter: int
    mean_conductivity: float


class MorphogeneticSynaptogenesisEngine:
    """
    Organic neural wiring engine dynamically growing and pruning connections across micro-brains.
    """

    _instance: Optional["MorphogeneticSynaptogenesisEngine"] = None

    def __init__(self, growth_threshold_vfe: float = 0.38, decay_rate: float = 0.99):
        self.growth_threshold_vfe = float(growth_threshold_vfe)
        self.decay_rate = float(decay_rate)
        self.axons: Dict[Tuple[str, str], MorphogeneticAxon] = {}
        self.brain_nodes: Set[str] = set()

    @classmethod
    def get_instance(cls) -> "MorphogeneticSynaptogenesisEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_brain_node(self, brain_name: str):
        self.brain_nodes.add(brain_name)

    def evaluate_demand_and_sprout(
        self,
        active_brain: str,
        current_vfe_error: float,
        candidate_targets: List[str]
    ) -> int:
        """
        If variational free energy error exceeds threshold, sprouts new axonal connections
        to candidate target micro-brains to share compute load.
        """
        self.register_brain_node(active_brain)
        sprouted = 0

        if current_vfe_error >= self.growth_threshold_vfe:
            # Need additional capacity: sprout connections
            import time
            now = time.time()
            for tgt in candidate_targets:
                if tgt == active_brain:
                    continue
                self.register_brain_node(tgt)
                key = (active_brain, tgt)

                if key not in self.axons:
                    # New axon sprouting
                    self.axons[key] = MorphogeneticAxon(
                        source_brain=active_brain,
                        target_brain=tgt,
                        weight=0.15,
                        conductivity=0.50,
                        last_activation_ts=now,
                    )
                    sprouted += 1
                else:
                    # Strengthen existing axon (Activity-dependent plasticity)
                    self.axons[key].conductivity = min(1.0, self.axons[key].conductivity + 0.10)
                    self.axons[key].last_activation_ts = now

        return sprouted

    def _compute_network_diameter(self) -> int:
        """Computes exact directed shortest-path BFS diameter of active axonal graph."""
        if not self.axons:
            return 0
        adj: Dict[str, List[str]] = {}
        nodes: Set[str] = set()
        for src, tgt in self.axons.keys():
            adj.setdefault(src, []).append(tgt)
            nodes.add(src)
            nodes.add(tgt)
        max_dist = 0
        for start in nodes:
            dist = {start: 0}
            queue = [start]
            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                for nxt in adj.get(curr, ()):
                    if nxt not in dist:
                        d = dist[curr] + 1
                        dist[nxt] = d
                        if d > max_dist:
                            max_dist = d
                        queue.append(nxt)
        return max_dist

    def prune_and_consolidate(self, max_idle_age_s: float = 3600.0) -> SynaptogenesisReport:
        """Decays unused axons and prunes connections that fall below minimal conductivity."""
        import time
        now = time.time()
        pruned = 0
        active_keys = list(self.axons.keys())

        for key in active_keys:
            axon = self.axons[key]
            # Exponential conductivity decay
            axon.conductivity *= self.decay_rate
            idle_time = now - axon.last_activation_ts

            if axon.conductivity < 0.05 or idle_time > max_idle_age_s:
                del self.axons[key]
                pruned += 1

        total_axons = len(self.axons)
        mean_cond = float(np.mean([a.conductivity for a in self.axons.values()])) if total_axons > 0 else 0.0

        return SynaptogenesisReport(
            active_brains_count=len(self.brain_nodes),
            total_axons_count=total_axons,
            sprouted_new_axons=0,
            pruned_dead_axons=pruned,
            network_diameter=self._compute_network_diameter(),
            mean_conductivity=round(mean_cond, 4),
        )

    def route_signal_through_connectome(self, origin_brain: str, signal_vector: np.ndarray) -> np.ndarray:
        """Propagates signal vector through connected sprouted axons with target lateral transformation."""
        sig = np.asarray(signal_vector, dtype=np.float64).flatten()
        out_signal = sig.copy()
        if sig.size == 0:
            return out_signal

        active_brains: Dict[str, Any] = {}
        try:
            from core.brain_nursery import AutonomousBrainNursery
            if AutonomousBrainNursery._instance is not None:
                active_brains = AutonomousBrainNursery._instance.active_brains
        except Exception:
            active_brains = {}

        for (src, tgt), axon in self.axons.items():
            if src == origin_brain:
                target_brain = active_brains.get(tgt) or active_brains.get(f"neuro_{tgt}")
                w_mlp = getattr(target_brain, "W_mlp", None) if target_brain is not None else None
                if isinstance(w_mlp, np.ndarray) and w_mlp.ndim == 2:
                    d_in = w_mlp.shape[0]
                    sig_aligned = sig[:d_in] if sig.size >= d_in else np.pad(sig, (0, d_in - sig.size))
                    proj = np.dot(sig_aligned, w_mlp.astype(np.float64))
                    lateral_vec = proj[:sig.size] if proj.size >= sig.size else np.pad(proj, (0, sig.size - proj.size))
                else:
                    shift = (sum(ord(c) for c in tgt) % max(1, sig.size - 1)) + 1
                    lateral_vec = np.roll(sig, shift)
                out_signal += lateral_vec * float(axon.weight) * float(axon.conductivity)
        norm = np.linalg.norm(out_signal)
        if norm > 1e-12:
            out_signal = out_signal / norm
        return out_signal

    def clear(self):
        """Clears all axons and brains."""
        self.axons.clear()
        self.brain_nodes.clear()

    def register_interaction(
        self,
        source_brain: str,
        target_brain: str,
        prediction_error: float,
        success: bool = True
    ) -> Optional[MorphogeneticAxon]:
        """Registers cross-brain interaction, sprouting or strengthening an axon."""
        import time
        now = time.time()
        key = (source_brain, target_brain)
        self.brain_nodes.add(source_brain)
        self.brain_nodes.add(target_brain)

        if key in self.axons:
            axon = self.axons[key]
            axon.last_activation_ts = now
            if prediction_error >= self.growth_threshold_vfe:
                axon.conductivity = min(2.0, axon.conductivity + float(prediction_error))
                axon.weight = min(1.0, axon.weight + 0.1)
                axon.sprout_count += 1
            return axon

        if prediction_error >= self.growth_threshold_vfe:
            axon = MorphogeneticAxon(
                source_brain=source_brain,
                target_brain=target_brain,
                weight=min(1.0, 0.5 + float(prediction_error) * 0.5),
                conductivity=min(2.0, float(prediction_error)),
                last_activation_ts=now,
                sprout_count=1
            )
            self.axons[key] = axon
            return axon

        return None

    def step_synaptic_lifecycle(self, decay_rate: float = 0.05) -> Dict[str, Any]:
        """Applies decay and returns summary."""
        for axon in self.axons.values():
            axon.conductivity *= (1.0 - decay_rate)
        total = len(self.axons)
        mean_cond = float(np.mean([a.conductivity for a in self.axons.values()])) if total > 0 else 0.0
        return {
            "total_axons": total,
            "active_axons": sum(1 for a in self.axons.values() if a.conductivity > 0.1),
            "mean_conductance": round(mean_cond, 4)
        }

    def get_routing_table(self) -> Dict[str, float]:
        return {f"{k[0]}->{k[1]}": round(a.conductivity, 4) for k, a in self.axons.items()}

    def prune_inactive_axons(self, min_conductance: float = 0.1) -> Dict[str, int]:
        dead_keys = [k for k, a in self.axons.items() if a.conductivity < min_conductance]
        for k in dead_keys:
            del self.axons[k]
        return {"pruned_axons": len(dead_keys)}


# Alias for backward compatibility and clean test suite imports
MorphogeneticSynaptogenesis = MorphogeneticSynaptogenesisEngine


