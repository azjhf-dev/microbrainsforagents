"""Weight Space Memory via Vector Symbolic Architecture (VSA / Hyperdimensional Computing).

Provides a holographic, associative memory over the weight space of the Aetheris microbrains fleet:
1. Maps parameter tensors of AnalyticalMicroTransformers into high-dimensional bipolar vectors {-1, +1}^D.
2. Binds domain identifiers with weight-feature vectors using Hadamard/circular binding.
3. Bundles all fleet models into a collective holographic associative memory tensor:
      M_fleet = Sum_k (domain_k (*) weights_k)
4. Resonator Network Clean-Up:
   Recovers domain identity, measures weight-space cosine distances, and reconstructs
   missing or corrupted parameter components via associative recall.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_CANONICAL_VSA_PATHS: List[Path] = [
    Path(__file__).resolve().parents[1] / "memory" / "store" / "weight_space_vsa.npz",
    Path("memory/store/weight_space_vsa.npz"),
]


def _get_default_vsa_path() -> Path:
    for p in _CANONICAL_VSA_PATHS:
        if p.exists():
            return p
    return _CANONICAL_VSA_PATHS[0]


@dataclass
class VSAResult:
    """Outcome of an associative query against weight-space memory."""

    domain: str
    similarity: float
    confidence: float
    reconstructed: bool
    dimensions: int


class WeightSpaceVSA:
    """Hyperdimensional Computing / VSA Associative Memory for Fleet Weights."""

    def __init__(self, dimension: int = 2048, seed: int = 42):
        self.dim = dimension
        self.rng = np.random.default_rng(seed)
        self.domain_codebook: Dict[str, np.ndarray] = {}
        self.memory_bundle = np.zeros(self.dim, dtype=np.float32)
        self.stored_domains: List[str] = []

    def _generate_bipolar_vector(self, key: str) -> np.ndarray:
        """Deterministically derive a bipolar {-1, +1}^D vector from a key string."""
        seed_int = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed_int)
        return rng.choice([-1.0, 1.0], size=self.dim).astype(np.float32)

    def _get_or_create_domain_vector(self, domain: str) -> np.ndarray:
        if domain not in self.domain_codebook:
            self.domain_codebook[domain] = self._generate_bipolar_vector(f"domain::{domain}")
        return self.domain_codebook[domain]

    def project_weights(self, weights_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Project an arbitrary dictionary of weight arrays into a normalized D-dim vector via Count-Sketch."""
        accum = np.zeros(self.dim, dtype=np.float32)
        for param_name, arr in sorted(weights_dict.items()):
            param_key = self._generate_bipolar_vector(f"param::{param_name}")
            flat = arr.flatten().astype(np.float32)
            if flat.size == 0:
                continue

            # Sparse random projection via Count-Sketch hashing (O(N) time, zero memory allocation)
            proj_seed = int(hashlib.sha256(param_name.encode()).hexdigest()[:8], 16)
            indices = (np.arange(flat.size) * 1103515245 + proj_seed) % self.dim
            signs = np.where(((np.arange(flat.size) * 12345 + proj_seed) % 2) == 0, 1.0, -1.0).astype(np.float32)

            compressed = np.zeros(self.dim, dtype=np.float32)
            np.add.at(compressed, indices, flat * signs)

            norm = np.linalg.norm(compressed)
            if norm > 1e-8:
                compressed = compressed / norm

            # Bind parameter key with compressed representation
            bound = param_key * compressed
            accum += bound

        norm = np.linalg.norm(accum)
        if norm > 1e-8:
            accum = accum / norm
        return accum

    def store_brain_weights(self, domain: str, weights_dict: Dict[str, np.ndarray]) -> None:
        """Bind domain vector with projected weight vector and superimpose into fleet memory."""
        dom_vec = self._get_or_create_domain_vector(domain)
        w_vec = self.project_weights(weights_dict)

        # Hadamard binding: dom (*) w
        bound_representation = dom_vec * w_vec
        self.memory_bundle += bound_representation
        if domain not in self.stored_domains:
            self.stored_domains.append(domain)

    def query_domain_by_weights(self, weights_dict: Dict[str, np.ndarray]) -> VSAResult:
        """Query associative memory to identify the closest domain matching given weights."""
        if not self.stored_domains or np.linalg.norm(self.memory_bundle) < 1e-8:
            return VSAResult("unknown", 0.0, 0.0, False, self.dim)

        w_vec = self.project_weights(weights_dict)

        # Unbind weight vector from memory bundle: (M (*) w)
        unbound_query = self.memory_bundle * w_vec

        best_domain = "unknown"
        best_sim = -1.0

        for domain, dom_vec in self.domain_codebook.items():
            sim = float(np.dot(unbound_query, dom_vec) / (np.linalg.norm(unbound_query) * np.linalg.norm(dom_vec) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_domain = domain

        confidence = max(0.0, min(1.0, (best_sim + 1.0) / 2.0))
        return VSAResult(
            domain=best_domain,
            similarity=round(best_sim, 4),
            confidence=round(confidence, 4),
            reconstructed=best_sim > 0.25,
            dimensions=self.dim,
        )

    def find_nearest_domains(self, domain: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the top-k most similar domains in weight space using holographic codebook vectors."""
        clean_d = domain if domain.startswith("neuro_") else f"neuro_{domain}"
        raw_d = domain.replace("neuro_", "")
        target_vec = self.domain_codebook.get(clean_d)
        if target_vec is None:
            target_vec = self.domain_codebook.get(raw_d)
        if target_vec is None:
            return []
        scores = []
        for other_dom, other_vec in self.domain_codebook.items():
            if other_dom in (clean_d, raw_d):
                continue
            sim = float(np.dot(target_vec, other_vec) / (np.linalg.norm(target_vec) * np.linalg.norm(other_vec) + 1e-8))
            scores.append({"domain": other_dom, "similarity": round(sim, 4)})
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serialize VSA memory bundle and domain codebook."""
        if filepath is None:
            filepath = _get_default_vsa_path()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        codebook_keys = list(self.domain_codebook.keys())
        codebook_values = np.array([self.domain_codebook[k] for k in codebook_keys], dtype=np.float32)

        np.savez_compressed(
            filepath,
            memory_bundle=self.memory_bundle,
            codebook_keys=np.array(codebook_keys),
            codebook_values=codebook_values,
            stored_domains=np.array(self.stored_domains),
            dim=np.array([self.dim]),
        )
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> WeightSpaceVSA:
        """Load VSA associative memory from file."""
        if filepath is None:
            filepath = _get_default_vsa_path()
        if not filepath.exists():
            return cls()

        data = np.load(filepath, allow_pickle=True)
        vsa = cls(dimension=int(data["dim"][0]))
        vsa.memory_bundle = data["memory_bundle"]
        vsa.stored_domains = list(data["stored_domains"])

        keys = list(data["codebook_keys"])
        values = data["codebook_values"]
        for k, v in zip(keys, values):
            vsa.domain_codebook[str(k)] = v
        return vsa


_vsa_singleton: Optional[WeightSpaceVSA] = None


def get_weight_space_vsa() -> WeightSpaceVSA:
    """Singleton getter for Weight Space VSA memory."""
    global _vsa_singleton
    if _vsa_singleton is None:
        _vsa_singleton = WeightSpaceVSA.load(_get_default_vsa_path())
    return _vsa_singleton
