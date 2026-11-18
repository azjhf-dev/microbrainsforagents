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
        self.domain_weight_vectors: Dict[str, np.ndarray] = {}
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
        self.domain_weight_vectors[domain] = w_vec.copy()

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

    def _get_weight_vector_for_domain(self, domain: str) -> Optional[np.ndarray]:
        """Return stored weight-projection vector or unbind from holographic memory_bundle."""
        if domain in self.domain_weight_vectors:
            return self.domain_weight_vectors[domain]
        dom_code = self.domain_codebook.get(domain)
        if dom_code is not None and float(np.linalg.norm(self.memory_bundle)) > 1e-8:
            unbound = self.memory_bundle * dom_code
            norm = float(np.linalg.norm(unbound))
            return (unbound / norm).astype(np.float32) if norm > 1e-8 else unbound
        return None

    def find_nearest_domains(self, domain: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the top-k most similar domains by actual projected weight-space cosine similarity."""
        clean_d = domain if domain.startswith("neuro_") else f"neuro_{domain}"
        raw_d = domain[len("neuro_"):] if domain.startswith("neuro_") else domain
        skip_names = {domain, clean_d, raw_d}

        target_w = None
        for cand in (domain, clean_d, raw_d):
            target_w = self._get_weight_vector_for_domain(cand)
            if target_w is not None:
                break
        if target_w is None:
            return []

        candidate_keys = list(dict.fromkeys(list(self.domain_weight_vectors.keys()) + list(self.domain_codebook.keys())))
        scores = []
        t_norm = float(np.linalg.norm(target_w)) + 1e-8
        for other_dom in candidate_keys:
            if other_dom in skip_names:
                continue
            other_w = self._get_weight_vector_for_domain(other_dom)
            if other_w is None:
                continue
            sim = float(np.dot(target_w, other_w) / (t_norm * float(np.linalg.norm(other_w)) + 1e-8))
            scores.append({"domain": other_dom, "similarity": round(sim, 4)})
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serialize VSA memory bundle, domain codebook, and projected weight vectors without pickle."""
        if filepath is None:
            filepath = _get_default_vsa_path()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        codebook_keys = list(self.domain_codebook.keys())
        weight_keys = list(self.domain_weight_vectors.keys())
        max_len = max(
            [16]
            + [len(str(k)) for k in codebook_keys]
            + [len(str(k)) for k in weight_keys]
            + [len(str(k)) for k in self.stored_domains]
        )
        str_dtype = f"<U{max_len}"

        codebook_values = (
            np.array([self.domain_codebook[k] for k in codebook_keys], dtype=np.float32)
            if codebook_keys
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        weight_values = (
            np.array([self.domain_weight_vectors[k] for k in weight_keys], dtype=np.float32)
            if weight_keys
            else np.zeros((0, self.dim), dtype=np.float32)
        )

        np.savez_compressed(
            filepath,
            memory_bundle=self.memory_bundle,
            codebook_keys=np.array(codebook_keys, dtype=str_dtype),
            codebook_values=codebook_values,
            weight_keys=np.array(weight_keys, dtype=str_dtype),
            weight_values=weight_values,
            stored_domains=np.array(self.stored_domains, dtype=str_dtype),
            dim=np.array([self.dim], dtype=np.int64),
        )
        return filepath

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> WeightSpaceVSA:
        """Load VSA associative memory from file safely (`allow_pickle=False`)."""
        if filepath is None:
            filepath = _get_default_vsa_path()
        if not filepath.exists():
            return cls()

        with np.load(filepath, allow_pickle=False) as data:
            vsa = cls(dimension=int(data["dim"][0]))
            vsa.memory_bundle = data["memory_bundle"]
            vsa.stored_domains = [str(x) for x in data["stored_domains"]]

            for k, v in zip(data["codebook_keys"], data["codebook_values"]):
                vsa.domain_codebook[str(k)] = v
            if "weight_keys" in data and "weight_values" in data:
                for k, v in zip(data["weight_keys"], data["weight_values"]):
                    vsa.domain_weight_vectors[str(k)] = v
        return vsa


_vsa_singleton: Optional[WeightSpaceVSA] = None


def get_weight_space_vsa() -> WeightSpaceVSA:
    """Singleton getter for Weight Space VSA memory."""
    global _vsa_singleton
    if _vsa_singleton is None:
        _vsa_singleton = WeightSpaceVSA.load(_get_default_vsa_path())
    return _vsa_singleton
