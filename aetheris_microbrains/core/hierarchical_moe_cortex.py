"""
Aetheris Micro-Brain Tool Router (formerly "Hierarchical MoE Cortex").

Routes a natural-language intent to a concrete tool or micro-brain by
deterministic keyword-hash embedding similarity over the REAL catalog:
  - stems of registered dynamic tools (execution/tools/dynamic/*.py)
  - grown nursery brains (memory/store/neuro_brains/*_meta.json)

Honesty note: the previous version hardcoded a 19-name cluster catalog that
contained several phantom entries (names existing nowhere in the tree) and
returned "routed_success" with a literal 0.003 ms latency and a claimed
O(log N) search even when nothing resolved. This version scans only real
catalog sources (a linear scan — labeled as such), reports the measured
wall-clock latency of the scan, and returns an explicit "unrouted" status
when no entry matches.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DYNAMIC_TOOLS_DIR = Path("execution/tools/dynamic")


class HierarchicalMoECortex:
    """Keyword-embedding router over the real tool + micro-brain catalog."""

    _instance: Optional["HierarchicalMoECortex"] = None

    # Below this embedding similarity a match is noise, not a route.
    MATCH_FLOOR = 0.05

    def __init__(self, latent_dim: int = 64, brains_dir: str = "memory/store/neuro_brains"):
        self.latent_dim = latent_dim
        _p = Path(brains_dir)
        self.brains_dir = _p if (_p.exists() or brains_dir != "memory/store/neuro_brains") else (Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")

    @classmethod
    def get_instance(cls) -> "HierarchicalMoECortex":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text into words and sub-words (e.g. 5-grams for concatenated tokens)."""
        clean_text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        raw_words = clean_text.split()
        tokens = []
        for w in raw_words:
            tokens.append(w)
            if len(w) > 6:
                # Add n-grams to catch concatenated compounds like 'consensusconstrained'
                for i in range(len(w) - 4):
                    tokens.append(w[i : i + 5])
        return tokens

    def _embed_text(self, text: str) -> np.ndarray:
        """Deterministic keyword-preserving projection into d=64 sphere."""
        vec = np.zeros(self.latent_dim, dtype=np.float32)
        tokens = self._tokenize(text)
        for tok in tokens:
            if len(tok) <= 2:
                continue
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            pos = h % self.latent_dim
            weight = 1.0 + (len(tok) * 0.2)
            vec[pos] += weight
        norm = np.linalg.norm(vec)
        return vec / (norm if norm > 1e-9 else 1.0)

    def _catalog(self) -> Dict[str, str]:
        """The real routable catalog: {name: source}, rebuilt per call (the sources are live)."""
        catalog: Dict[str, str] = {}
        try:
            for p in DYNAMIC_TOOLS_DIR.glob("*.py"):
                if p.name != "__init__.py":
                    catalog[p.stem] = "dynamic_tool"
        except OSError:
            pass
        try:
            for p in self.brains_dir.glob("*_meta.json"):
                stem = p.name[: -len("_meta.json")]
                tool_name = stem[len("neuro_"):] if stem.startswith("neuro_") else stem
                catalog.setdefault(tool_name, "nursery_brain")
        except OSError:
            pass
        return catalog

    def _resolve_brain_file(self, tool_name: str):
        brain_stem = f"neuro_{tool_name}"
        bitnet_path = self.brains_dir / f"{brain_stem}_meta_bitnet158.json"
        standard_path = self.brains_dir / f"{brain_stem}_meta.json"
        if bitnet_path.exists():
            return str(bitnet_path), True
        if standard_path.exists():
            return str(standard_path), False
        return None, False

    def route_query(self, query_text: str) -> Dict[str, Any]:
        """
        Single-stage linear scan over the real catalog (embedding similarity +
        exact-substring bonus). Honest result contract: `routed_success` only
        when a real catalog entry matches above the floor AND micro-brain
        weights resolve for it, `unrouted` otherwise.
        """
        t0 = time.perf_counter()
        catalog = self._catalog()

        result: Dict[str, Any] = {
            "query": query_text[:80],
            "catalog_size": len(catalog),
            "search_complexity": "linear_scan",
        }

        if not catalog:
            result["status"] = "unrouted"
            result["reason"] = "catalog_empty"
            result["routing_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
            return result

        q_vec = self._embed_text(query_text)
        q_words = [w for w in re.sub(r"[^a-z0-9\s]", " ", query_text.lower()).split() if w]

        best_tool = None
        best_source = None
        best_sim = -1.0
        for t_name, source in catalog.items():
            sim = float(np.dot(q_vec, self._embed_text(t_name)))
            # Exact token match bonus
            for word in q_words:
                if len(word) > 3 and word in t_name.lower():
                    sim += 0.15
            if sim > best_sim:
                best_sim = sim
                best_tool = t_name
                best_source = source

        result["routing_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)

        if best_tool is None or best_sim < self.MATCH_FLOOR:
            result["status"] = "unrouted"
            result["reason"] = "no_catalog_entry_matched"
            result["best_candidate"] = best_tool
            result["best_candidate_similarity"] = round(best_sim, 4)
            return result

        resolved_brain_file, is_quantized = self._resolve_brain_file(best_tool)
        result.update({
            "status": "routed_success",
            "target_tool": best_tool,
            "target_source": best_source,
            "target_microbrain": f"neuro_{best_tool}",
            "microbrain_confidence": round(best_sim, 4),
            "resolved_brain_file": resolved_brain_file,
            "is_bitnet_quantized": is_quantized,
        })
        if resolved_brain_file is None:
            # Routed to a real catalog entry, but no micro-brain weights exist
            # for it — the route FAILS honestly (the previous shape kept
            # status="routed_success" with a target_microbrain nobody could
            # load).  The candidate fields stay for transparency.
            result["status"] = "unrouted"
            result["reason"] = "no_microbrain_weights_for_tool"
        return result
