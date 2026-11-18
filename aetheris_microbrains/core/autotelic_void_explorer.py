"""
Autotelic Topological Void Explorer for Aetheris.
Ported and evolved from NeoAura/ECHO (echo_core/autopoietic_goal_generator.py) + 2025/2026 SOTA.

Scans the micro-brain semantic space and tool capability graph to detect:
1. Betti-1 Topological Voids (chordless cycles of length >= 4): Conceptual gaps
   where existing capabilities encircle an unexplored capability.
2. Missing 2-simplex fills (triangles with dense pairwise similarity but no unifying synthesis).
3. Exposes the discovered voids via scan_microbrain_population() for callers to inspect.
   No goal synthesis is wired to a consumer: the Curriculum Generator and Evolution
   Reactor do NOT read these results — the only production importer today is
   core/concentric_spiral_dag.py, which counts the voids for telemetry.
"""

from __future__ import annotations
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

log = logging.getLogger("aetheris.autotelic_voids")


@dataclass
class TopologicalVoid:
    """Represents a discovered cognitive void in capability space."""
    void_id: str
    void_type: str  # "BETTI_1_CYCLE" or "MISSING_SIMPLEX"
    member_nodes: List[str]
    endpoints: Tuple[str, str]
    radius: float
    curiosity_priority: float
    suggested_goal_title: str
    suggested_contract_hint: str


class AutotelicVoidExplorer:
    """
    Constructs a Vietoris-Rips proximity graph from tool/microbrain embeddings,
    detects simplicial homology voids, and proposes autotelic self-improvement targets.
    """

    _instance: Optional["AutotelicVoidExplorer"] = None

    def __init__(self, proximity_threshold: float = 0.65, max_cycle_len: int = 6):
        self.proximity_threshold = float(proximity_threshold)
        self.max_cycle_len = int(max_cycle_len)
        self.discovered_voids: List[TopologicalVoid] = []

    @classmethod
    def get_instance(cls) -> "AutotelicVoidExplorer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def scan_microbrain_population(self, brain_dir: Optional[Path] = None) -> List[TopologicalVoid]:
        """
        Scans all micro-brains in memory/store/neuro_brains/, extracts hypervectors/weights,
        builds the similarity manifold, and extracts topological voids.
        """
        _cwd_brains = Path("memory/store/neuro_brains")
        target_dir = brain_dir or (_cwd_brains if _cwd_brains.exists() else Path(__file__).resolve().parents[1] / "memory" / "store" / "neuro_brains")
        if not target_dir.exists():
            return []

        names: List[str] = []
        vectors: List[np.ndarray] = []

        # Load weights from .npz or .npy files
        brain_files = sorted(list(target_dir.glob("*.npz")) + list(target_dir.glob("*.npy")))
        for p in brain_files:
            try:
                loaded = np.load(p, allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    # For npz archives, extract concatenated weight matrices or primary W_embed/W_mlp
                    arrays = [loaded[k] for k in loaded.files if isinstance(loaded[k], np.ndarray) and loaded[k].size > 0]
                    if not arrays:
                        continue
                    # Flatten and average
                    flattened = [a.flatten() for a in arrays]
                    vec = np.concatenate(flattened)
                    if len(vec) > 64:
                        vec = vec[:64]
                elif loaded.ndim == 1:
                    vec = loaded
                elif loaded.ndim >= 2:
                    vec = np.mean(loaded, axis=0) if loaded.shape[0] > 0 else np.zeros(16)
                else:
                    vec = np.zeros(16)

                norm = float(np.linalg.norm(vec))
                if norm > 1e-12:
                    vec = vec / norm
                vectors.append(vec)
                names.append(p.stem)
            except Exception as exc:
                log.warning("Skipping corrupted or unreadable brain file %s: %s", p, exc)
                continue

        if len(names) < 4:
            return []

        # Align dimensionality to uniform length
        max_dim = max(len(v) for v in vectors)
        uniform_vectors = []
        for v in vectors:
            if len(v) < max_dim:
                pad = np.pad(v, (0, max_dim - len(v)), mode="constant")
                uniform_vectors.append(pad)
            else:
                uniform_vectors.append(v[:max_dim])

        matrix = np.stack(uniform_vectors, axis=0)
        return self.find_voids_in_embedding_manifold(names, matrix)

    def find_voids_in_embedding_manifold(self, node_names: List[str], embeddings: np.ndarray) -> List[TopologicalVoid]:
        """
        Builds the adjacency graph based on cosine similarity and extracts chordless cycles
        (Betti-1 holes) and hollow 2-simplices (missing ternary synthesis nodes).
        """
        n = len(node_names)
        if n < 3:
            return []

        # Pairwise cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = embeddings / norms
        sim_matrix = np.dot(normed, normed.T)

        # Adjacency list for graph
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.proximity_threshold:
                    adj[i].add(j)
                    adj[j].add(i)

        discovered: List[TopologicalVoid] = []

        # Find chordless cycles of length 4 to max_cycle_len (Betti-1 holes)
        if n >= 4:
            for i in range(n):
                self._find_cycles_from(i, adj, node_names, sim_matrix, discovered)

        # Find hollow 2-simplices (triangles with dense pairwise edges but no unifying mediator)
        self._find_missing_simplices(adj, node_names, normed, sim_matrix, discovered)

        # Deduplicate voids by set of members + void_type
        unique_voids: List[TopologicalVoid] = []
        seen_keys: Set[str] = set()
        for v in discovered:
            key = f"{v.void_type}::" + "_".join(sorted(v.member_nodes))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_voids.append(v)

        # Sort by curiosity priority descending
        unique_voids.sort(key=lambda x: x.curiosity_priority, reverse=True)
        self.discovered_voids = unique_voids[:10]
        return self.discovered_voids

    def _find_missing_simplices(
        self,
        adj: Dict[int, Set[int]],
        names: List[str],
        normed: np.ndarray,
        sim_matrix: np.ndarray,
        results: List[TopologicalVoid],
    ) -> None:
        """Detects hollow 2-simplices: triangles lacking a unifying synthesis node near their barycenter."""
        n = len(names)
        fill_threshold = max(0.88, self.proximity_threshold + 0.12)
        for i in range(n):
            nbrs_i = sorted(v for v in adj.get(i, set()) if v > i)
            for idx_a, j in enumerate(nbrs_i):
                for k in nbrs_i[idx_a + 1:]:
                    if k not in adj.get(j, set()):
                        continue
                    if len(results) >= 20:
                        return
                    mean_sim = float((sim_matrix[i, j] + sim_matrix[i, k] + sim_matrix[j, k]) / 3.0)
                    if mean_sim >= 0.98:
                        continue
                    bary = normed[i] + normed[j] + normed[k]
                    bary_norm = float(np.linalg.norm(bary))
                    if bary_norm <= 1e-12:
                        continue
                    bary_unit = bary / bary_norm
                    has_filler = any(
                        float(np.dot(normed[m], bary_unit)) >= fill_threshold
                        for m in range(n)
                        if m not in (i, j, k)
                    )
                    if not has_filler:
                        triad_names = [names[i], names[j], names[k]]
                        v_id = f"simplex_{hashlib.md5(''.join(sorted(triad_names)).encode()).hexdigest()[:8]}"
                        curiosity = round((1.0 - mean_sim) * 3.5, 3)
                        results.append(TopologicalVoid(
                            void_id=v_id,
                            void_type="MISSING_SIMPLEX",
                            member_nodes=triad_names,
                            endpoints=(names[i], names[j]),
                            radius=round(1.0 - mean_sim, 3),
                            curiosity_priority=curiosity,
                            suggested_goal_title=f"synthesize_simplex_{names[i][:12]}_{names[j][:12]}",
                            suggested_contract_hint=(
                                f"Unify hollow 2-simplex triad ({names[i]}, {names[j]}, {names[k]}) "
                                "into a composite synthesis capability."
                            ),
                        ))

    def _find_cycles_from(
        self,
        start_node: int,
        adj: Dict[int, Set[int]],
        names: List[str],
        sim_matrix: np.ndarray,
        results: List[TopologicalVoid]
    ):
        """Simple DFS cycle finder for small subgraphs."""
        def dfs(curr: int, path: List[int], visited: Set[int]):
            if len(path) > self.max_cycle_len or len(results) >= 20:
                return
            for nxt in sorted(adj.get(curr, set())):
                if nxt == start_node and len(path) >= 4:
                    # Found potential cycle: check chordlessness
                    cycle_nodes = list(path)
                    if self._is_chordless(cycle_nodes, adj):
                        v_id = f"void_{hashlib.md5(''.join(sorted(names[x] for x in cycle_nodes)).encode()).hexdigest()[:8]}"
                        ep1 = names[cycle_nodes[0]]
                        ep2 = names[cycle_nodes[len(cycle_nodes) // 2]]
                        # Compute radius as avg distance
                        mean_sim = float(np.mean([sim_matrix[cycle_nodes[k], cycle_nodes[(k+1)%len(cycle_nodes)]] for k in range(len(cycle_nodes))]))
                        # Curiosity reward is higher for larger cycles with tight perimeter
                        curiosity = round((1.0 - mean_sim) * float(len(cycle_nodes)) * 1.5, 3)

                        title = f"synthesize_bridge_for_{ep1[:15]}_{ep2[:15]}"
                        hint = f"Bridge semantic topological void between {ep1} and {ep2} encircling {len(cycle_nodes)} nodes."

                        results.append(TopologicalVoid(
                            void_id=v_id,
                            void_type="BETTI_1_CYCLE",
                            member_nodes=[names[x] for x in cycle_nodes],
                            endpoints=(ep1, ep2),
                            radius=round(1.0 - mean_sim, 3),
                            curiosity_priority=curiosity,
                            suggested_goal_title=title,
                            suggested_contract_hint=hint,
                        ))
                    continue
                elif nxt not in visited and len(path) < self.max_cycle_len:
                    # Enforce ordering to avoid exploring symmetric cyclic permutations
                    if nxt > start_node:
                        visited.add(nxt)
                        path.append(nxt)
                        dfs(nxt, path, visited)
                        path.pop()
                        visited.remove(nxt)

        dfs(start_node, [start_node], {start_node})

    def _is_chordless(self, cycle: List[int], adj: Dict[int, Set[int]]) -> bool:
        """Verifies no shortcut edges exist in the cycle."""
        k = len(cycle)
        cycle_set = set(cycle)
        edge_count = 0
        for i in range(k):
            u = cycle[i]
            for v in adj.get(u, set()):
                if v in cycle_set:
                    edge_count += 1
        # In a chordless simple cycle, total directed edges between members is exactly 2 * k
        return edge_count == 2 * k

