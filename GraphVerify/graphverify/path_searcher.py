"""
Finds top-K support and conflict paths for a claim triple in the evidence graph.

Support paths connect (head → ... → tail) under the claimed relation.
Conflict paths connect (head → ... → different_tail) where the tail is
incompatible with the claim's tail value.

Search is DFS-limited to L_max hops and returns the top-K paths by score.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .evidence_graph import EvidenceGraph
from .incompatibility import is_incompatible
from .path_scorer import PathScorer, ScoredPath
from .entity_linker import EntityLinker
from .config import L_MAX, TOP_K_PATHS


class PathSearcher:

    def __init__(
        self,
        graph: EvidenceGraph,
        entity_linker: EntityLinker,
        path_scorer: PathScorer,
        l_max: int = L_MAX,
        top_k: int = TOP_K_PATHS,
    ) -> None:
        self._G = graph
        self._G_nx = graph.nx_graph()
        self._linker = entity_linker
        self._scorer = path_scorer
        self._l_max = l_max
        self._top_k = top_k

    def search(
        self,
        head: str,
        relation: str,
        tail: str,
    ) -> Tuple[List[ScoredPath], List[ScoredPath]]:
        """
        Returns (support_paths, conflict_paths) each sorted by score descending.
        Returns empty lists when head or tail cannot be located in the graph.
        """
        head_nodes = self._candidate_nodes(head)
        tail_nodes  = self._candidate_nodes(tail)

        support_paths: List[ScoredPath] = []
        conflict_paths: List[ScoredPath] = []

        for h_node in head_nodes:
            for node_seq in self._enumerate_paths(h_node):
                if len(node_seq) < 2:
                    continue

                edge_list = self._path_to_edges(node_seq)
                if not edge_list:
                    continue

                scored = self._scorer.score_path(edge_list, head, relation, tail)

                path_tail_node = self._G.get_node(node_seq[-1])
                path_tail_str  = path_tail_node.label if path_tail_node else node_seq[-1]

                tail_matches = any(t_node == node_seq[-1] for t_node in tail_nodes)
                if tail_matches or scored.tail_score >= 0.75:
                    support_paths.append(scored)
                elif is_incompatible(claim_tail=tail, path_tail=path_tail_str, relation=relation):
                    conflict_paths.append(scored)

        support_paths  = sorted(support_paths,  key=lambda x: x.score, reverse=True)
        conflict_paths = sorted(conflict_paths, key=lambda x: x.score, reverse=True)

        return support_paths[: self._top_k], conflict_paths[: self._top_k]

    def _candidate_nodes(self, entity_label: str) -> List[str]:
        node_labels = self._G.node_labels
        node_ids    = self._G.node_ids

        if not node_labels:
            return []

        linker = EntityLinker(node_labels)
        matched_idx, score = linker.link(entity_label)

        if matched_idx is not None and score > 0:
            return [node_ids[matched_idx]]

        # Fuzzy fallback when exact/alias/embed matching all miss
        key = entity_label.lower().strip()
        return [
            nid for nid, node in self._G._nodes.items()
            if key in node.label.lower() or node.label.lower() in key
        ][:5]

    def _enumerate_paths(self, source: str) -> List[List[str]]:
        paths: List[List[str]] = []
        stack: List[Tuple[str, List[str]]] = [(source, [source])]
        max_paths = self._top_k * 10

        while stack and len(paths) < max_paths:
            node, path = stack.pop()
            if len(path) > 1:
                paths.append(path)

            if len(path) >= self._l_max + 1:
                continue

            for nbr in self._G_nx.successors(node):
                if nbr not in path:
                    stack.append((nbr, path + [nbr]))

        return paths

    def _path_to_edges(self, node_seq: List[str]) -> List[Dict]:
        edges = []
        for i in range(len(node_seq) - 1):
            src, dst = node_seq[i], node_seq[i + 1]
            edge_data_list = self._G.get_edges(src, dst)
            if not edge_data_list:
                return []

            best = max(
                edge_data_list,
                key=lambda e: e.get("provenance", {}).get("confidence", 0.0)
                if isinstance(e.get("provenance"), dict) else 0.0,
            )
            src_node = self._G.get_node(src)
            dst_node = self._G.get_node(dst)
            best["src_label"] = src_node.label if src_node else src
            best["dst_label"] = dst_node.label if dst_node else dst
            edges.append(best)
        return edges
