"""
HippoRAG, adapted as a verifier (Gutiérrez et al., 2024, "HippoRAG:
Neurobiologically Inspired Long-Term Memory for Large Language Models").

Original method: builds a knowledge graph over the corpus and retrieves by
running personalized PageRank seeded at query-relevant entities, mimicking
associative memory retrieval.

**Adaptation, per the revision plan's fairness table.** Like GraphRAG,
HippoRAG is a retrieval system, not a claim verifier. This adaptation
builds the same provenance-linked graph GraphVerify would build, reproduces
HippoRAG's defining retrieval mechanism — personalized PageRank
(`networkx.pagerank`) seeded at the claim's head/tail entities — to select
the top-scoring subgraph for a claim, and hands that subgraph to
GraphVerify's own shared verdict head (see
`baselines/_graph_retrieval_adapted.py`). This row must always be reported
as an *adapted graph-retrieval control*, never as a native post-hoc
verifier baseline.
"""
from __future__ import annotations

import networkx as nx

from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraph

from ._graph_retrieval_adapted import GraphRetrievalAdaptedVerifier


class HippoRAGAdaptedVerifier(GraphRetrievalAdaptedVerifier):
    name = "hipporag_adapted"
    citation = "Gutiérrez et al. 2024, 'HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models' — adapted as a verifier"

    def __init__(self, *args, top_n_nodes: int = 30, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._top_n_nodes = top_n_nodes

    def _select_subgraph(
        self,
        full_graph: EvidenceGraph,
        entity_linker_full: EntityLinker,
        head_label: str,
        tail_label: str,
    ) -> EvidenceGraph:
        graph = full_graph.nx_graph()
        if graph.number_of_nodes() < 2:
            return full_graph

        seeds = []
        for label in (head_label, tail_label):
            idx, score = entity_linker_full.link(label)
            if idx is not None:
                seeds.append(full_graph.node_ids[idx])
        if not seeds:
            return full_graph

        personalization = {n: (1.0 if n in seeds else 0.0) for n in graph.nodes()}
        try:
            scores = nx.pagerank(graph, personalization=personalization, alpha=0.85, weight="weight")
        except Exception:
            # pagerank fails to converge (or the graph is degenerate) on rare
            # small/disconnected graphs; fall back to the full graph rather
            # than silently returning an empty subgraph.
            return full_graph

        ranked_nodes = sorted(scores, key=scores.get, reverse=True)[: self._top_n_nodes]
        keep = set(ranked_nodes) | set(seeds)
        return full_graph.induced_subgraph(keep)
