"""
GraphRAG, adapted as a verifier (Edge et al., 2025, "From Local to Global:
A Graph RAG Approach to Query-Focused Summarization").

Original method: builds an entity-relation graph from a corpus, detects
communities, and summarizes each community so retrieval can answer
global/query-focused questions from community summaries rather than raw
passages.

**Adaptation, per the revision plan's fairness table.** GraphRAG is a
retrieval system, not a claim verifier, so it cannot be evaluated
"natively" on claim-level verdicts. This adaptation builds the same
provenance-linked graph GraphVerify would build, reproduces GraphRAG's
defining retrieval mechanism — community detection over the entity-relation
graph (`networkx.algorithms.community.greedy_modularity_communities`) — to
select the subgraph for a claim (the union of the communities containing
its head and tail entities), and then hands that subgraph to GraphVerify's
own shared verdict head (see `baselines/_graph_retrieval_adapted.py`). This
row must always be reported as an *adapted graph-retrieval control*, never
as a native post-hoc verifier baseline.
"""
from __future__ import annotations

from networkx.algorithms.community import greedy_modularity_communities

from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraph

from ._graph_retrieval_adapted import GraphRetrievalAdaptedVerifier


class GraphRAGAdaptedVerifier(GraphRetrievalAdaptedVerifier):
    name = "graphrag_adapted"
    citation = "Edge et al. 2025, 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization' — adapted as a verifier"

    def _select_subgraph(
        self,
        full_graph: EvidenceGraph,
        entity_linker_full: EntityLinker,
        head_label: str,
        tail_label: str,
    ) -> EvidenceGraph:
        undirected = full_graph.nx_graph().to_undirected()
        if undirected.number_of_nodes() < 2 or undirected.number_of_edges() == 0:
            return full_graph

        communities = list(greedy_modularity_communities(undirected))

        keep: set = set()
        for label in (head_label, tail_label):
            keep |= self._community_containing(label, full_graph, entity_linker_full, communities)

        if not keep:
            return full_graph
        return full_graph.induced_subgraph(keep)

    @staticmethod
    def _community_containing(label, full_graph, entity_linker_full, communities) -> set:
        idx, score = entity_linker_full.link(label)
        if idx is None:
            return set()
        node_id = full_graph.node_ids[idx]
        for community in communities:
            if node_id in community:
                return set(community)
        return {node_id}
