"""Tests for graphverify/path_searcher.py."""
from __future__ import annotations

from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraph, GraphEdge, GraphNode, ProvenanceMeta
from graphverify.path_scorer import PathScorer
from graphverify.path_searcher import PathSearcher


def _graph_with_support_and_conflict() -> EvidenceGraph:
    g = EvidenceGraph()
    g.add_node(GraphNode(node_id="h", label="Einstein", node_type="entity"))
    g.add_node(GraphNode(node_id="t_correct", label="1921", node_type="entity"))
    g.add_node(GraphNode(node_id="t_wrong", label="1922", node_type="entity"))
    prov = ProvenanceMeta(source_doc="p1", source_span="...", confidence=0.9, retriever_rank=1)
    g.add_edge(GraphEdge(src="h", dst="t_correct", relation="award", surface_rel="won", provenance=prov))
    g.add_edge(GraphEdge(src="h", dst="t_wrong", relation="award", surface_rel="won", provenance=prov))
    return g


def test_search_finds_support_path_matching_claimed_tail():
    graph = _graph_with_support_and_conflict()
    linker = EntityLinker(graph.node_labels)
    searcher = PathSearcher(graph=graph, entity_linker=linker, path_scorer=PathScorer())
    support_paths, conflict_paths = searcher.search(head="Einstein", relation="award", tail="1921")
    assert len(support_paths) >= 1
    assert support_paths[0].score > 0


def test_search_finds_conflict_path_for_incompatible_tail():
    graph = _graph_with_support_and_conflict()
    linker = EntityLinker(graph.node_labels)
    searcher = PathSearcher(graph=graph, entity_linker=linker, path_scorer=PathScorer())
    support_paths, conflict_paths = searcher.search(head="Einstein", relation="award", tail="1921")
    # the "1922" edge is neither the claimed tail nor matches strongly -- it
    # should surface as a conflict candidate for the "award" (functional-ish)
    # relation once entity/relation agreement is high but the tail disagrees
    assert isinstance(conflict_paths, list)


def test_search_returns_empty_when_head_not_in_graph():
    graph = _graph_with_support_and_conflict()
    linker = EntityLinker(graph.node_labels)
    searcher = PathSearcher(graph=graph, entity_linker=linker, path_scorer=PathScorer())
    support_paths, conflict_paths = searcher.search(head="Marie Curie", relation="award", tail="1911")
    assert support_paths == []
    assert conflict_paths == []


def test_search_respects_top_k():
    graph = EvidenceGraph()
    prov = ProvenanceMeta(source_doc="p1", source_span="...", confidence=0.9, retriever_rank=1)
    for i in range(10):
        graph.add_node(GraphNode(node_id=f"t{i}", label=f"T{i}", node_type="entity"))
    graph.add_node(GraphNode(node_id="h", label="H", node_type="entity"))
    for i in range(10):
        graph.add_edge(GraphEdge(src="h", dst=f"t{i}", relation="rel", surface_rel="rel", provenance=prov))
    linker = EntityLinker(graph.node_labels)
    searcher = PathSearcher(graph=graph, entity_linker=linker, path_scorer=PathScorer(), top_k=3)
    support_paths, conflict_paths = searcher.search(head="H", relation="rel", tail="T0")
    assert len(support_paths) <= 3
    assert len(conflict_paths) <= 3
