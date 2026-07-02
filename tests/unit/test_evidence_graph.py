"""Tests for graphverify/evidence_graph.py."""
from __future__ import annotations

import json

import pytest

from graphverify.evidence_graph import (
    EvidenceGraph,
    EvidenceGraphBuilder,
    GraphEdge,
    GraphNode,
    ProvenanceMeta,
    merge_external_kg,
)
from graphverify.relation_normalizer import RelationNormalizer


def _small_graph() -> EvidenceGraph:
    g = EvidenceGraph()
    g.add_node(GraphNode(node_id="h1", label="Einstein", node_type="entity"))
    g.add_node(GraphNode(node_id="t1", label="Ulm", node_type="entity"))
    g.add_edge(GraphEdge(
        src="h1", dst="t1", relation="birthPlace", surface_rel="born in",
        provenance=ProvenanceMeta(source_doc="p1", source_span="...", confidence=0.9, retriever_rank=1),
    ))
    return g


def test_add_node_and_edge():
    g = _small_graph()
    assert len(g) == 2
    assert g.nx_graph().number_of_edges() == 1
    assert "Einstein" in g.node_labels


def test_to_dict_and_from_dict_roundtrip():
    g = _small_graph()
    d = g.to_dict()
    g2 = EvidenceGraph.from_dict(d)
    assert g2.node_labels == g.node_labels
    assert g2.nx_graph().number_of_edges() == g.nx_graph().number_of_edges()
    edges = g2.get_edges("h1", "t1")
    assert edges[0]["relation"] == "birthPlace"


def test_to_json_and_from_json_roundtrip():
    g = _small_graph()
    g2 = EvidenceGraph.from_json(g.to_json())
    assert g2.node_labels == g.node_labels


def test_induced_subgraph_keeps_only_selected_nodes_and_their_edges():
    g = _small_graph()
    g.add_node(GraphNode(node_id="h2", label="Other", node_type="entity"))
    sub = g.induced_subgraph({"h1", "t1"})
    assert set(sub.node_ids) == {"h1", "t1"}
    assert sub.nx_graph().number_of_edges() == 1


def test_builder_builds_graph_from_passages(fake_llm):
    builder = EvidenceGraphBuilder(llm_client=fake_llm, relation_normalizer=RelationNormalizer())
    passages = [{"id": "p1", "text": "Einstein was born in Ulm.", "rank": 1, "score": 0.9}]
    graph = builder.build("query", passages)
    assert len(graph) > 0
    assert graph.nx_graph().number_of_edges() > 0


def test_builder_empty_passages_returns_empty_graph(fake_llm):
    builder = EvidenceGraphBuilder(llm_client=fake_llm, relation_normalizer=RelationNormalizer())
    graph = builder.build("query", [])
    assert len(graph) == 0


def test_merge_external_kg_adds_triples(tmp_path):
    g = EvidenceGraph()
    kg_path = tmp_path / "kg.jsonl"
    kg_path.write_text(
        json.dumps({"head": "Einstein", "relation": "award", "tail": "Nobel Prize"}) + "\n"
    )
    n_merged = merge_external_kg(g, str(kg_path))
    assert n_merged == 1
    assert len(g) == 2
    edges = list(g.nx_graph().edges(data=True))
    assert edges[0][2]["provenance"]["source_doc"] == "external_kg"


def test_merge_external_kg_missing_file_raises(tmp_path):
    g = EvidenceGraph()
    with pytest.raises(FileNotFoundError):
        merge_external_kg(g, str(tmp_path / "does_not_exist.jsonl"))
