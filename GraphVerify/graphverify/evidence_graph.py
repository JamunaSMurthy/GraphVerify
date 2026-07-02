"""
Provenance-linked directed graph built from retrieved passages.

Nodes represent entities, passages, and evidence spans.
Edges represent normalized relations, each carrying provenance metadata:
source span, source document, extraction confidence, and retriever rank.

The graph is serialisable to/from JSON for disk caching between pipeline stages.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import networkx as nx

from .prompts import load_prompt


@dataclass
class ProvenanceMeta:
    source_doc:     str
    source_span:    str
    confidence:     float
    retriever_rank: int
    timestamp:      Optional[str] = None


@dataclass
class GraphNode:
    node_id:   str
    label:     str
    node_type: str = "entity"   # entity | passage | claim | span
    aliases:   List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    src:         str
    dst:         str
    relation:    str
    surface_rel: str
    provenance:  ProvenanceMeta
    weight:      float = 1.0


class EvidenceGraph:

    def __init__(self) -> None:
        self._G: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: Dict[str, GraphNode] = {}

    def add_node(self, node: GraphNode) -> None:
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._G.add_node(node.node_id, **asdict(node))

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.src not in self._nodes:
            self.add_node(GraphNode(node_id=edge.src, label=edge.src))
        if edge.dst not in self._nodes:
            self.add_node(GraphNode(node_id=edge.dst, label=edge.dst))
        self._G.add_edge(
            edge.src,
            edge.dst,
            relation=edge.relation,
            surface_rel=edge.surface_rel,
            provenance=asdict(edge.provenance),
            weight=edge.weight,
        )

    @property
    def node_labels(self) -> List[str]:
        return [self._nodes[nid].label for nid in self._G.nodes()]

    @property
    def node_ids(self) -> List[str]:
        return list(self._G.nodes())

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def get_edges(self, src: str, dst: str) -> List[Dict]:
        data = self._G.get_edge_data(src, dst)
        return list(data.values()) if data else []

    def neighbors(self, node_id: str) -> List[str]:
        return list(self._G.successors(node_id))

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nx_graph(self) -> nx.MultiDiGraph:
        return self._G

    def induced_subgraph(self, node_ids) -> "EvidenceGraph":
        """
        Returns a new EvidenceGraph containing only the given node ids and
        the edges between them, preserving provenance metadata. Used by the
        GraphRAG-adapted and HippoRAG-adapted baselines
        (`baselines/graphrag_adapted.py`, `baselines/hipporag_adapted.py`)
        to restrict path search to a retrieved community/personalized-PageRank
        subgraph instead of the full provenance-linked graph, so the shared
        verdict head (:class:`graphverify.path_searcher.PathSearcher` +
        :class:`graphverify.verdict_assigner.VerdictAssigner`) only sees the
        evidence that baseline's retrieval strategy actually selected.
        """
        keep = set(node_ids)
        sub = EvidenceGraph()
        for nid in keep:
            node = self._nodes.get(nid)
            if node is not None:
                sub.add_node(node)
        for u, v, data in self._G.edges(data=True):
            if u in keep and v in keep:
                prov = data.get("provenance", {})
                sub.add_edge(GraphEdge(
                    src=u, dst=v,
                    relation=data.get("relation", ""),
                    surface_rel=data.get("surface_rel", ""),
                    provenance=ProvenanceMeta(**prov) if isinstance(prov, dict) else prov,
                    weight=data.get("weight", 1.0),
                ))
        return sub

    def __len__(self) -> int:
        return self._G.number_of_nodes()

    def to_dict(self) -> Dict[str, Any]:
        nodes = [asdict(n) for n in self._nodes.values()]
        edges = []
        for u, v, k, data in self._G.edges(data=True, keys=True):
            edges.append({"src": u, "dst": v, "key": k, **data})
        return {"nodes": nodes, "edges": edges}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvidenceGraph":
        g = cls()
        for n in d.get("nodes", []):
            n.pop("provenance", None)
            g.add_node(GraphNode(**n))
        for e in d.get("edges", []):
            src = e.pop("src")
            dst = e.pop("dst")
            e.pop("key", None)
            prov = ProvenanceMeta(**e.pop("provenance", {}))
            g.add_edge(GraphEdge(
                src=src, dst=dst,
                relation=e.get("relation", ""),
                surface_rel=e.get("surface_rel", ""),
                provenance=prov,
                weight=e.get("weight", 1.0),
            ))
        return g

    @classmethod
    def from_json(cls, s: str) -> "EvidenceGraph":
        return cls.from_dict(json.loads(s))


class EvidenceGraphBuilder:
    """Builds an EvidenceGraph from retrieved passages using LLM triple extraction."""

    def __init__(self, llm_client: Any, relation_normalizer: Any,
                 embed_model: str = "BAAI/bge-base-en-v1.5") -> None:
        self._llm = llm_client
        self._rel_norm = relation_normalizer
        self._embed_model = embed_model

    def build(self, query: str, passages: List[Dict[str, Any]]) -> EvidenceGraph:
        """
        Build a graph from retrieved passages.

        Each passage dict requires: text, id, rank.
        Optional: score (defaults to 1/rank if missing).
        """
        graph = EvidenceGraph()

        for passage in passages:
            pid        = str(passage.get("id", "p_unknown"))
            rank       = int(passage.get("rank", 1))
            text       = passage.get("text", "")
            retr_score = float(passage.get("score", 1.0 / rank))

            for triple in self._extract_triples(text):
                subj = str(triple.get("subject", "")).strip()
                rel  = str(triple.get("relation", "")).strip()
                obj  = str(triple.get("object",  "")).strip()
                span = str(triple.get("span", text[:120])).strip()
                ts   = triple.get("timestamp")

                if not subj or not rel or not obj:
                    continue

                canon_rel, rel_conf = self._rel_norm.normalize(rel)

                h_id = _node_id(subj)
                t_id = _node_id(obj)
                p_id = f"passage_{pid}"

                graph.add_node(GraphNode(node_id=h_id, label=subj, node_type="entity"))
                graph.add_node(GraphNode(node_id=t_id, label=obj,  node_type="entity"))
                graph.add_node(GraphNode(node_id=p_id, label=pid,  node_type="passage"))

                prov_conf = min(1.0, retr_score * (0.5 + 0.5 * rel_conf))

                graph.add_edge(GraphEdge(
                    src=h_id, dst=t_id,
                    relation=canon_rel,
                    surface_rel=rel,
                    provenance=ProvenanceMeta(
                        source_doc=pid,
                        source_span=span[:300],
                        confidence=prov_conf,
                        retriever_rank=rank,
                        timestamp=str(ts) if ts else None,
                    ),
                    weight=prov_conf,
                ))

        return graph

    def _extract_triples(self, passage: str) -> List[Dict]:
        if not passage.strip():
            return []
        messages = [
            {"role": "system", "content": load_prompt("graph_triple_extraction_system")},
            {"role": "user",   "content": load_prompt("graph_triple_extraction_user").format(passage=passage[:2000])},
        ]
        result = self._llm.chat_json(messages)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "triples" in result:
            return result["triples"]
        return []


def _node_id(label: str) -> str:
    return hashlib.md5(label.lower().strip().encode()).hexdigest()[:12]


def merge_external_kg(graph: EvidenceGraph, kg_path: str) -> int:
    """
    Merges curated (head, relation, tail) triples from an external KG file
    into `graph` in place, used by ``evidence_mode="kg_paths"`` / ``"hybrid"``
    (see :class:`graphverify.config.GraphVerifyConfig`).

    `kg_path` is a JSONL file, one triple per line:
    ``{"head": "...", "relation": "...", "tail": "...", "confidence": 1.0}``
    (``confidence`` optional, defaults to 1.0 — external KG triples are
    assumed curated/reliable unless stated otherwise). Each triple becomes an
    edge with ``provenance.source_doc="external_kg"`` and
    ``provenance.retriever_rank=0`` so path scoring
    (:mod:`graphverify.path_scorer`) can still weight it, but downstream
    analysis can identify which edges came from the KG rather than from a
    retrieved passage.

    Returns the number of triples merged. Raises FileNotFoundError if
    `kg_path` does not exist — callers that want a silent no-op for a
    missing/unset path should check `kg_path` themselves before calling this
    (see ``GraphVerify._build_graph`` for the documented fallback behavior).
    """
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"External KG file not found: {kg_path}")

    n_merged = 0
    with open(kg_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triple = json.loads(line)
            subj = str(triple.get("head", "")).strip()
            rel  = str(triple.get("relation", "")).strip()
            obj  = str(triple.get("tail", "")).strip()
            if not subj or not rel or not obj:
                continue
            confidence = float(triple.get("confidence", 1.0))

            h_id = _node_id(subj)
            t_id = _node_id(obj)
            graph.add_node(GraphNode(node_id=h_id, label=subj, node_type="entity"))
            graph.add_node(GraphNode(node_id=t_id, label=obj, node_type="entity"))
            graph.add_edge(GraphEdge(
                src=h_id, dst=t_id,
                relation=rel, surface_rel=rel,
                provenance=ProvenanceMeta(
                    source_doc="external_kg", source_span="", confidence=confidence,
                    retriever_rank=0, timestamp=triple.get("timestamp"),
                ),
                weight=confidence,
            ))
            n_merged += 1

    return n_merged
