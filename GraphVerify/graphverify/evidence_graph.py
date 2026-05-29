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
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import networkx as nx


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

    EXTRACT_PROMPT = """Extract factual relation triples from the passage below.
Return a JSON list: [{{"subject": "...", "relation": "...", "object": "...", "span": "...", "timestamp": null}}]
- subject and object: named entities or key noun phrases.
- relation: a concise predicate (e.g. "birthPlace", "directed by", "won award").
- span: exact substring of the passage that supports this triple.
- timestamp: year string (e.g. "2018") if time-stamped, else null.
Return only the JSON list.

Passage:
{passage}"""

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
            {"role": "system", "content": "You are an information-extraction assistant."},
            {"role": "user",   "content": self.EXTRACT_PROMPT.format(passage=passage[:2000])},
        ]
        result = self._llm.chat_json(messages)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "triples" in result:
            return result["triples"]
        return []


def _node_id(label: str) -> str:
    return hashlib.md5(label.lower().strip().encode()).hexdigest()[:12]
