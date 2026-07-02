"""
GraphCheck-style KG fact-checking (Chen et al., 2025b, "GraphCheck: Breaking
Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking").

Original method: encodes an extracted claim knowledge graph and a document
knowledge graph as graph embeddings and performs binary fact-checking by
comparing them.

**Reimplementation note.** We do not have the original GNN encoder or
training data, so this is an independent, from-description
reimplementation: the "claim KG" is the single (head, relation, tail)
triple extracted from the claim (`prompts/graphcheck_triple_*.txt`, no
entity linking against the document graph — GraphCheck's whole point is
that it does *not* require discrete entity linking, only embedding
similarity); the "document KG" is the same provenance-linked graph
GraphVerify builds from the retrieved passages
(:class:`graphverify.evidence_graph.EvidenceGraphBuilder`). Each document
edge is embedded as ``"{head} {relation} {tail}"`` text
(:class:`graphverify.embedder.Embedder`) and compared to the claim triple's
embedding by cosine similarity; the highest-similarity document edge is the
match. If the match's similarity clears ``support_threshold`` and its tail
agrees with the claim's tail (:func:`graphverify.incompatibility.classify_incompatibility`
returns None), the claim is Supported; if it clears the threshold but the
tail is incompatible, it is Contradictory; otherwise Unsupported. This
mechanism is deliberately unlike GraphVerify's symbolic path search — no
provenance scoring, no multi-hop paths, no conflict-priority ordering —
matching the "graph embeddings for binary fact-checking" description in the
revision plan's baseline table, extended to a three-way verdict only
through the same incompatibility check GraphVerify itself uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from graphverify.embedder import Embedder
from graphverify.evidence_graph import EvidenceGraphBuilder
from graphverify.incompatibility import classify_incompatibility
from graphverify.prompts import load_prompt
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier


class GraphCheckAdaptedVerifier(BaselineVerifier):
    name = "graphcheck_adapted"
    citation = "Chen et al. 2025b, 'GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking'"
    category = "kg_fact_checking"
    uses_graph = True

    def __init__(
        self,
        llm_client: Any,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        support_threshold: float = 0.75,
        contradict_threshold: float = 0.70,
    ) -> None:
        super().__init__(llm_client)
        self._embed_model = embed_model
        self._rel_norm = RelationNormalizer(embed_model=embed_model)
        self._embedder: Optional[Embedder] = None  # lazily constructed; see `_get_embedder`
        self._support_threshold = support_threshold
        self._contradict_threshold = contradict_threshold

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(self._embed_model)
        return self._embedder

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        doc_graph = EvidenceGraphBuilder(
            llm_client=self._llm, relation_normalizer=self._rel_norm, embed_model=self._embed_model,
        ).build(query, passages)
        doc_edges = self._doc_edge_texts(doc_graph)

        records = []
        for claim in claims:
            records.append(self._verify_one(claim, doc_edges))
        return records

    def _doc_edge_texts(self, doc_graph) -> List[Tuple[str, str, str, str]]:
        """Returns (text, head_label, relation, tail_label) for every document-KG edge."""
        edges = []
        for u, v, data in doc_graph.nx_graph().edges(data=True):
            u_node, v_node = doc_graph.get_node(u), doc_graph.get_node(v)
            head = u_node.label if u_node else u
            tail = v_node.label if v_node else v
            relation = data.get("relation", "")
            edges.append((f"{head} {relation} {tail}", head, relation, tail))
        return edges

    def _verify_one(self, claim: str, doc_edges: List[Tuple[str, str, str, str]]) -> VerificationRecord:
        triple = self._extract_claim_triple(claim)
        if triple is None or not doc_edges:
            return VerificationRecord(
                claim=claim, head=None, relation="", tail=None,
                verdict="Unsupported", best_path=None, reliability=0.0,
                path_type="none", triple_linked=False, verdict_mode=self.name,
                rationale="No claim triple extracted or empty document KG.",
            )
        head, relation, tail = triple

        embedder = self._get_embedder()
        claim_vec = embedder.encode([f"{head} {relation} {tail}"])[0]
        doc_texts = [e[0] for e in doc_edges]
        doc_vecs = embedder.encode(doc_texts)
        sims = embedder.cosine_sim_matrix(claim_vec.reshape(1, -1), doc_vecs)[0]

        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        _, match_head, match_relation, match_tail = doc_edges[best_idx]

        if best_sim >= self._contradict_threshold and classify_incompatibility(tail, match_tail, relation) is not None:
            verdict = "Contradictory"
            confidence = best_sim
        elif best_sim >= self._support_threshold:
            verdict = "Supported"
            confidence = best_sim
        else:
            verdict = "Unsupported"
            confidence = best_sim

        return VerificationRecord(
            claim=claim, head=head, relation=relation, tail=tail,
            verdict=verdict,
            best_path=[f"{match_head} --[{match_relation}]--> {match_tail}"] if verdict != "Unsupported" else None,
            reliability=confidence,
            support_score=confidence if verdict == "Supported" else 0.0,
            contradict_score=confidence if verdict == "Contradictory" else 0.0,
            path_type="kg_embedding_match" if verdict != "Unsupported" else "none",
            triple_linked=True, verdict_mode=self.name,
            rationale=f"max KG-embedding cosine similarity={best_sim:.3f} against '{match_head} {match_relation} {match_tail}'.",
        )

    def _extract_claim_triple(self, claim: str) -> Optional[Tuple[str, str, str]]:
        result = self._llm.chat_json([
            {"role": "system", "content": load_prompt("graphcheck_triple_system")},
            {"role": "user", "content": load_prompt("graphcheck_triple_user").format(claim=claim)},
        ])
        if not isinstance(result, dict):
            return None
        head = str(result.get("head", "")).strip()
        relation = str(result.get("relation", "")).strip()
        tail = str(result.get("tail", "")).strip()
        if not head or not relation or not tail:
            return None
        return head, relation, tail
