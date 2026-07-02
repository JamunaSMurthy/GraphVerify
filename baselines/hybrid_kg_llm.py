"""
Hybrid KG-LLM fact-checking (Rosenbaum et al., 2025, "Hybrid fact-checking
that integrates knowledge graphs, large language models, and search-based
retrieval agents improves interpretable claim verification").

Original method: combines structured knowledge-graph signals with LLM
judgment and search-based retrieval agents for interpretable claim
verification.

**Reimplementation note.** This baseline is deliberately *simpler* than
GraphVerify-Hybrid (`graphverify/hybrid_verdict.py`): it gives the LLM a
compact structural summary — whether the claim's head/tail entities linked
to the evidence graph at all, their node degree, and the surface relations
observed directly between them (one hop only) — with no multi-hop scored
paths and no conflict-predicate check
(:func:`graphverify.incompatibility.classify_incompatibility` is not used
here). The LLM alone decides the verdict from that summary plus the claim
text. This keeps GraphVerify-Hybrid and Hybrid-KG-LLM mechanistically
distinct: the former lets an LLM interpret rule-scored, provenance-weighted
paths; this one lets an LLM interpret raw graph connectivity statistics
with no scoring or conflict logic layered on top.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraph, EvidenceGraphBuilder
from graphverify.prompts import load_prompt
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.triple_extractor import TripleExtractor
from graphverify.verdict_assigner import VerificationRecord

from .base import BaselineVerifier
from ._response_parsing import parse_verdict_response


class HybridKGLLMVerifier(BaselineVerifier):
    name = "hybrid_kg_llm"
    citation = (
        "Rosenbaum et al. 2025, 'Hybrid fact-checking that integrates knowledge graphs, "
        "large language models, and search-based retrieval agents improves interpretable "
        "claim verification'"
    )
    category = "kg_fact_checking"
    uses_graph = True

    def __init__(self, llm_client: Any, embed_model: str = "BAAI/bge-base-en-v1.5") -> None:
        super().__init__(llm_client)
        self._embed_model = embed_model
        self._rel_norm = RelationNormalizer(embed_model=embed_model)

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        graph = EvidenceGraphBuilder(
            llm_client=self._llm, relation_normalizer=self._rel_norm, embed_model=self._embed_model,
        ).build(query, passages)
        entity_linker = EntityLinker(graph.node_labels, embed_model=self._embed_model)
        triple_extractor = TripleExtractor(self._llm, entity_linker, self._rel_norm)

        records = []
        for claim in claims:
            records.append(self._verify_one(claim, graph, entity_linker, triple_extractor))
        return records

    def _verify_one(self, claim, graph: EvidenceGraph, entity_linker: EntityLinker, triple_extractor: TripleExtractor) -> VerificationRecord:
        triple = triple_extractor.extract(claim)
        summary = self._structural_summary(graph, entity_linker, triple.head, triple.tail)

        result = self._llm.chat_json([
            {"role": "system", "content": load_prompt("hybrid_kg_llm_system")},
            {"role": "user", "content": load_prompt("hybrid_kg_llm_user").format(
                claim=claim, head=triple.head or "?", relation=triple.relation or "?", tail=triple.tail or "?",
                head_linked=summary["head_linked"], head_match_type=summary["head_match_type"],
                tail_linked=summary["tail_linked"], tail_match_type=summary["tail_match_type"],
                head_degree=summary["head_degree"], tail_degree=summary["tail_degree"],
                direct_relations=summary["direct_relations"] or "(none)",
            )},
        ])
        verdict, confidence, rationale = parse_verdict_response(result)

        return VerificationRecord(
            claim=claim, head=triple.head, relation=triple.relation, tail=triple.tail,
            verdict=verdict, best_path=None, reliability=confidence,
            support_score=confidence if verdict == "Supported" else 0.0,
            contradict_score=confidence if verdict == "Contradictory" else 0.0,
            path_type="kg_structural_summary", triple_linked=triple.linked,
            verdict_mode=self.name, rationale=rationale,
        )

    @staticmethod
    def _structural_summary(graph: EvidenceGraph, entity_linker: EntityLinker, head: Optional[str], tail: Optional[str]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "head_linked": False, "head_match_type": "none", "head_degree": 0,
            "tail_linked": False, "tail_match_type": "none", "tail_degree": 0,
            "direct_relations": "",
        }
        head_id = tail_id = None
        if head:
            idx, score = entity_linker.link(head)
            if idx is not None:
                head_id = graph.node_ids[idx]
                summary["head_linked"] = True
                summary["head_match_type"] = "exact" if score >= 0.999 else "fuzzy"
                summary["head_degree"] = len(graph.neighbors(head_id))
        if tail:
            idx, score = entity_linker.link(tail)
            if idx is not None:
                tail_id = graph.node_ids[idx]
                summary["tail_linked"] = True
                summary["tail_match_type"] = "exact" if score >= 0.999 else "fuzzy"
                summary["tail_degree"] = len(graph.neighbors(tail_id))

        if head_id and tail_id:
            edges = graph.get_edges(head_id, tail_id)
            relations = sorted({e.get("relation", "") for e in edges if e.get("relation")})
            summary["direct_relations"] = ", ".join(relations)

        return summary
