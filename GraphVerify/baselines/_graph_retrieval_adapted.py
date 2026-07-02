"""
Shared scaffolding for the two "adapted graph retrieval" baselines
(GraphRAG, HippoRAG). Both are retrieval systems, not verifiers, so the
revision plan's fairness protocol requires evaluating them as: build the
same provenance-linked graph GraphVerify would build, use the *baseline's*
retrieval strategy to select a subgraph per claim, then hand that subgraph
to GraphVerify's own shared verdict head
(:class:`graphverify.path_searcher.PathSearcher` +
:class:`graphverify.verdict_assigner.VerdictAssigner`) — never a separately
implemented verdict rule, so any performance difference is attributable to
the retrieval strategy alone, not to a different scoring function.

Subclasses implement :meth:`_select_subgraph`; everything else (graph
construction, triple extraction/entity linking, invoking the shared verdict
head) lives here once.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from graphverify.entity_linker import EntityLinker
from graphverify.evidence_graph import EvidenceGraph, EvidenceGraphBuilder
from graphverify.path_scorer import PathScorer
from graphverify.path_searcher import PathSearcher
from graphverify.relation_normalizer import RelationNormalizer
from graphverify.triple_extractor import TripleExtractor
from graphverify.verdict_assigner import VerdictAssigner, VerificationRecord

from .base import BaselineVerifier


class GraphRetrievalAdaptedVerifier(BaselineVerifier):
    category = "adapted_graph_retrieval"
    uses_graph = True

    def __init__(
        self,
        llm_client: Any,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        embed_cosine_cutoff: float = 0.75,
        l_max: int = 3,
        top_k_paths: int = 20,
        support_threshold: float = 0.60,
        contradict_threshold: float = 0.55,
    ) -> None:
        super().__init__(llm_client)
        self._embed_model = embed_model
        self._embed_cosine_cutoff = embed_cosine_cutoff
        self._l_max = l_max
        self._top_k_paths = top_k_paths
        self._rel_norm = RelationNormalizer(embed_model=embed_model, cosine_cutoff=embed_cosine_cutoff)
        self._scorer = PathScorer(embed_model=embed_model, cosine_cutoff=embed_cosine_cutoff)
        self._verdict = VerdictAssigner(support_threshold=support_threshold, contradict_threshold=contradict_threshold)

    @abstractmethod
    def _select_subgraph(
        self,
        full_graph: EvidenceGraph,
        entity_linker_full: EntityLinker,
        head_label: str,
        tail_label: str,
    ) -> EvidenceGraph:
        """Returns the subgraph this baseline's retrieval strategy would surface for one claim."""
        raise NotImplementedError

    def verify_claims(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        claims: List[str],
    ) -> List[VerificationRecord]:
        full_graph = EvidenceGraphBuilder(
            llm_client=self._llm, relation_normalizer=self._rel_norm, embed_model=self._embed_model,
        ).build(query, passages)
        entity_linker_full = EntityLinker(
            full_graph.node_labels, embed_model=self._embed_model, cosine_cutoff=self._embed_cosine_cutoff,
        )
        triple_extractor = TripleExtractor(self._llm, entity_linker_full, self._rel_norm)

        records = []
        for claim in claims:
            records.append(self._verify_one(claim, full_graph, entity_linker_full, triple_extractor))
        return records

    def _verify_one(
        self,
        claim: str,
        full_graph: EvidenceGraph,
        entity_linker_full: EntityLinker,
        triple_extractor: TripleExtractor,
    ) -> VerificationRecord:
        triple = triple_extractor.extract(claim)
        if not triple.linked:
            record = self._verdict.assign(
                claim=claim, head=triple.raw_head or None, relation=triple.relation,
                tail=triple.raw_tail or None, support_paths=[], conflict_paths=[], triple_linked=False,
            )
            record.verdict_mode = self.name
            return record

        subgraph = self._select_subgraph(full_graph, entity_linker_full, triple.head, triple.tail)
        entity_linker_sub = EntityLinker(
            subgraph.node_labels, embed_model=self._embed_model, cosine_cutoff=self._embed_cosine_cutoff,
        )
        path_searcher = PathSearcher(
            graph=subgraph, entity_linker=entity_linker_sub, path_scorer=self._scorer,
            l_max=self._l_max, top_k=self._top_k_paths,
        )
        support_paths, conflict_paths = path_searcher.search(
            head=triple.head, relation=triple.relation, tail=triple.tail,
        )
        record = self._verdict.assign(
            claim=claim, head=triple.head, relation=triple.relation, tail=triple.tail,
            support_paths=support_paths, conflict_paths=conflict_paths, triple_linked=True,
        )
        record.verdict_mode = self.name
        return record
