"""
Main GraphVerify pipeline.

Given a query, a list of retrieved passages, and a generated answer,
GraphVerify returns a set of claim-level verification records — one per
atomic claim in the answer.

Each record contains:
  - claim text
  - extracted (head, relation, tail) triple
  - verdict: Supported | Unsupported | Contradictory
  - best evidence path (support or conflict)
  - calibrated reliability score
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from .config import GraphVerifyConfig
from .llm_client import LLMClient
from .claim_decomposer import ClaimDecomposer
from .triple_extractor import TripleExtractor
from .entity_linker import EntityLinker
from .relation_normalizer import RelationNormalizer
from .evidence_graph import EvidenceGraph, EvidenceGraphBuilder
from .path_searcher import PathSearcher
from .path_scorer import PathScorer
from .verdict_assigner import VerdictAssigner, VerificationRecord
from .calibrator import TemperatureCalibrator


@dataclass
class VerificationOutput:
    query:       str
    answer:      str
    records:     List[Dict]
    graph_stats: Dict

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @property
    def verdicts(self) -> List[str]:
        return [r["verdict"] for r in self.records]

    @property
    def n_supported(self) -> int:
        return sum(1 for v in self.verdicts if v == "Supported")

    @property
    def n_unsupported(self) -> int:
        return sum(1 for v in self.verdicts if v == "Unsupported")

    @property
    def n_contradictory(self) -> int:
        return sum(1 for v in self.verdicts if v == "Contradictory")


class GraphVerify:
    """
    Post-generation claim-level verifier.

    Usage
    -----
    >>> cfg = GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini")
    >>> gv  = GraphVerify(cfg)
    >>> out = gv.verify(query="...", passages=[...], answer="...")
    >>> for rec in out.records:
    ...     print(rec["claim"], "→", rec["verdict"])
    """

    def __init__(self, cfg: Optional[GraphVerifyConfig] = None) -> None:
        self.cfg = cfg or GraphVerifyConfig()

        self._llm      = LLMClient(self.cfg)
        self._rel_norm = RelationNormalizer(
            embed_model=self.cfg.embed_model,
            cosine_cutoff=self.cfg.embed_cosine_cutoff,
        )
        self._decomposer = ClaimDecomposer(self._llm)
        self._scorer     = PathScorer(
            lambda_head=self.cfg.lambda_head,
            lambda_rel=self.cfg.lambda_rel,
            lambda_tail=self.cfg.lambda_tail,
            lambda_prov=self.cfg.lambda_prov,
            embed_model=self.cfg.embed_model,
            cosine_cutoff=self.cfg.embed_cosine_cutoff,
        )
        self._verdict    = VerdictAssigner(
            support_threshold=self.cfg.support_threshold,
            contradict_threshold=self.cfg.contradict_threshold,
        )
        self._calibrator: Optional[TemperatureCalibrator] = None

    def verify(
        self,
        query:    str,
        passages: List[Dict[str, Any]],
        answer:   str,
        graph:    Optional[EvidenceGraph] = None,
    ) -> VerificationOutput:
        """
        Verify all claims in an answer against a provenance-linked evidence graph.

        Parameters
        ----------
        query    : user query string
        passages : list of passage dicts — each needs text, id, rank, score
        answer   : the generated answer to verify
        graph    : pre-built EvidenceGraph (skip building if provided)
        """
        if graph is None:
            graph = self._build_graph(query, passages)

        claims = self._decomposer.decompose(answer)

        node_labels  = graph.node_labels
        entity_linker = EntityLinker(
            node_labels,
            embed_model=self.cfg.embed_model,
            cosine_cutoff=self.cfg.embed_cosine_cutoff,
        )
        triple_extractor = TripleExtractor(self._llm, entity_linker, self._rel_norm)
        path_searcher    = PathSearcher(
            graph=graph,
            entity_linker=entity_linker,
            path_scorer=self._scorer,
            l_max=self.cfg.l_max,
            top_k=self.cfg.top_k_paths,
        )

        records: List[VerificationRecord] = []
        for claim in claims:
            rec = self._verify_claim(claim, triple_extractor, path_searcher)
            records.append(rec)

        if self._calibrator is not None:
            for rec in records:
                rec.reliability = self._calibrator.calibrate(rec.reliability)

        return VerificationOutput(
            query=query,
            answer=answer,
            records=[self._record_to_dict(r) for r in records],
            graph_stats={
                "n_nodes":  len(graph),
                "n_edges":  graph.nx_graph().number_of_edges(),
                "n_claims": len(claims),
            },
        )

    def build_graph(self, query: str, passages: List[Dict[str, Any]]) -> EvidenceGraph:
        return self._build_graph(query, passages)

    def load_calibrator(self, path: str) -> None:
        self._calibrator = TemperatureCalibrator()
        self._calibrator.load(path)

    def _build_graph(self, query: str, passages: List[Dict[str, Any]]) -> EvidenceGraph:
        return EvidenceGraphBuilder(
            llm_client=self._llm,
            relation_normalizer=self._rel_norm,
            embed_model=self.cfg.embed_model,
        ).build(query, passages)

    def _verify_claim(
        self,
        claim: str,
        triple_extractor: TripleExtractor,
        path_searcher: PathSearcher,
    ) -> VerificationRecord:
        triple = triple_extractor.extract(claim)

        if not triple.linked:
            return self._verdict.assign(
                claim=claim,
                head=triple.raw_head or None,
                relation=triple.relation,
                tail=triple.raw_tail or None,
                support_paths=[],
                conflict_paths=[],
                triple_linked=False,
            )

        support_paths, conflict_paths = path_searcher.search(
            head=triple.head,
            relation=triple.relation,
            tail=triple.tail,
        )

        return self._verdict.assign(
            claim=claim,
            head=triple.head,
            relation=triple.relation,
            tail=triple.tail,
            support_paths=support_paths,
            conflict_paths=conflict_paths,
            triple_linked=triple.linked,
        )

    @staticmethod
    def _record_to_dict(rec: VerificationRecord) -> Dict:
        d = asdict(rec)
        if d["best_path"] and isinstance(d["best_path"], list):
            d["best_path"] = [
                {k: v for k, v in e.items() if k != "provenance"}
                for e in d["best_path"] if isinstance(e, dict)
            ]
        return d
