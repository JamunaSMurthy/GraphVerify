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

Two public classes cover the two ``verdict_mode`` values in
:class:`graphverify.config.GraphVerifyConfig`:

  - :class:`GraphVerify` — GraphVerify-score (``verdict_mode="score_only"``).
  - :class:`HybridGraphVerify` — GraphVerify-hybrid
    (``verdict_mode="hybrid_llm"``), a thin subclass that adds the LLM
    verdict head from :mod:`graphverify.hybrid_verdict` on top of the same
    rule-based pipeline.

Use :func:`build_graphverify` to construct the correct class from
``cfg.verdict_mode`` without an explicit if/else in calling code.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, List, Optional

from .config import GraphVerifyConfig
from .llm_client import LLMClient
from .claim_decomposer import ClaimDecomposer
from .triple_extractor import TripleExtractor
from .entity_linker import EntityLinker
from .relation_normalizer import RelationNormalizer
from .evidence_graph import EvidenceGraph, EvidenceGraphBuilder, merge_external_kg
from .path_searcher import PathSearcher
from .path_scorer import PathScorer, ScoredPath
from .verdict_assigner import VerdictAssigner, VerificationRecord, VERDICT_UNSUPPORTED, record_to_dict
from .calibrator import TemperatureCalibrator
from .text_evidence import text_entailment_verdict
from .hybrid_verdict import HybridVerdictHead


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
    Post-generation claim-level verifier — GraphVerify-score
    (``cfg.verdict_mode == "score_only"``).

    Usage
    -----
    >>> cfg = GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini")
    >>> gv  = GraphVerify(cfg)
    >>> out = gv.verify(query="...", passages=[...], answer="...")
    >>> for rec in out.records:
    ...     print(rec["claim"], "→", rec["verdict"])
    """

    def __init__(
        self,
        cfg: Optional[GraphVerifyConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """
        `llm_client` is an injection point for tests and for callers that
        want to share one LLM client across several verifiers/baselines
        (e.g. `experiments/run_main_verification_benchmark.py`, which
        constructs one client per method and reuses it). When omitted, a
        real `LLMClient` is built from `cfg` (requires live credentials for
        the "openai"/"anthropic" backends).
        """
        self.cfg = cfg or GraphVerifyConfig()

        self._llm      = llm_client or LLMClient(self.cfg)
        self._rel_norm = RelationNormalizer(
            embed_model=self.cfg.embed_model,
            cosine_cutoff=self.cfg.embed_cosine_cutoff,
            disabled=self.cfg.disable_relation_normalization,
        )
        self._decomposer = ClaimDecomposer(self._llm)
        self._scorer     = PathScorer(
            lambda_head=self.cfg.lambda_head,
            lambda_rel=self.cfg.lambda_rel,
            lambda_tail=self.cfg.lambda_tail,
            lambda_prov=self.cfg.lambda_prov,
            embed_model=self.cfg.embed_model,
            cosine_cutoff=self.cfg.embed_cosine_cutoff,
            match_mode=self.cfg.entity_match_mode,
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
        claims:   Optional[List[str]] = None,
    ) -> VerificationOutput:
        """
        Verify all claims in an answer against a provenance-linked evidence graph.

        Parameters
        ----------
        query    : user query string
        passages : list of passage dicts — each needs text, id, rank, score
        answer   : the generated answer to verify
        graph    : pre-built EvidenceGraph (skip building if provided). Also
                   the mechanism for ``evidence_source="gold_oracle"``: pass
                   a gold/oracle graph here to bypass extraction entirely.
        claims   : pre-decomposed atomic claims, bypassing this instance's
                   own :class:`~graphverify.claim_decomposer.ClaimDecomposer`
                   call. Experiment scripts comparing GraphVerify against
                   baselines (see ``baselines/base.py``'s fairness protocol)
                   should always pass the same shared claim list here that
                   every baseline receives, so a difference in results is
                   never explained by two methods decomposing the same
                   answer differently. When omitted, `answer` is decomposed
                   internally as before (unchanged default behavior).
        """
        if claims is not None:
            pass
        elif self.cfg.disable_claim_decomposition:
            claims = [answer] if answer and answer.strip() else []
        else:
            claims = self._decomposer.decompose(answer)

        if self.cfg.evidence_mode == "text":
            records = [self._text_only_record(claim, passages) for claim in claims]
            graph_stats = {"n_nodes": 0, "n_edges": 0, "n_claims": len(claims), "evidence_mode": "text"}
        else:
            if graph is None:
                graph = self._build_graph(query, passages)

            node_labels   = graph.node_labels
            entity_linker = EntityLinker(
                node_labels,
                embed_model=self.cfg.embed_model,
                cosine_cutoff=self.cfg.embed_cosine_cutoff,
                match_mode=self.cfg.entity_match_mode,
            )
            triple_extractor = TripleExtractor(self._llm, entity_linker, self._rel_norm)
            path_searcher    = PathSearcher(
                graph=graph,
                entity_linker=entity_linker,
                path_scorer=self._scorer,
                l_max=self.cfg.l_max,
                top_k=self.cfg.top_k_paths,
            )

            records = [
                self._verify_claim(claim, triple_extractor, path_searcher, passages)
                for claim in claims
            ]
            graph_stats = {
                "n_nodes":  len(graph),
                "n_edges":  graph.nx_graph().number_of_edges(),
                "n_claims": len(claims),
                "evidence_mode": self.cfg.evidence_mode,
            }

        if self._calibrator is not None:
            for i, rec in enumerate(records):
                records[i] = replace(rec, reliability=self._calibrator.calibrate(rec.reliability))

        return VerificationOutput(
            query=query,
            answer=answer,
            records=[self._record_to_dict(r) for r in records],
            graph_stats=graph_stats,
        )

    def build_graph(self, query: str, passages: List[Dict[str, Any]]) -> EvidenceGraph:
        return self._build_graph(query, passages)

    def load_calibrator(self, path: str) -> None:
        self._calibrator = TemperatureCalibrator()
        self._calibrator.load(path)

    def _build_graph(self, query: str, passages: List[Dict[str, Any]]) -> EvidenceGraph:
        graph = EvidenceGraphBuilder(
            llm_client=self._llm,
            relation_normalizer=self._rel_norm,
            embed_model=self.cfg.embed_model,
        ).build(query, passages)

        if self.cfg.evidence_mode in ("kg_paths", "hybrid"):
            if self.cfg.external_kg_path:
                merge_external_kg(graph, self.cfg.external_kg_path)
            elif self.cfg.evidence_mode == "kg_paths":
                warnings.warn(
                    "evidence_mode='kg_paths' requested but cfg.external_kg_path is unset; "
                    "falling back to retrieved-only graph evidence.",
                    stacklevel=2,
                )

        return graph

    def _verify_claim(
        self,
        claim: str,
        triple_extractor: TripleExtractor,
        path_searcher: PathSearcher,
        passages: List[Dict[str, Any]],
    ) -> VerificationRecord:
        triple = triple_extractor.extract(claim)

        support_paths: List[ScoredPath] = []
        conflict_paths: List[ScoredPath] = []

        if not triple.linked:
            record = self._verdict.assign(
                claim=claim,
                head=triple.raw_head or None,
                relation=triple.relation,
                tail=triple.raw_tail or None,
                support_paths=[],
                conflict_paths=[],
                triple_linked=False,
            )
        else:
            support_paths, conflict_paths = path_searcher.search(
                head=triple.head,
                relation=triple.relation,
                tail=triple.tail,
            )
            record = self._verdict.assign(
                claim=claim,
                head=triple.head,
                relation=triple.relation,
                tail=triple.tail,
                support_paths=support_paths,
                conflict_paths=conflict_paths,
                triple_linked=triple.linked,
            )

        if self.cfg.evidence_mode == "hybrid" and record.verdict == VERDICT_UNSUPPORTED:
            record = self._apply_text_fallback(record, claim, passages)

        return self._postprocess_record(
            record, claim, triple.head, triple.relation, triple.tail,
            support_paths, conflict_paths,
        )

    def _text_only_record(self, claim: str, passages: List[Dict[str, Any]]) -> VerificationRecord:
        result = text_entailment_verdict(self._llm, claim, passages)
        return VerificationRecord(
            claim=claim, head=None, relation="", tail=None,
            verdict=result.verdict, best_path=None, reliability=result.confidence,
            support_score=result.confidence if result.verdict == "Supported" else 0.0,
            contradict_score=result.confidence if result.verdict == "Contradictory" else 0.0,
            path_type="text", triple_linked=False,
            verdict_mode="score_only", rationale=result.rationale,
        )

    def _apply_text_fallback(
        self, record: VerificationRecord, claim: str, passages: List[Dict[str, Any]],
    ) -> VerificationRecord:
        """
        Applied only to claims the graph pipeline left Unsupported when
        ``evidence_mode="hybrid"``. Implements the "textual fallback
        matching" the method description names as an extension for claims
        that cannot be mapped to a valid graph triple (or whose graph paths
        never cleared threshold). Only overrides when the text check is
        itself confident (``>= cfg.text_fallback_threshold``); otherwise the
        original Unsupported rule-based record is kept unchanged.
        """
        result = text_entailment_verdict(self._llm, claim, passages)
        if result.verdict == VERDICT_UNSUPPORTED or result.confidence < self.cfg.text_fallback_threshold:
            return record
        return replace(
            record,
            verdict=result.verdict,
            reliability=result.confidence,
            support_score=result.confidence if result.verdict == "Supported" else record.support_score,
            contradict_score=result.confidence if result.verdict == "Contradictory" else record.contradict_score,
            path_type="text_fallback",
            rationale=result.rationale,
        )

    def _postprocess_record(
        self,
        record: VerificationRecord,
        claim: str,
        head: Optional[str],
        relation: str,
        tail: Optional[str],
        support_paths: List[ScoredPath],
        conflict_paths: List[ScoredPath],
    ) -> VerificationRecord:
        """
        Hook applied to every claim's rule-based record before it is
        returned. No-op in the score-only pipeline beyond stamping
        ``verdict_mode``; :class:`HybridGraphVerify` overrides this to run
        the LLM verdict head.
        """
        return replace(record, verdict_mode="score_only")

    @staticmethod
    def _record_to_dict(rec: VerificationRecord) -> Dict:
        return record_to_dict(rec)


class HybridGraphVerify(GraphVerify):
    """
    GraphVerify-hybrid — the same rule-based pipeline as :class:`GraphVerify`
    (unchanged, auditable backbone), plus an LLM verdict head
    (:class:`graphverify.hybrid_verdict.HybridVerdictHead`) that reads the
    claim, canonical triple, and top scored support/conflict paths, and
    confirms or overrides the rule-based prior verdict. See
    :mod:`graphverify.hybrid_verdict` for the exact contract.

    Records produced via the text-evidence path (``path_type in
    {"text", "text_fallback"}``) are LLM-decided verdicts already and are
    passed through unchanged — the hybrid verdict head's job is to interpret
    *graph* paths, not to re-judge a text-entailment decision that has no
    paths to show it.
    """

    def __init__(
        self,
        cfg: Optional[GraphVerifyConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        super().__init__(cfg, llm_client=llm_client)
        self._hybrid_head = HybridVerdictHead(self._llm)

    def _postprocess_record(
        self,
        record: VerificationRecord,
        claim: str,
        head: Optional[str],
        relation: str,
        tail: Optional[str],
        support_paths: List[ScoredPath],
        conflict_paths: List[ScoredPath],
    ) -> VerificationRecord:
        if record.path_type in ("text", "text_fallback") or not record.triple_linked:
            return replace(record, verdict_mode="hybrid_llm")

        decision = self._hybrid_head.decide(
            claim=claim, head=head, relation=relation, tail=tail,
            rule_based=record, support_paths=support_paths, conflict_paths=conflict_paths,
        )

        best_path = record.best_path
        path_type = record.path_type
        if decision.overrode_rule_based:
            if decision.verdict == "Supported" and support_paths:
                best_path, path_type = support_paths[0].path_edges, "support"
            elif decision.verdict == "Contradictory" and conflict_paths:
                best_path, path_type = conflict_paths[0].path_edges, "conflict"
            elif decision.verdict == "Unsupported":
                best_path, path_type = None, "none"

        return replace(
            record,
            verdict=decision.verdict,
            reliability=decision.confidence,
            best_path=best_path,
            path_type=path_type,
            verdict_mode="hybrid_llm",
            rationale=decision.rationale,
        )


def build_graphverify(
    cfg: Optional[GraphVerifyConfig] = None,
    llm_client: Optional[LLMClient] = None,
) -> GraphVerify:
    """
    Constructs :class:`GraphVerify` or :class:`HybridGraphVerify` from
    ``cfg.verdict_mode`` ("score_only" -> GraphVerify, "hybrid_llm" ->
    HybridGraphVerify). Experiment scripts that sweep ``verdict_mode``
    should call this instead of instantiating a class directly.
    """
    cfg = cfg or GraphVerifyConfig()
    if cfg.verdict_mode == "hybrid_llm":
        return HybridGraphVerify(cfg, llm_client=llm_client)
    return GraphVerify(cfg, llm_client=llm_client)
