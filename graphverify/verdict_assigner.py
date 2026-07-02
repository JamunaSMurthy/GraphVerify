"""
Assigns a three-way verdict given scored support and conflict paths.

  s_plus  = max score across support paths
  s_minus = max score across conflict paths

  Verdict:
    Contradictory  if s_minus >= threshold_c
    Supported      if s_plus  >= threshold_s
    Unsupported    otherwise

Contradiction is checked before support because a claim can be correctly
linked with the right entity and relation but assert the wrong value — in
that case both a support path and a conflict path may exist, and the
conflict should win.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .config import SUPPORT_THRESHOLD, CONTRADICT_THRESHOLD
from .path_scorer import ScoredPath


VERDICT_SUPPORTED     = "Supported"
VERDICT_UNSUPPORTED   = "Unsupported"
VERDICT_CONTRADICTORY = "Contradictory"


@dataclass
class VerificationRecord:
    claim:            str
    head:             Optional[str]
    relation:         str
    tail:             Optional[str]
    verdict:          str
    best_path:        Optional[List]
    reliability:      float
    support_score:    float = 0.0
    contradict_score: float = 0.0
    path_type:        str   = ""   # "support" | "conflict" | "text" | "text_fallback" | "none"
    triple_linked:    bool  = False
    verdict_mode:     str   = "score_only"  # "score_only" | "hybrid_llm"
    rationale:        str   = ""   # free-text explanation, populated by text-fallback / hybrid verdict steps


class VerdictAssigner:

    def __init__(
        self,
        support_threshold:    float = SUPPORT_THRESHOLD,
        contradict_threshold: float = CONTRADICT_THRESHOLD,
    ) -> None:
        self._theta_s = support_threshold
        self._theta_c = contradict_threshold

    def verdict_from_scores(self, s_plus: float, s_minus: float) -> str:
        """
        Applies just the threshold rule (contradiction-before-support, see
        module docstring) to already-computed support/conflict scores,
        without needing `ScoredPath` objects. `assign()` calls this
        internally; it is also public so callers that cache raw
        `support_score`/`contradict_score` per claim (e.g.
        `experiments/run_label_efficiency_experiment.py` and
        `experiments/run_threshold_sensitivity_sweep.py`) can re-derive
        verdicts under many candidate threshold pairs without re-running
        path search or any LLM call for each candidate.
        """
        if s_minus >= self._theta_c:
            return VERDICT_CONTRADICTORY
        if s_plus >= self._theta_s:
            return VERDICT_SUPPORTED
        return VERDICT_UNSUPPORTED

    def assign(
        self,
        claim: str,
        head:  Optional[str],
        relation: str,
        tail:  Optional[str],
        support_paths:  List[ScoredPath],
        conflict_paths: List[ScoredPath],
        triple_linked:  bool = False,
    ) -> VerificationRecord:
        if not triple_linked or (not head and not tail):
            return VerificationRecord(
                claim=claim, head=head, relation=relation, tail=tail,
                verdict=VERDICT_UNSUPPORTED,
                best_path=None, reliability=0.0,
                path_type="none", triple_linked=triple_linked,
            )

        s_plus  = max((p.score for p in support_paths),  default=0.0)
        s_minus = max((p.score for p in conflict_paths), default=0.0)

        best_support  = support_paths[0]  if support_paths  else None
        best_conflict = conflict_paths[0] if conflict_paths else None

        verdict = self.verdict_from_scores(s_plus, s_minus)
        if verdict == VERDICT_CONTRADICTORY:
            best_path = best_conflict.path_edges if best_conflict else None
            path_type = "conflict"
        elif verdict == VERDICT_SUPPORTED:
            best_path = best_support.path_edges if best_support else None
            path_type = "support"
        else:
            best_path = None
            path_type = "none"

        return VerificationRecord(
            claim=claim,
            head=head,
            relation=relation,
            tail=tail,
            verdict=verdict,
            best_path=best_path,
            reliability=max(s_plus, s_minus),
            support_score=s_plus,
            contradict_score=s_minus,
            path_type=path_type,
            triple_linked=triple_linked,
        )


def record_to_dict(rec: VerificationRecord) -> Dict:
    """
    Serializes a VerificationRecord to a plain dict, stripping the bulky
    per-edge provenance blob from the returned `best_path` (kept in the
    evidence graph / prediction cache, not duplicated in every record).
    Shared by :class:`graphverify.verifier.GraphVerify` and every baseline
    in `baselines/` so predictions from any method serialize identically.

    `best_path` holds two different shapes depending on which method
    produced the record: a list of graph-edge dicts for GraphVerify and the
    graph-based baselines (GraphRAG-adapted, HippoRAG-adapted), or a list of
    plain evidence-passage-id strings for text-based methods that report
    "evidence spans" instead of a graph path (e.g. FIRE, CiteFix). Edge
    dicts are stripped of provenance; strings are passed through unchanged.
    """
    d = asdict(rec)
    if d["best_path"] and isinstance(d["best_path"], list):
        d["best_path"] = [
            ({k: v for k, v in e.items() if k != "provenance"} if isinstance(e, dict) else e)
            for e in d["best_path"]
        ]
    return d
