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

from dataclasses import dataclass
from typing import List, Optional

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
    path_type:        str   = ""   # "support" | "conflict" | "none"
    triple_linked:    bool  = False


class VerdictAssigner:

    def __init__(
        self,
        support_threshold:    float = SUPPORT_THRESHOLD,
        contradict_threshold: float = CONTRADICT_THRESHOLD,
    ) -> None:
        self._theta_s = support_threshold
        self._theta_c = contradict_threshold

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

        # Contradiction is evaluated before support (see module docstring)
        if s_minus >= self._theta_c:
            verdict   = VERDICT_CONTRADICTORY
            best_path = best_conflict.path_edges if best_conflict else None
            path_type = "conflict"
        elif s_plus >= self._theta_s:
            verdict   = VERDICT_SUPPORTED
            best_path = best_support.path_edges if best_support else None
            path_type = "support"
        else:
            verdict   = VERDICT_UNSUPPORTED
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
