"""
Extracts a canonical (head, relation, tail) triple from an atomic claim.

Steps:
  1. LLM extracts raw (head, relation, tail) from the claim text
  2. Entity linker maps head/tail to graph node labels
  3. Relation normalizer canonicalises the relation

If either entity fails to link, the triple is marked unlinked and the
verifier will assign Unsupported without a graph search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .entity_linker import EntityLinker
from .relation_normalizer import RelationNormalizer
from .prompts import load_prompt


@dataclass
class ClaimTriple:
    raw_head:     str
    raw_relation: str
    raw_tail:     str
    head:         Optional[str]
    relation:     str
    tail:         Optional[str]
    head_score:   float = 0.0
    rel_score:    float = 0.0
    tail_score:   float = 0.0
    linked:       bool = False


class TripleExtractor:

    def __init__(
        self,
        llm_client,
        entity_linker: EntityLinker,
        relation_normalizer: RelationNormalizer,
    ) -> None:
        self._llm = llm_client
        self._linker = entity_linker
        self._rel_norm = relation_normalizer

    def extract(self, claim: str) -> ClaimTriple:
        """
        Returns a ClaimTriple. If extraction or linking fails, returns an
        unlinked triple (claim will be assigned Unsupported by the verifier).
        """
        raw = self._llm_extract(claim)
        if raw is None:
            return ClaimTriple(
                raw_head="", raw_relation="", raw_tail="",
                head=None, relation="", tail=None, linked=False,
            )

        raw_h = str(raw.get("head", "")).strip()
        raw_r = str(raw.get("relation", "")).strip()
        raw_t = str(raw.get("tail", "")).strip()

        canon_r, r_score       = self._rel_norm.normalize(raw_r)
        head_label, h_score    = self._linker.link_text(raw_h)
        tail_label, t_score    = self._linker.link_text(raw_t)

        return ClaimTriple(
            raw_head=raw_h,
            raw_relation=raw_r,
            raw_tail=raw_t,
            head=head_label,
            relation=canon_r,
            tail=tail_label,
            head_score=h_score,
            rel_score=r_score,
            tail_score=t_score,
            linked=(head_label is not None) and (tail_label is not None),
        )

    def _llm_extract(self, claim: str) -> Optional[dict]:
        messages = [
            {"role": "system", "content": load_prompt("triple_extraction_system")},
            {"role": "user",   "content": load_prompt("triple_extraction_user").format(claim=claim)},
        ]
        result = self._llm.chat_json(messages)
        if isinstance(result, dict) and all(k in result for k in ("head", "relation", "tail")):
            return result
        return None
