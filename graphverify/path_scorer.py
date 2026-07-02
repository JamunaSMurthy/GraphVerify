"""
Scores a candidate evidence path against a claim triple.

Score = λ_head · a_h  +  λ_rel · a_r  +  λ_tail · a_t  +  λ_prov · a_p

where a_h, a_t are entity agreement scores (exact / alias / embedding),
a_r is relation agreement, and a_p is provenance confidence weighted by
retriever rank. All agreement scores are in [0, 1].
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from .config import LAMBDA_HEAD, LAMBDA_REL, LAMBDA_TAIL, LAMBDA_PROV, EMBED_COSINE_CUTOFF


@dataclass
class ScoredPath:
    path_edges: List[Dict]
    score:      float
    head_score: float
    rel_score:  float
    tail_score: float
    prov_score: float
    head_match: str   # "exact" | "alias" | "embed" | "none"
    tail_match: str


class PathScorer:

    def __init__(
        self,
        lambda_head: float = LAMBDA_HEAD,
        lambda_rel:  float = LAMBDA_REL,
        lambda_tail: float = LAMBDA_TAIL,
        lambda_prov: float = LAMBDA_PROV,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        cosine_cutoff: float = EMBED_COSINE_CUTOFF,
        match_mode: str = "exact_alias_embed",
    ) -> None:
        self.lh = lambda_head
        self.lr = lambda_rel
        self.lt = lambda_tail
        self.lp = lambda_prov
        self._cutoff = cosine_cutoff
        self._embed_model = embed_model
        self._match_mode = match_mode
        self._embedder = None

    def score_path(
        self,
        path_edges: List[Dict],
        head_label: str,
        canonical_rel: str,
        tail_label: str,
    ) -> ScoredPath:
        if not path_edges:
            return ScoredPath([], 0.0, 0.0, 0.0, 0.0, 0.0, "none", "none")

        first_edge = path_edges[0]
        last_edge  = path_edges[-1]

        path_head = first_edge.get("src_label", first_edge.get("src", ""))
        a_h, h_match = self._entity_agreement(head_label, path_head)

        path_tail = last_edge.get("dst_label", last_edge.get("dst", ""))
        a_t, t_match = self._entity_agreement(tail_label, path_tail)

        path_rel = _dominant_relation(path_edges)
        a_r = self._relation_agreement(canonical_rel, path_rel)

        a_p = _prov_confidence(path_edges)

        score = self.lh * a_h + self.lr * a_r + self.lt * a_t + self.lp * a_p

        return ScoredPath(
            path_edges=path_edges,
            score=score,
            head_score=a_h,
            rel_score=a_r,
            tail_score=a_t,
            prov_score=a_p,
            head_match=h_match,
            tail_match=t_match,
        )

    def _entity_agreement(self, claim_entity: str, path_entity: str):
        if not claim_entity or not path_entity:
            return 0.0, "none"

        ce = claim_entity.lower().strip()
        pe = path_entity.lower().strip()

        if self._match_mode != "embed_only" and ce == pe:
            return 1.0, "exact"

        if self._match_mode in ("exact_alias", "exact_alias_embed"):
            ce_tok = set(ce.split())
            pe_tok = set(pe.split())
            if ce_tok and pe_tok:
                overlap = len(ce_tok & pe_tok) / max(len(ce_tok), len(pe_tok))
                if overlap >= 0.75:
                    return overlap, "alias"

        if self._match_mode in ("exact_alias_embed", "embed_only"):
            try:
                sim = self._embed_sim(claim_entity, path_entity)
                if sim >= self._cutoff:
                    return sim, "embed"
            except Exception:
                pass

        return 0.0, "none"

    def _relation_agreement(self, canon_rel: str, path_rel: str) -> float:
        if not canon_rel or not path_rel:
            return 0.0
        if self._match_mode != "embed_only" and canon_rel.lower() == path_rel.lower():
            return 1.0
        if self._match_mode in ("exact_alias_embed", "embed_only"):
            try:
                sim = self._embed_sim(canon_rel, path_rel)
                return sim if sim >= self._cutoff else sim * 0.5
            except Exception:
                return 0.0
        return 0.0

    def _embed_sim(self, a: str, b: str) -> float:
        from .embedder import Embedder
        if self._embedder is None:
            self._embedder = Embedder(self._embed_model)
        vecs = self._embedder.encode([a, b])
        return float(self._embedder.cosine_sim(vecs[0], vecs[1]))


def _dominant_relation(edges: List[Dict]) -> str:
    rels = [e.get("relation", "") for e in edges if e.get("relation")]
    return Counter(rels).most_common(1)[0][0] if rels else ""


def _prov_confidence(edges: List[Dict]) -> float:
    """
    Provenance confidence across a multi-hop path.
    Multiplied across edges so longer, lower-confidence chains score lower.
    Rank decay: rank 1 → 1.0, rank 10 → ~0.5.
    """
    conf = 1.0
    for e in edges:
        prov = e.get("provenance", {})
        edge_conf   = float(prov.get("confidence", 0.5)) if isinstance(prov, dict) else 0.5
        rank        = int(prov.get("retriever_rank", 1)) if isinstance(prov, dict) else 1
        rank_factor = 1.0 / (1.0 + 0.1 * (rank - 1))
        conf *= edge_conf * rank_factor
    return min(max(conf, 0.0), 1.0)
