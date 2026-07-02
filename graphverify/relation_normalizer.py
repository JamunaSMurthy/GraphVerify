"""
Maps surface relation strings to canonical relation names.

Matching proceeds in three stages:
  1. Exact / alias lookup (case-insensitive dictionary)
  2. Partial substring match with token overlap scoring
  3. Embedding cosine similarity fallback

Returns the canonical name and an agreement score in [0, 1].

`disabled=True` (wired from ``GraphVerifyConfig.disable_relation_normalization``,
used by the "w/o relation normalization" component-ablation variant in
``eval/ablation.py``) turns `normalize()` into an identity function: the raw
surface string is returned unchanged as if it were already canonical. This
is a real behavioral change, not a cosmetic flag — downstream, canonical-
relation-keyed lookups (`FUNCTIONAL_RELATIONS`, `TEMPORAL_RELATIONS` in
`graphverify/config.py`, consumed by `graphverify/incompatibility.py`) will
then almost never match a raw surface string, so functional/temporal
contradiction detection degrades exactly as the ablation intends.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

from .config import RELATION_ALIASES, EMBED_COSINE_CUTOFF


class RelationNormalizer:

    def __init__(self, embed_model: str = "BAAI/bge-base-en-v1.5",
                 cosine_cutoff: float = EMBED_COSINE_CUTOFF,
                 disabled: bool = False) -> None:
        self._cutoff = cosine_cutoff
        self._disabled = disabled
        self._alias_map: Dict[str, str] = {}
        for canonical, aliases in RELATION_ALIASES.items():
            self._alias_map[canonical.lower()] = canonical
            for alias in aliases:
                self._alias_map[alias.lower().strip()] = canonical
        self._canonical_list = list(RELATION_ALIASES.keys())
        self._embedder = None
        self._canonical_vecs = None
        self._embed_model_name = embed_model

    def normalize(self, surface: str) -> Tuple[str, float]:
        """
        Returns (canonical_relation, agreement_score).
        Returns (surface_unchanged, 0.0) when no match is found.
        Returns (surface_unchanged, 1.0) unconditionally if this normalizer
        was constructed with `disabled=True`.
        """
        if not surface or not surface.strip():
            return surface, 0.0

        if self._disabled:
            return surface.strip(), 1.0

        cleaned = self._clean(surface)

        if cleaned in self._alias_map:
            return self._alias_map[cleaned], 1.0

        for alias, canonical in self._alias_map.items():
            if alias in cleaned or cleaned in alias:
                score = len(alias) / max(len(cleaned), len(alias))
                if score >= 0.7:
                    return canonical, score

        try:
            canon, sim = self._embed_match(cleaned)
            if sim >= self._cutoff:
                return canon, sim
        except Exception:
            pass

        return surface, 0.0

    def _clean(self, s: str) -> str:
        return re.sub(r"[_/\\]", " ", s).lower().strip()

    def _embed_match(self, surface: str) -> Tuple[str, float]:
        from .embedder import Embedder
        import numpy as np

        if self._embedder is None:
            self._embedder = Embedder(self._embed_model_name)
            self._canonical_vecs = self._embedder.encode(
                [self._clean(c) for c in self._canonical_list]
            )

        q_vec = self._embedder.encode([surface])
        sims = self._embedder.cosine_sim_matrix(q_vec, self._canonical_vecs)[0]
        best_idx = int(np.argmax(sims))
        return self._canonical_list[best_idx], float(sims[best_idx])
