"""
Links a surface entity mention to a canonical node in the evidence graph.

Matching strategy:
  1. Exact string match (case-insensitive, unicode-normalised)
  2. Token-overlap partial match (threshold 0.6)
  3. Embedding cosine similarity fallback (threshold configurable)

Returns (matched_node_index, agreement_score) or (None, 0.0).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import EMBED_COSINE_CUTOFF


def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


class EntityLinker:

    def __init__(
        self,
        node_labels: List[str],
        node_aliases: Optional[Dict[int, List[str]]] = None,
        embed_model: str = "BAAI/bge-base-en-v1.5",
        cosine_cutoff: float = EMBED_COSINE_CUTOFF,
    ) -> None:
        self._labels = node_labels
        self._aliases = node_aliases or {}
        self._cutoff = cosine_cutoff
        self._embed_model = embed_model
        self._embedder = None
        self._node_vecs = None

        self._exact_map: Dict[str, int] = {}
        for idx, label in enumerate(node_labels):
            key = _normalize_text(label)
            if key and key not in self._exact_map:
                self._exact_map[key] = idx
            for alias in self._aliases.get(idx, []):
                ak = _normalize_text(alias)
                if ak and ak not in self._exact_map:
                    self._exact_map[ak] = idx

    def link(self, mention: str) -> Tuple[Optional[int], float]:
        """Returns (node_index, score) or (None, 0.0) if not matched."""
        if not mention or not mention.strip():
            return None, 0.0

        key = _normalize_text(mention)

        if key in self._exact_map:
            return self._exact_map[key], 1.0

        best_idx, best_score = None, 0.0
        key_tokens: Set[str] = set(key.split())
        for norm_label, idx in self._exact_map.items():
            tokens: Set[str] = set(norm_label.split())
            if not tokens:
                continue
            overlap = len(key_tokens & tokens) / max(len(key_tokens), len(tokens))
            if overlap > best_score and overlap >= 0.6:
                best_score = overlap
                best_idx = idx

        if best_idx is not None:
            return best_idx, best_score

        if not self._labels:
            return None, 0.0
        try:
            return self._embed_link(mention)
        except Exception:
            return None, 0.0

    def link_text(self, mention: str) -> Tuple[Optional[str], float]:
        """Returns (canonical_label, score) instead of (index, score)."""
        idx, score = self.link(mention)
        if idx is not None:
            return self._labels[idx], score
        return None, 0.0

    def _embed_link(self, mention: str) -> Tuple[Optional[int], float]:
        from .embedder import Embedder
        if self._embedder is None:
            self._embedder = Embedder(self._embed_model)
            self._node_vecs = self._embedder.encode(self._labels)

        q_vec = self._embedder.encode([mention])
        sims = self._embedder.cosine_sim_matrix(q_vec, self._node_vecs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self._cutoff:
            return best_idx, best_sim
        return None, 0.0
