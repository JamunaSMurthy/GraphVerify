"""
Dense passage retriever using FAISS and a sentence-transformer embedding model.

Build an index from a corpus once, then call retrieve() per query.
Also provides a PassThroughRetriever for datasets that already include passages.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


class DenseRetriever:

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        top_k: int = 10,
    ) -> None:
        self._model_name = model_name
        self._top_k = top_k
        self._index = None
        self._passages: List[Dict] = []
        self._embedder = None

    def build_index(self, passages: List[Dict[str, Any]]) -> None:
        """Build a FAISS inner-product index from a passage corpus."""
        import faiss
        from graphverify.embedder import Embedder

        self._embedder = Embedder(self._model_name)
        self._passages = passages

        vecs = self._embedder.encode([p["text"] for p in passages], normalize=True)
        dim  = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs.astype(np.float32))

    def save_index(self, path: str) -> None:
        import faiss, json
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        faiss.write_index(self._index, path + ".faiss")
        with open(path + ".passages.json", "w") as f:
            json.dump(self._passages, f)

    def load_index(self, path: str) -> None:
        import faiss, json
        from graphverify.embedder import Embedder
        self._index    = faiss.read_index(path + ".faiss")
        self._passages = json.load(open(path + ".passages.json"))
        self._embedder = Embedder(self._model_name)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")
        k = top_k or self._top_k
        q_vec = self._embedder.encode([query], normalize=True).astype(np.float32)
        scores, indices = self._index.search(q_vec, k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if 0 <= idx < len(self._passages):
                p = dict(self._passages[idx])
                p["rank"]  = rank
                p["score"] = float(score)
                results.append(p)
        return results


class PassThroughRetriever:
    """Use when passages are already provided with the dataset sample."""

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k

    def retrieve_from(self, passages: List[Dict], query: str, top_k: Optional[int] = None) -> List[Dict]:
        k = top_k or self._top_k
        ranked = sorted(passages, key=lambda p: p.get("score", 0.0), reverse=True)[:k]
        for i, p in enumerate(ranked, start=1):
            p = dict(p)
            p["rank"] = i
            if "score" not in p:
                p["score"] = 1.0 / i
        return ranked
