"""Singleton embedding model wrapper using sentence-transformers."""
from __future__ import annotations

import numpy as np
from typing import List, Union

_MODEL_INSTANCE = None
_MODEL_NAME: str = ""


def _get_model(model_name: str):
    global _MODEL_INSTANCE, _MODEL_NAME
    if _MODEL_INSTANCE is None or _MODEL_NAME != model_name:
        from sentence_transformers import SentenceTransformer
        _MODEL_INSTANCE = SentenceTransformer(model_name)
        _MODEL_NAME = model_name
    return _MODEL_INSTANCE


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self._model_name = model_name
        self._model = _get_model(model_name)

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vecs = self._model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return np.array(vecs, dtype=np.float32)

    def cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(a, b))

    def cosine_sim_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """a: (m, d), b: (n, d) → (m, n) cosine similarity matrix."""
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (a @ b.T).astype(np.float32)
