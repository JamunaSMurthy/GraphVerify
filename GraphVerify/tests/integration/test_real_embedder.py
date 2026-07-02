"""
Integration test: downloads and runs the real BAAI/bge-base-en-v1.5
sentence-transformers model. Skipped unless RUN_INTEGRATION_TESTS=1 (no API
key needed, but this does download ~440MB on first run).

Run explicitly with:
  RUN_INTEGRATION_TESTS=1 pytest tests/integration/test_real_embedder.py -m integration
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from graphverify.embedder import Embedder

pytestmark = pytest.mark.integration

_RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(not _RUN, reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
def test_real_embedder_produces_768_dim_vectors():
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    vecs = embedder.encode(["hello world", "goodbye world"])
    assert vecs.shape == (2, 768)


@pytest.mark.skipif(not _RUN, reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
def test_real_embedder_similar_sentences_score_higher_than_unrelated():
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    vecs = embedder.encode([
        "Albert Einstein was a theoretical physicist.",
        "Einstein developed the theory of relativity.",
        "Bananas are a good source of potassium.",
    ])
    sim_related = embedder.cosine_sim(vecs[0], vecs[1])
    sim_unrelated = embedder.cosine_sim(vecs[0], vecs[2])
    assert sim_related > sim_unrelated


@pytest.mark.skipif(not _RUN, reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
def test_real_embedder_normalized_vectors_have_unit_norm():
    embedder = Embedder("BAAI/bge-base-en-v1.5")
    vecs = embedder.encode(["a test sentence"], normalize=True)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-3)
