"""
Shared pytest fixtures usable by both tests/unit/ and tests/integration/:
fake LLM double and small sample data. The autouse fake-embedder patch
lives in tests/unit/conftest.py specifically -- integration tests need the
*real* embedder, so it must not be patched at this (shared) level.
"""
from __future__ import annotations

import pytest

from tests.fakes import FakeLLMClient


@pytest.fixture
def fake_llm():
    return FakeLLMClient()


@pytest.fixture
def sample_passages():
    return [
        {
            "id": "p1", "rank": 1, "score": 0.95,
            "text": "Albert Einstein was born in Ulm, Germany in 1879. He received the "
                    "Nobel Prize in Physics in 1921 for the photoelectric effect.",
            "title": "Albert Einstein",
        },
        {
            "id": "p2", "rank": 2, "score": 0.80,
            "text": "The Nobel Prize in Physics is awarded annually by the Royal Swedish "
                    "Academy of Sciences.",
            "title": "Nobel Prize",
        },
    ]


@pytest.fixture
def sample_answer():
    return "Albert Einstein won the Nobel Prize in Physics in 1921. He was born in Ulm."


@pytest.fixture
def sample_query():
    return "When did Einstein win the Nobel Prize and where was he born?"
