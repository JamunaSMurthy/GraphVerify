"""
Integration test: makes a real OpenAI API call. Skipped unless
RUN_INTEGRATION_TESTS=1 and OPENAI_API_KEY is set (see .env.example).

Run explicitly with:
  RUN_INTEGRATION_TESTS=1 pytest tests/integration/test_openai_llm_client.py -m integration
"""
from __future__ import annotations

import os

import pytest

from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient

pytestmark = pytest.mark.integration

_RUN = os.getenv("RUN_INTEGRATION_TESTS") == "1"
_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))


@pytest.mark.skipif(not _RUN, reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY is not set.")
def test_real_chat_completion_returns_text():
    client = LLMClient(GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini"))
    response = client.chat([
        {"role": "system", "content": "Reply with exactly one word."},
        {"role": "user", "content": "Say 'pong'."},
    ])
    assert isinstance(response, str)
    assert len(response.strip()) > 0


@pytest.mark.skipif(not _RUN, reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
@pytest.mark.skipif(not _HAS_KEY, reason="OPENAI_API_KEY is not set.")
def test_real_chat_json_returns_parsed_object():
    client = LLMClient(GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini"))
    result = client.chat_json([
        {"role": "system", "content": "You return only JSON."},
        {"role": "user", "content": 'Return this exact JSON object: {"status": "ok"}'},
    ])
    assert isinstance(result, dict)
    assert "status" in result
