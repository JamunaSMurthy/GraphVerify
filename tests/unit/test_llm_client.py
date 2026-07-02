"""
Tests for graphverify/llm_client.py that don't require real network access:
`chat_json`'s markdown-fence stripping / recovery parsing, and backend
selection. Real API calls are covered by tests/integration/.
"""
from __future__ import annotations

import pytest

from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient


class _StubClient(LLMClient):
    """Bypasses real backend construction so we can test chat_json() parsing in isolation."""

    def __init__(self, raw_response: str) -> None:
        self.cfg = GraphVerifyConfig()
        self._backend = "openai"
        self._raw_response = raw_response

    def chat(self, messages, json_mode: bool = False) -> str:
        return self._raw_response


def test_chat_json_parses_plain_json():
    client = _StubClient('{"a": 1}')
    assert client.chat_json([]) == {"a": 1}


def test_chat_json_strips_markdown_code_fence():
    client = _StubClient('```json\n{"a": 1}\n```')
    assert client.chat_json([]) == {"a": 1}


def test_chat_json_recovers_json_embedded_in_prose():
    client = _StubClient('Sure, here is the JSON: {"a": 1} -- hope that helps!')
    assert client.chat_json([]) == {"a": 1}


def test_chat_json_recovers_list():
    client = _StubClient('The result is: [{"x": 1}, {"x": 2}]')
    assert client.chat_json([]) == [{"x": 1}, {"x": 2}]


def test_chat_json_returns_none_on_total_garbage():
    client = _StubClient("not json at all, sorry")
    assert client.chat_json([]) is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        LLMClient(GraphVerifyConfig(llm_backend="not_a_real_backend"))
