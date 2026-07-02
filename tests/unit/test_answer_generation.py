"""Tests for dataset/answer_generation.py."""
from __future__ import annotations

from dataset.answer_generation import generate_answer, generate_answers_for_dataset


class _EchoLLM:
    """Returns a fixed, recognizable answer so we can assert generation happened."""

    def chat(self, messages, json_mode=False):
        return "GENERATED ANSWER"

    def chat_json(self, messages):
        return {}


def test_generate_answer_returns_text_when_passages_present():
    passages = [{"id": "p1", "text": "Paris is the capital of France.", "rank": 1, "score": 0.9}]
    answer = generate_answer(_EchoLLM(), "What is the capital of France?", passages)
    assert answer == "GENERATED ANSWER"


def test_generate_answer_empty_without_passages():
    answer = generate_answer(_EchoLLM(), "Some query", [])
    assert answer == ""


def test_generate_answer_respects_max_passages():
    calls = []

    class TrackingLLM(_EchoLLM):
        def chat(self, messages, json_mode=False):
            calls.append(messages[-1]["content"])
            return "ANSWER"

    passages = [{"id": f"p{i}", "text": f"text {i}", "rank": i, "score": 1.0} for i in range(5)]
    generate_answer(TrackingLLM(), "query", passages, max_passages=2)
    assert "p2" not in calls[0]  # only p0, p1 should appear in the prompt
    assert "p0" in calls[0] and "p1" in calls[0]


def test_generate_answers_for_dataset_fills_missing_generated_field():
    records = [
        {"id": "1", "query": "q1", "passages": [{"id": "p1", "text": "t", "rank": 1, "score": 1.0}], "generated": ""},
        {"id": "2", "query": "q2", "passages": [{"id": "p2", "text": "t2", "rank": 1, "score": 1.0}], "generated": "already there"},
    ]
    result = generate_answers_for_dataset(_EchoLLM(), records)
    assert result[0]["generated"] == "GENERATED ANSWER"
    assert result[1]["generated"] == "already there"  # left unchanged


def test_generate_answers_for_dataset_does_not_mutate_input():
    records = [{"id": "1", "query": "q1", "passages": [{"id": "p1", "text": "t", "rank": 1, "score": 1.0}], "generated": ""}]
    generate_answers_for_dataset(_EchoLLM(), records)
    assert records[0]["generated"] == ""
