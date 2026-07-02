"""Tests for dataset/loader.py: schema validation and local-file loading (no network)."""
from __future__ import annotations

import json
import os

import pytest

from dataset.loader import _chunk_source_into_passages, load_dataset, save_jsonl, validate_schema

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")


def test_load_dataset_from_local_data_dir():
    samples = load_dataset("hotpotqa", split="validation", data_dir=FIXTURES_DIR)
    assert len(samples) == 3
    assert samples[0]["id"] == "hp1"


def test_load_dataset_respects_max_samples():
    samples = load_dataset("hotpotqa", split="validation", data_dir=FIXTURES_DIR, max_samples=2)
    assert len(samples) == 2


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


def test_fixture_data_passes_schema_validation():
    samples = load_dataset("hotpotqa", split="validation", data_dir=FIXTURES_DIR)
    errors = validate_schema(samples)
    assert errors == []


def test_validate_schema_flags_missing_field():
    errors = validate_schema([{"id": "1", "query": "q"}])  # missing most required fields
    assert any("missing required field" in e for e in errors)


def test_validate_schema_flags_invalid_verdict():
    record = {
        "id": "1", "query": "q", "answer": "a", "generated": "g",
        "passages": [], "gold_verdict": "MaybeSupported", "gold_path": "", "label": "",
    }
    errors = validate_schema([record])
    assert any("invalid gold_verdict" in e for e in errors)


def test_validate_schema_flags_duplicate_ids():
    record = {
        "id": "1", "query": "q", "answer": "a", "generated": "g",
        "passages": [], "gold_verdict": "Supported", "gold_path": "", "label": "",
    }
    errors = validate_schema([record, dict(record)])
    assert any("duplicate id" in e for e in errors)


def test_validate_schema_flags_bad_passage_shape():
    record = {
        "id": "1", "query": "q", "answer": "a", "generated": "g",
        "passages": [{"id": "p1"}],  # missing text/rank/score
        "gold_verdict": "Supported", "gold_path": "", "label": "",
    }
    errors = validate_schema([record])
    assert any("passage[0]" in e for e in errors)


def test_save_jsonl_and_load_roundtrip(tmp_path):
    records = [{"id": "1", "query": "q"}, {"id": "2", "query": "q2"}]
    path = str(tmp_path / "out.jsonl")
    save_jsonl(records, path)
    with open(path) as f:
        lines = [json.loads(line) for line in f]
    assert lines == records


def test_chunk_source_into_passages_produces_ranked_passages():
    text = "First paragraph here.\nSecond paragraph here.\nThird paragraph."
    passages = _chunk_source_into_passages(text, base_id="doc1", max_chars=1000)
    assert len(passages) == 3
    assert all(p["id"].startswith("doc1_") for p in passages)
    assert [p["rank"] for p in passages] == [1, 2, 3]


def test_chunk_source_into_passages_splits_long_single_paragraph():
    text = "word " * 500  # no newlines -- forced sentence/char-based chunking
    passages = _chunk_source_into_passages(text, base_id="doc1", max_chars=200)
    assert len(passages) > 1
    assert all(len(p["text"]) <= 200 for p in passages)


def test_chunk_source_into_passages_handles_empty_text():
    passages = _chunk_source_into_passages("", base_id="doc1")
    assert len(passages) == 1
    assert passages[0]["text"] == ""
