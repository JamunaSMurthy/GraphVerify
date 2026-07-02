"""Tests for experiments/_common.py."""
from __future__ import annotations

import argparse
import os

import pytest

from experiments._common import (
    add_dataset_args,
    add_llm_args,
    build_config,
    decompose_claims,
    load_samples,
    parse_dataset_list,
    save_csv,
    save_json,
)

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")


def test_parse_dataset_list_splits_and_strips():
    assert parse_dataset_list(" a, b ,c") == ["a", "b", "c"]
    assert parse_dataset_list("") == []


def test_add_dataset_args_and_add_llm_args_populate_namespace():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    add_llm_args(parser)
    args = parser.parse_args([])
    assert args.split == "validation"
    assert args.llm_backend == "openai"
    assert args.llm_model == "gpt-4o-mini"


def test_build_config_maps_namespace_fields():
    parser = argparse.ArgumentParser()
    add_llm_args(parser)
    args = parser.parse_args(["--llm_backend", "anthropic", "--llm_model", "claude-x"])
    cfg = build_config(args)
    assert cfg.llm_backend == "anthropic"
    assert cfg.llm_model == "claude-x"


def test_build_config_applies_overrides():
    parser = argparse.ArgumentParser()
    add_llm_args(parser)
    args = parser.parse_args([])
    cfg = build_config(args, evidence_mode="text")
    assert cfg.evidence_mode == "text"


def test_load_samples_from_data_dir():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser)
    args = parser.parse_args(["--data_dir", FIXTURES_DIR])
    samples = load_samples("hotpotqa", args)
    assert len(samples) == 3


def test_decompose_claims_keys_by_sample_id(fake_llm):
    samples = [
        {"id": "s1", "generated": "Einstein was born in Ulm. He won a prize."},
        {"id": "s2", "generated": ""},
    ]
    result = decompose_claims(fake_llm, samples)
    assert len(result["s1"]) == 2
    assert result["s2"] == []


def test_save_json_and_save_csv_write_files(tmp_path):
    json_path = str(tmp_path / "out.json")
    save_json({"a": 1}, json_path)
    assert os.path.exists(json_path)

    csv_path = str(tmp_path / "out.csv")
    save_csv([{"a": 1, "b": 2}], csv_path)
    assert os.path.exists(csv_path)


def test_save_csv_requires_fieldnames_when_rows_empty(tmp_path):
    with pytest.raises(ValueError):
        save_csv([], str(tmp_path / "empty.csv"))
