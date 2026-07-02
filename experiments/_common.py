"""
Shared CLI plumbing for every script in `experiments/`: argument parsing
conventions, LLM/dataset construction, and the shared-claim-decomposition
step the baseline-fairness protocol requires
(see `baselines/base.py`'s module docstring).

Every script in this package follows the same shape: a small, directly
callable `run_*(...)` function containing the actual experiment logic
(taking already-constructed objects like an `LLMClient`, so tests can pass
a fake), and a `main(argv=None)` CLI wrapper that parses arguments, builds
real objects via this module, calls `run_*`, and writes output. This keeps
every script both runnable from the command line and unit-testable without
a real API key.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

from graphverify.claim_decomposer import ClaimDecomposer
from graphverify.config import GraphVerifyConfig
from graphverify.llm_client import LLMClient
from dataset.loader import load_dataset

ALL_DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "fever", "ragtruth"]


def add_dataset_args(parser: argparse.ArgumentParser, default_datasets: str = ",".join(ALL_DATASETS)) -> None:
    parser.add_argument("--datasets", type=str, default=default_datasets,
                         help="Comma-separated dataset names.")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)


def add_llm_args(parser: argparse.ArgumentParser, default_model: str = "gpt-4o-mini") -> None:
    parser.add_argument("--llm_backend", type=str, default="openai", choices=["openai", "anthropic", "local"])
    parser.add_argument("--llm_model", type=str, default=default_model)
    parser.add_argument("--local_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embed_model", type=str, default="BAAI/bge-base-en-v1.5")


def add_output_args(parser: argparse.ArgumentParser, default_dir: str) -> None:
    parser.add_argument("--output_dir", type=str, default=default_dir)


def build_config(args: argparse.Namespace, **overrides: Any) -> GraphVerifyConfig:
    kwargs: Dict[str, Any] = dict(
        llm_backend=getattr(args, "llm_backend", "openai"),
        llm_model=getattr(args, "llm_model", "gpt-4o-mini"),
        local_model_path=getattr(args, "local_model", "Qwen/Qwen2.5-7B-Instruct"),
        embed_model=getattr(args, "embed_model", "BAAI/bge-base-en-v1.5"),
    )
    kwargs.update(overrides)
    return GraphVerifyConfig(**kwargs)


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    return LLMClient(build_config(args))


def parse_dataset_list(datasets_arg: str) -> List[str]:
    return [d.strip() for d in datasets_arg.split(",") if d.strip()]


def load_samples(dataset: str, args: argparse.Namespace) -> List[Dict[str, Any]]:
    return load_dataset(
        dataset, split=args.split, data_dir=args.data_dir, max_samples=args.max_samples,
    )


def decompose_claims(llm_client: LLMClient, samples: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Runs claim decomposition exactly once per sample, keyed by sample id.
    Every method compared in an experiment script must be handed this same
    mapping — see `baselines/base.py`'s fairness-protocol docstring — rather
    than each method decomposing the generated answer independently.
    """
    decomposer = ClaimDecomposer(llm_client)
    claims_by_id: Dict[str, List[str]] = {}
    for sample in samples:
        sid = str(sample.get("id", ""))
        answer = sample.get("generated") or sample.get("answer", "")
        claims_by_id[sid] = decomposer.decompose(answer) if answer else []
    return claims_by_id


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], path: str, fieldnames: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"save_csv({path}): no rows and no explicit fieldnames given.")
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
