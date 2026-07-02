"""
Dataset loaders for verification benchmarks.

Supported datasets:
  hotpotqa          — multi-hop QA
  2wikimultihopqa   — multi-hop QA
  musique           — compositional multi-hop QA
  fever             — fact verification
  ragtruth          — hallucination detection in RAG

Each loader returns a list of dicts with a unified schema:
  {
    id           : str
    query        : str
    answer       : str          # gold answer (QA) or claim (FEVER)
    generated    : str          # generated RAG answer if pre-produced
    passages     : List[Dict]   # retrieved context (text, id, rank, score)
    gold_verdict : str          # Supported | Unsupported | Contradictory
    gold_path    : str          # evidence path string for path correctness eval
    label        : str          # original dataset label
  }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPPORTED_DATASETS = [
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "fever",
    "ragtruth",
]


def load_dataset(
    name: str,
    split: str = "validation",
    data_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load a benchmark dataset.

    Checks for a local JSONL file first; falls back to HuggingFace datasets.
    """
    name = name.lower().strip()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {SUPPORTED_DATASETS}")

    if data_dir:
        for fname in (f"{name}_{split}.jsonl", f"{name}.jsonl"):
            p = Path(data_dir) / fname
            if p.exists():
                return _load_jsonl(str(p), max_samples)

    return {
        "hotpotqa":        _load_hotpotqa,
        "2wikimultihopqa": _load_2wiki,
        "musique":         _load_musique,
        "fever":           _load_fever,
        "ragtruth":        _load_ragtruth,
    }[name](split=split, max_samples=max_samples)


def _load_hotpotqa(split: str = "validation", max_samples: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset as hf_load
    hf_split = "validation" if split in ("validation", "val", "dev") else split
    ds = hf_load("hotpot_qa", "distractor", split=hf_split, trust_remote_code=True)
    records = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        passages = [
            {"id": f"{ex['id']}_ctx_{j}", "text": " ".join(sents),
             "title": title, "rank": j + 1, "score": 1.0 / (j + 1)}
            for j, (title, sents) in enumerate(
                zip(ex["context"]["title"], ex["context"]["sentences"])
            )
        ]
        records.append({
            "id": ex["id"], "query": ex["question"], "answer": ex["answer"],
            "generated": "", "passages": passages, "gold_verdict": "Supported",
            "gold_path": _hotpot_evidence_path(ex), "label": ex.get("level", ""),
        })
    return records


def _hotpot_evidence_path(ex: Dict) -> str:
    facts  = ex.get("supporting_facts", {})
    parts  = [f"{t}[{s}]" for t, s in zip(facts.get("title", []), facts.get("sent_id", []))]
    return " → ".join(parts)


def _load_2wiki(split: str = "validation", max_samples: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset as hf_load
    hf_split = "validation" if split in ("validation", "val", "dev") else split
    ds = hf_load("wiki_multi_hop_qa", split=hf_split, trust_remote_code=True)
    records = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        passages = _contexts_to_passages(
            ex.get("context", {}).get("title", []),
            ex.get("context", {}).get("content", []),
            ex["_id"],
        )
        records.append({
            "id": ex["_id"], "query": ex["question"], "answer": ex["answer"],
            "generated": "", "passages": passages, "gold_verdict": "Supported",
            "gold_path": "", "label": ex.get("type", ""),
        })
    return records


def _load_musique(split: str = "validation", max_samples: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset as hf_load
    hf_split = "validation" if split in ("validation", "val", "dev") else split
    ds = hf_load("musique_ans_v1.0", split=hf_split, trust_remote_code=True)
    records = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        passages = [
            {"id": f"{ex['id']}_p{p['idx']}", "text": p["paragraph_text"],
             "title": p.get("title", ""), "rank": p["idx"] + 1, "score": 1.0 / (p["idx"] + 1)}
            for p in ex.get("paragraphs", [])
        ]
        records.append({
            "id": ex["id"], "query": ex["question"], "answer": ex["answer"],
            "generated": "", "passages": passages,
            "gold_verdict": "Supported" if ex.get("answerable", True) else "Unsupported",
            "gold_path": "", "label": str(ex.get("answerable", True)),
        })
    return records


def _load_fever(split: str = "validation", max_samples: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset as hf_load
    hf_split = "validation" if split in ("validation", "val", "dev") else split
    ds = hf_load("fever", "v1.0", split=hf_split, trust_remote_code=True)
    label_map = {"SUPPORTS": "Supported", "REFUTES": "Contradictory", "NOT ENOUGH INFO": "Unsupported"}
    records = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        passages = [
            {"id": f"fever_{ex['id']}_ev_{j}", "text": ev, "rank": j + 1, "score": 1.0 / (j + 1)}
            for j, ev in enumerate(
                ex.get("evidence_annotation", [{}])[0].get("evidence", [])
            ) if isinstance(ev, str) and ev
        ] or [{"id": f"fever_{ex['id']}_claim", "text": ex["claim"], "rank": 1, "score": 1.0}]
        records.append({
            "id": str(ex["id"]), "query": ex["claim"], "answer": ex["claim"],
            "generated": "", "passages": passages,
            "gold_verdict": label_map.get(ex.get("label", ""), "Unsupported"),
            "gold_path": "", "label": ex.get("label", ""),
        })
    return records


RAGTRUTH_SOURCE_INFO_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl"
RAGTRUTH_RESPONSE_URL    = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl"

# RAGTruth (Niu et al. 2024) label_type values that mark a hallucination span
# as an explicit factual conflict with the source vs. ungrounded/added
# content with no direct conflict. See the official release at
# https://github.com/ParticleMedia/RAGTruth for the annotation guideline
# this taxonomy comes from.
_RAGTRUTH_CONFLICT_LABELS  = {"Evident Conflict", "Subtle Conflict"}
_RAGTRUTH_BASELESS_LABELS  = {"Evident Baseless Info", "Subtle Baseless Info"}


def _load_ragtruth(split: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
    """
    Loads RAGTruth (Niu et al. 2024), joining ``response.jsonl`` (one
    generated response per row, with character-span hallucination labels)
    against ``source_info.jsonl`` (the source document each response was
    generated from) on ``source_id``. Files are downloaded once from the
    official GitHub release and cached under ``dataset/data/``.

    RAGTruth ships only ``train``/``test`` splits (no ``validation``); by
    convention shared with the other loaders in this module,
    ``"validation"``/``"val"``/``"dev"`` are mapped to RAGTruth's ``train``
    split, used here as the development/threshold-tuning pool.

    RAGTruth is not a retrieval benchmark, so a response's source document is
    not pre-chunked into ranked passages the way HotpotQA/2Wiki/MuSiQue/FEVER
    are. ``source_info`` is split into paragraph-sized passages (see
    :func:`_chunk_source_into_passages`) to give the graph-construction
    pipeline something structurally equivalent to retrieved evidence; this
    is a documented approximation, not a claim that RAGTruth includes gold
    passage rankings.

    ``gold_verdict`` is derived per response, not per claim: a response with
    no hallucination labels is "Supported"; a response with at least one
    "*Conflict" label is "Contradictory" (an explicit factual conflict with
    the source takes priority — consistent with the conflict-before-support
    priority `graphverify.verdict_assigner.VerdictAssigner` uses); a
    response with only "*Baseless Info" labels (ungrounded but not directly
    conflicting) is "Unsupported". Downstream claim-level evaluation should
    re-derive per-claim labels from the response-level character spans in
    ``label`` (kept in the record) rather than treating this response-level
    verdict as a claim-level gold label.
    """
    hf_split = "train" if split in ("validation", "val", "dev") else split
    if hf_split not in ("train", "test"):
        raise ValueError(f"RAGTruth only has train/test splits (mapped from validation/val/dev); got '{split}'")

    source_by_id = _load_ragtruth_source_info()
    responses = _load_ragtruth_responses(hf_split)

    records = []
    for i, resp in enumerate(responses):
        if max_samples and i >= max_samples:
            break
        source = source_by_id.get(resp.get("source_id"))
        if source is None:
            continue

        source_text = str(source.get("source_info", ""))
        passages = _chunk_source_into_passages(source_text, base_id=f"ragtruth_{resp['source_id']}")

        labels = resp.get("labels", [])
        label_types = {lbl.get("label_type") for lbl in labels}
        if not labels:
            gold_verdict = "Supported"
        elif label_types & _RAGTRUTH_CONFLICT_LABELS:
            gold_verdict = "Contradictory"
        else:
            gold_verdict = "Unsupported"

        records.append({
            "id": f"ragtruth_{resp['id']}",
            "query": source.get("prompt", ""),
            "answer": "",
            "generated": resp.get("response", ""),
            "passages": passages,
            "gold_verdict": gold_verdict,
            "gold_path": "",
            "label": json.dumps({
                "task_type": source.get("task_type", ""),
                "quality": resp.get("quality", ""),
                "model": resp.get("model", ""),
                "hallucination_spans": labels,
            }),
        })
    return records


def _ragtruth_cache_dir() -> str:
    cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _download_ragtruth_file(url: str, cache_path: str) -> str:
    if os.path.exists(cache_path):
        return cache_path
    import requests
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return cache_path


def _load_ragtruth_source_info() -> Dict[str, Dict]:
    cache_path = os.path.join(_ragtruth_cache_dir(), "ragtruth_source_info.jsonl")
    _download_ragtruth_file(RAGTRUTH_SOURCE_INFO_URL, cache_path)
    return {str(rec["source_id"]): rec for rec in _load_jsonl(cache_path, max_samples=None)}


def _load_ragtruth_responses(split: str) -> List[Dict]:
    cache_path = os.path.join(_ragtruth_cache_dir(), "ragtruth_response.jsonl")
    _download_ragtruth_file(RAGTRUTH_RESPONSE_URL, cache_path)
    return [rec for rec in _load_jsonl(cache_path, max_samples=None) if rec.get("split") == split]


def _chunk_source_into_passages(source_text: str, base_id: str, max_chars: int = 600) -> List[Dict]:
    """
    Splits a RAGTruth source document into paragraph-sized passage dicts
    (falling back to sentence grouping if the source has no paragraph
    breaks), each capped at `max_chars`, so it has the same
    ``{id, text, rank, score}`` shape as every other loader's passages.
    """
    paragraphs = [p.strip() for p in source_text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        import re
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text.strip()) if s.strip()]
        paragraphs, chunk = [], []
        chunk_len = 0
        for sent in sentences:
            chunk.append(sent)
            chunk_len += len(sent)
            if chunk_len >= max_chars:
                paragraphs.append(" ".join(chunk))
                chunk, chunk_len = [], 0
        if chunk:
            paragraphs.append(" ".join(chunk))

    passages = []
    for j, para in enumerate(paragraphs):
        for k in range(0, len(para), max_chars):
            passages.append({
                "id": f"{base_id}_p{j}_{k // max_chars}",
                "text": para[k: k + max_chars],
                "rank": len(passages) + 1,
                "score": 1.0 / (len(passages) + 1),
            })
    return passages or [{"id": f"{base_id}_p0", "text": source_text[:max_chars], "rank": 1, "score": 1.0}]


def _contexts_to_passages(titles, contents, base_id):
    return [
        {"id": f"{base_id}_ctx_{j}",
         "text": content if isinstance(content, str) else " ".join(content),
         "title": title, "rank": j + 1, "score": 1.0 / (j + 1)}
        for j, (title, content) in enumerate(zip(titles, contents))
    ]


def _load_jsonl(path: str, max_samples: Optional[int]) -> List[Dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


_REQUIRED_FIELDS = ("id", "query", "answer", "generated", "passages", "gold_verdict", "gold_path", "label")
_VALID_VERDICTS = {"Supported", "Unsupported", "Contradictory"}
_REQUIRED_PASSAGE_FIELDS = ("id", "text", "rank", "score")


def validate_schema(records: List[Dict[str, Any]], strict_verdict: bool = True) -> List[str]:
    """
    Validates that every record matches the unified dataset schema
    documented at the top of this module. Returns a list of human-readable
    error strings (empty list == valid). Used by
    ``experiments/build_dataset_statistics.py`` and the dataset-loader test
    suite to catch schema drift before it silently corrupts downstream
    graph-building/verification runs.

    `strict_verdict` controls whether `gold_verdict` must be one of
    Supported/Unsupported/Contradictory (it always must be a string; some
    exploratory data — e.g. an unlabeled corpus — may legitimately use "").
    """
    errors: List[str] = []
    seen_ids = set()

    for i, rec in enumerate(records):
        prefix = f"record[{i}]"
        if not isinstance(rec, dict):
            errors.append(f"{prefix}: not a dict")
            continue

        for field in _REQUIRED_FIELDS:
            if field not in rec:
                errors.append(f"{prefix}: missing required field '{field}'")

        rid = rec.get("id")
        if rid is not None:
            if rid in seen_ids:
                errors.append(f"{prefix}: duplicate id '{rid}'")
            seen_ids.add(rid)

        verdict = rec.get("gold_verdict", "")
        if strict_verdict and verdict and verdict not in _VALID_VERDICTS:
            errors.append(f"{prefix} (id={rid}): invalid gold_verdict '{verdict}'")

        passages = rec.get("passages")
        if not isinstance(passages, list):
            errors.append(f"{prefix} (id={rid}): 'passages' must be a list")
            continue
        for j, p in enumerate(passages):
            if not isinstance(p, dict):
                errors.append(f"{prefix} (id={rid}) passage[{j}]: not a dict")
                continue
            for field in _REQUIRED_PASSAGE_FIELDS:
                if field not in p:
                    errors.append(f"{prefix} (id={rid}) passage[{j}]: missing field '{field}'")

    return errors
