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


def _load_ragtruth(split: str = "test", max_samples: Optional[int] = None) -> List[Dict]:
    default_path = os.path.join(os.path.dirname(__file__), "data", "ragtruth.jsonl")
    if os.path.exists(default_path):
        return _load_jsonl(default_path, max_samples)
    print(f"[WARNING] RAGTruth not found at {default_path}. "
          "Download and place as dataset/data/ragtruth.jsonl")
    return []


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
