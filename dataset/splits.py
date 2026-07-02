"""
Deterministic dev/test split construction and split-file persistence.

The revision plan's reproducibility requirement is explicit: "Data split
files and item IDs for every dataset." A split file here is nothing more
than a JSON list of item ids per split — the minimal, auditable artifact a
reviewer needs to confirm which items were used for threshold tuning
(dev) versus final evaluation (test), and to re-run the exact same split
later. Splits are constructed by a seeded shuffle, so the same
`(item_ids, dev_fraction, seed)` always reproduces the same split without
needing to ship the shuffled id list separately from the code that made it
— though the split file should still be saved and versioned per the
reproducibility package, since dataset item ordering from an upstream
source (e.g. a HuggingFace dataset revision) is not itself guaranteed
stable forever.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class DatasetSplit:
    dataset:  str
    seed:     int
    dev_ids:  List[str]
    test_ids: List[str]

    def to_dict(self) -> Dict:
        return {"dataset": self.dataset, "seed": self.seed, "dev_ids": self.dev_ids, "test_ids": self.test_ids}

    @classmethod
    def from_dict(cls, d: Dict) -> "DatasetSplit":
        return cls(dataset=d["dataset"], seed=d["seed"], dev_ids=list(d["dev_ids"]), test_ids=list(d["test_ids"]))


def build_split(
    item_ids: Sequence[str],
    dataset: str,
    dev_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplit:
    """
    Deterministically partitions `item_ids` into a dev set (used only for
    threshold/calibration tuning — see `GraphVerifyConfig.support_threshold`
    etc.) and a test set (used only for final evaluation), via a seeded
    shuffle. Raises ValueError on duplicate ids, since a duplicate would
    silently let one item land in both splits.
    """
    ids = list(item_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids contains duplicates; a dataset item must not appear twice.")
    if not (0.0 < dev_fraction < 1.0):
        raise ValueError(f"dev_fraction must be in (0, 1); got {dev_fraction}")

    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n_dev = max(1, round(len(shuffled) * dev_fraction)) if shuffled else 0
    dev_ids, test_ids = shuffled[:n_dev], shuffled[n_dev:]
    return DatasetSplit(dataset=dataset, seed=seed, dev_ids=dev_ids, test_ids=test_ids)


def save_split(split: DatasetSplit, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2)


def load_split(path: str) -> DatasetSplit:
    with open(path, encoding="utf-8") as f:
        return DatasetSplit.from_dict(json.load(f))


def apply_split(records: List[Dict], split: DatasetSplit, subset: str) -> List[Dict]:
    """Filters `records` (each needing an "id" field) to the given `subset` ("dev" or "test")."""
    if subset not in ("dev", "test"):
        raise ValueError(f"subset must be 'dev' or 'test'; got '{subset}'")
    keep_ids = set(split.dev_ids if subset == "dev" else split.test_ids)
    return [r for r in records if str(r.get("id", "")) in keep_ids]
