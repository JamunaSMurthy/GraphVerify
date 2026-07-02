"""
Thin CLI entrypoint for the component ablation sweep. The actual ablation
logic (variant definitions, config overrides, per-variant evaluation) lives
in `eval/ablation.py` — this wrapper exists only so every P0/P1/P2
experiment has a discoverable, consistently-named entrypoint under
`experiments/`, without duplicating `eval/ablation.py`'s logic.

Usage: identical to `eval/ablation.py`.
  python experiments/run_component_ablation.py \\
      --dataset hotpotqa --split validation --max_samples 200 \\
      --output output/results/component_ablation
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.ablation import main as run_ablation_main

if __name__ == "__main__":
    run_ablation_main()
