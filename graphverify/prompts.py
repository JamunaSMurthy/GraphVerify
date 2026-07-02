"""
Loads versioned prompt templates from the top-level `prompts/` directory.

Every LLM prompt used anywhere in this codebase (core pipeline, hybrid verdict
head, baselines) lives as a plain-text file under `prompts/` rather than as an
in-code string constant. This keeps the reproducibility package self-contained:
`prompts/` can be shipped as-is alongside code, splits, and configs, and a
reviewer can read the exact prompt text without reading Python.

Templates use ``str.format``-style ``{placeholder}`` substitution. Braces that
must be treated as literal text (e.g. inside an example JSON object) are
written doubled, ``{{`` / ``}}``, exactly as in ``str.format``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """
    Load a prompt template by file stem, e.g. ``load_prompt("claim_decomposition_system")``
    reads ``prompts/claim_decomposition_system.txt``.

    Raises FileNotFoundError with the resolved path if the template is missing,
    so a broken reference fails immediately instead of silently sending an
    empty prompt to the LLM.
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").rstrip("\n")
