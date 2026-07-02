"""
Baseline claim-level verifiers compared against GraphVerify.

``BASELINE_REGISTRY`` maps each method's ``name`` (matching
:attr:`baselines.base.BaselineVerifier.name`) to its class, so
``experiments/`` scripts can iterate over every baseline uniformly:

>>> from baselines import BASELINE_REGISTRY
>>> for name, cls in BASELINE_REGISTRY.items():
...     verifier = cls(llm_client)
...     records = verifier.verify_claims(query, passages, claims)

See ``baselines/base.py`` for the shared interface and the fairness
protocol (claims must be decomposed once, upstream, and the same list
passed to every method).
"""
from __future__ import annotations

from typing import Dict, Type

from .base import BaselineVerifier
from .safe import SAFEVerifier
from .rarr import RARRVerifier
from .fire import FIREVerifier
from .citefix import CiteFixVerifier
from .graphrag_adapted import GraphRAGAdaptedVerifier
from .hipporag_adapted import HippoRAGAdaptedVerifier
from .graphcheck_adapted import GraphCheckAdaptedVerifier
from .hybrid_kg_llm import HybridKGLLMVerifier
from .llm_text_verifier import LLMTextVerifier

BASELINE_REGISTRY: Dict[str, Type[BaselineVerifier]] = {
    cls.name: cls
    for cls in (
        SAFEVerifier,
        RARRVerifier,
        FIREVerifier,
        CiteFixVerifier,
        GraphRAGAdaptedVerifier,
        HippoRAGAdaptedVerifier,
        GraphCheckAdaptedVerifier,
        HybridKGLLMVerifier,
        LLMTextVerifier,
    )
}

__all__ = [
    "BaselineVerifier",
    "BASELINE_REGISTRY",
    "SAFEVerifier",
    "RARRVerifier",
    "FIREVerifier",
    "CiteFixVerifier",
    "GraphRAGAdaptedVerifier",
    "HippoRAGAdaptedVerifier",
    "GraphCheckAdaptedVerifier",
    "HybridKGLLMVerifier",
    "LLMTextVerifier",
]
