"""
Deterministic, offline test doubles for the two external dependencies every
production module talks to: an LLM (`graphverify.llm_client.LLMClient`) and
a sentence embedding model (`graphverify.embedder.Embedder`).

Nothing here makes a network call. `FakeLLMClient` implements the same
`chat`/`chat_json` interface as `LLMClient` with simple, documented
heuristics per prompt type (routed by matching text in the system prompt,
since every prompt in `prompts/` has a distinct, stable system message) and
supports `overrides` for tests that need a specific scripted answer.
`install_fake_embedder` monkeypatches the process-wide embedding model
singleton in `graphverify.embedder` so every code path that lazily builds
an `Embedder` (entity linking, relation normalization, path scoring,
GraphCheck-adapted) gets deterministic hash-based vectors instead of
downloading `BAAI/bge-base-en-v1.5`.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9]+")
_ID_RE = re.compile(r"\[([^\]\s]+)\]")


def _tokenize(text: str) -> set:
    return set(_WORD_RE.findall(text.lower()))


def _entailment_heuristic(claim: str, evidence_text: str) -> Tuple[str, float]:
    """
    Token-overlap heuristic used as the default behavior for every
    entailment-style prompt (text evidence, SAFE, RARR agreement, FIRE,
    CiteFix, Hybrid KG-LLM). High overlap -> Supported; low overlap ->
    Unsupported. Never returns Contradictory by default — tests that need
    to exercise contradiction detection should use `FakeLLMClient(overrides=...)`
    for precise control, since "the evidence conflicts with the claim" is
    not something a generic token-overlap heuristic can detect reliably.
    """
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return "Unsupported", 0.0
    evidence_tokens = _tokenize(evidence_text)
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if overlap >= 0.8:
        return "Supported", 0.9
    if overlap <= 0.2:
        return "Unsupported", 0.1
    return "Unsupported", 0.3


def _naive_triple(claim: str) -> Dict[str, str]:
    """Crude, deterministic (head, relation, tail) split used by the fake triple-extraction routes."""
    words = claim.strip().rstrip(".").split()
    if not words:
        return {"head": "", "relation": "", "tail": ""}
    if len(words) == 1:
        return {"head": words[0], "relation": "is", "tail": words[0]}
    return {
        "head": words[0],
        "relation": " ".join(words[1:-1]) if len(words) > 2 else "relatedTo",
        "tail": words[-1],
    }


def _extract_ids(text: str) -> List[str]:
    return _ID_RE.findall(text)


Predicate = Callable[[str, str], bool]
Responder = Callable[[str, str], Any]


class FakeLLMClient:
    """
    Deterministic, offline stand-in for `graphverify.llm_client.LLMClient`.

    `overrides` is a list of `(predicate, response)` pairs checked, in
    order, before the default heuristics: `predicate(system, user) -> bool`
    selects the override, and `response` is either a literal dict/list or a
    callable `(system, user) -> dict/list` for dynamic responses. This lets
    a test pin down the exact behavior for one specific prompt (e.g. force
    a Contradictory verdict) while every other call still gets sensible
    default behavior.

    Every call is recorded in `self.calls` (list of message lists) so tests
    can assert on how many LLM calls a code path made, or inspect prompt
    content.
    """

    def __init__(self, overrides: Optional[List[Tuple[Predicate, Responder]]] = None) -> None:
        self.overrides = overrides or []
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        return json.dumps(self.chat_json(messages))

    def chat_json(self, messages: List[Dict[str, str]]) -> Any:
        self.calls.append(messages)
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = messages[-1]["content"] if messages else ""

        for predicate, response in self.overrides:
            if predicate(system, user):
                return response(system, user) if callable(response) else response

        return self._default(system, user)

    # -- default routing, keyed on stable phrases in each prompts/*_system.txt.
    # Matching is done against a whitespace-normalized copy of `system` so a
    # routing phrase that happens to span a line wrap in the .txt file still
    # matches (see the SAFE prompt, whose defining phrase wraps mid-sentence).
    def _default(self, system: str, user: str) -> Any:
        norm_system = re.sub(r"\s+", " ", system)

        if "claim decomposition assistant" in norm_system:
            return self._decompose(user)
        if "graph-embedding-based fact-checking" in norm_system:
            return _naive_triple(_between(user, "Claim:", "Return JSON"))
        if "triple extraction assistant" in norm_system:
            return _naive_triple(_between(user, "Claim:", "Return JSON"))
        if "information-extraction assistant" in norm_system:
            return self._extract_passage_triples(user)
        if "verdict head of GraphVerify-Hybrid" in norm_system:
            return self._hybrid_verdict(user)
        if "\"Research\" step of RARR" in norm_system:
            return self._rarr_question(user)
        if "\"Agreement\" (Editor) step of RARR" in norm_system:
            return self._rarr_agreement(user)
        if "one round of FIRE" in norm_system:
            return self._fire_round(user)
        if "citation-repair step of CiteFix" in norm_system:
            return self._citefix_repair(user)
        if "citation-check step of CiteFix" in norm_system:
            claim = _between(user, "Claim:", "Cited passage")
            evidence = _between(user, "Cited passage", "Return JSON")
            return self._entailment_response(claim, evidence)
        if "SAFE (Search-Augmented" in norm_system:
            claim = _between(user, "Atomic fact:", "Search results")
            evidence = _between(user, "Search results", "Rate the fact")
            return self._entailment_response(claim, evidence)
        if "Hybrid fact-checking that integrates knowledge graphs" in norm_system:
            claim = _between(user, "Claim:", "Extracted triple")
            evidence = _between(user, "structural summary:", "Return JSON")
            return self._entailment_response(claim, evidence)
        if "fact-checking assistant. You verify a single factual claim" in norm_system:
            claim = _between(user, "Claim:", "Retrieved evidence passages")
            evidence = _between(user, "Retrieved evidence passages", "Decide whether")
            return self._entailment_response(claim, evidence)
        return {}

    @staticmethod
    def _decompose(user: str) -> Dict[str, List[str]]:
        answer = _after(user, "Answer:")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
        return {"claims": sentences or [answer.strip()]}

    @staticmethod
    def _extract_passage_triples(user: str) -> List[Dict[str, Any]]:
        passage = _after(user, "Passage:")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage.strip()) if s.strip()]
        triples = []
        for sent in sentences:
            t = _naive_triple(sent)
            if t["head"] and t["tail"]:
                triples.append({
                    "subject": t["head"], "relation": t["relation"], "object": t["tail"],
                    "span": sent, "timestamp": None,
                })
        return triples

    @staticmethod
    def _hybrid_verdict(user: str) -> Dict[str, Any]:
        m = re.search(r"Rule-based prior verdict:\s*(\w+)", user)
        verdict = m.group(1) if m else "Unsupported"
        return {"verdict": verdict, "confidence": 0.8, "rationale": "fake: confirmed rule-based prior"}

    @staticmethod
    def _rarr_question(user: str) -> Dict[str, Any]:
        evidence = _between(user, "Evidence passages", "Generate one verification question")
        found = bool(evidence.strip())
        return {
            "question": "Does the evidence support the claim?",
            "answer_from_evidence": evidence.strip()[:200] if found else "",
            "evidence_found": found,
        }

    @staticmethod
    def _rarr_agreement(user: str) -> Dict[str, Any]:
        claim = _between(user, "Claim:", "Verification question")
        answer = _between(user, "Answer obtained from evidence:", "Does the evidence-based answer agree")
        verdict, confidence = _entailment_heuristic(claim, answer)
        return {"verdict": verdict, "confidence": confidence, "rationale": "fake: token-overlap agreement check"}

    @staticmethod
    def _fire_round(user: str) -> Dict[str, Any]:
        claim = _between(user, "Claim:", "Evidence available this round")
        evidence = _between(user, "Evidence available this round", "Return JSON")
        verdict, confidence = _entailment_heuristic(claim, evidence)
        return {
            "verdict": verdict, "confidence": confidence, "confident_enough": True,
            "evidence_ids": _extract_ids(evidence), "rationale": "fake: token-overlap over current round's evidence",
        }

    @staticmethod
    def _citefix_repair(user: str) -> Dict[str, Any]:
        claim = _between(user, "Claim:", "Originally cited passage")
        candidates = _between(user, "Candidate passages to search for a repaired citation", "Return JSON")
        verdict, confidence = _entailment_heuristic(claim, candidates)
        ids = _extract_ids(candidates)
        return {
            "verdict": verdict, "confidence": confidence,
            "repaired_passage_id": ids[0] if (verdict != "Unsupported" and ids) else None,
            "rationale": "fake: token-overlap repair search",
        }

    @staticmethod
    def _entailment_response(claim: str, evidence: str) -> Dict[str, Any]:
        verdict, confidence = _entailment_heuristic(claim, evidence)
        return {"verdict": verdict, "confidence": confidence, "rationale": "fake: token-overlap entailment"}


def _after(text: str, marker: str) -> str:
    """Returns the text following the first occurrence of `marker`, or "" if absent."""
    idx = text.find(marker)
    if idx == -1:
        return ""
    return text[idx + len(marker):].strip()


def _between(text: str, start_marker: str, end_marker: str) -> str:
    """
    Returns the text strictly between the first occurrence of `start_marker`
    and the next occurrence of `end_marker` after it. Falls back to
    everything after `start_marker` if `end_marker` is not found, and to ""
    if `start_marker` itself is absent. Used to isolate one section of a
    prompt template (e.g. just the claim line, not the trailing evidence and
    instructions) when routing a fake LLM response.
    """
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end == -1:
        return text[start:].strip()
    return text[start:end].strip()


class _FakeSentenceTransformer:
    """
    Deterministic, offline stand-in for `sentence_transformers.SentenceTransformer`.
    Encodes each string to a fixed-dimensional vector derived from an MD5
    hash of its normalized text, so identical/near-identical strings get
    identical/similar vectors (needed for entity-linking and path-scoring
    tests that check exact/near matches) without loading any real model.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = np.array([self._vec(t) for t in texts], dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.clip(norms, 1e-12, None)
        return vecs

    def _vec(self, text: str) -> np.ndarray:
        key = re.sub(r"\s+", " ", text.lower().strip())
        rng = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32))
        return rng.standard_normal(self.dim)


def install_fake_embedder(monkeypatch) -> None:
    """
    Monkeypatches `graphverify.embedder._get_model` (a pytest `monkeypatch`
    fixture) so every lazily-constructed `Embedder(...)` anywhere in the
    codebase returns deterministic offline vectors instead of downloading
    `BAAI/bge-base-en-v1.5`. Call this from a fixture/test before exercising
    any code path that might hit the embedding fallback (fuzzy entity
    linking, relation normalization, path scoring, GraphCheck-adapted).
    """
    import graphverify.embedder as embedder_module

    fake_model = _FakeSentenceTransformer()
    monkeypatch.setattr(embedder_module, "_get_model", lambda model_name: fake_model)
