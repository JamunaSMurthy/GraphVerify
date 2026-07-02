"""
Deterministic, seeded perturbations of a retrieved-passage list, used by
``experiments/run_retrieval_noise_stress_test.py`` to test whether
GraphVerify's provenance-linked graph evidence is actually more robust to
noisy retrieval than a text-only verifier, rather than just performing
better on clean retrieval.

Every function here is pure: it takes a passage list (and, for a few
functions, dataset-specific context a caller must supply) and returns a
*new* list; none mutate their input. All randomness is seeded so a given
`(passages, seed)` pair always produces the same perturbed output.
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, List

_CAP_SPAN_RE = re.compile(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){0,3}\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _renumber(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(passages, start=1):
        p = dict(p)
        p["rank"] = i
        p.setdefault("score", 1.0 / i)
        out.append(p)
    return out


def perturb_top_k(passages: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Restricts the passage list to the top `k` by existing rank, simulating a
    smaller/larger retriever budget. No randomness involved — this tests
    sensitivity to evidence *quantity*, not noise.
    """
    ranked = sorted(passages, key=lambda p: p.get("rank", 10**9))
    return _renumber(ranked[:k])


def inject_distractor_passages(
    passages: List[Dict[str, Any]],
    distractor_pool: List[Dict[str, Any]],
    n: int = 2,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Inserts `n` passages sampled from `distractor_pool` (typically passages
    retrieved for *other* queries in the same dataset, i.e. topically
    unrelated to the current query) at random positions in `passages`,
    simulating retrieval noise where the retriever surfaces irrelevant
    context alongside the true evidence.
    """
    rng = random.Random(seed)
    if not distractor_pool:
        return _renumber(list(passages))

    n = min(n, len(distractor_pool))
    distractors = rng.sample(distractor_pool, n)

    combined = list(passages)
    for d in distractors:
        d = dict(d)
        d["is_distractor"] = True
        insert_at = rng.randint(0, len(combined))
        combined.insert(insert_at, d)
    return _renumber(combined)


def remove_bridge_evidence(
    passages: List[Dict[str, Any]],
    bridge_titles: List[str],
) -> List[Dict[str, Any]]:
    """
    Removes every passage whose ``title`` field matches one of
    `bridge_titles`, simulating a retriever that fails to surface the
    connecting ("bridge") evidence a multi-hop question needs before the
    second reasoning hop.

    `bridge_titles` must be supplied by the caller from dataset annotation
    (e.g. the first-hop title in HotpotQA's ``supporting_facts``) — which
    passage constitutes "the bridge" is a property of the question's
    reasoning chain, not something this function can infer from passage
    text alone. Raises ValueError on an empty list rather than silently
    returning the input unchanged, since that would look like a no-op
    stress condition succeeded when it never removed anything.
    """
    if not bridge_titles:
        raise ValueError(
            "remove_bridge_evidence requires bridge_titles from dataset annotation "
            "(e.g. supporting_facts titles); refusing to silently no-op."
        )
    bridge_set = set(bridge_titles)
    kept = [p for p in passages if p.get("title") not in bridge_set]
    return _renumber(kept)


def inject_entity_alias_noise(
    passages: List[Dict[str, Any]],
    seed: int = 0,
    noise_rate: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Replaces a random subset of capitalized entity-like spans (regex
    heuristic — consecutive capitalized tokens, up to 4 words) in each
    passage's text with an alias variant: either an acronym of its initials
    ("United States" -> "US") or the span with a middle token dropped
    ("Albert Einstein Jr." -> "Albert Jr."). This has no dependency on a
    curated alias gazetteer, so it is dataset-agnostic; it stresses whether
    entity linking (`graphverify.entity_linker.EntityLinker`) degrades
    gracefully under alias mismatch instead of silently losing recall.

    `noise_rate` is the probability, per detected span, that it is
    perturbed (default 0.5 so a passage with several entity mentions still
    keeps some exact matches, avoiding a degenerate all-or-nothing test).
    """
    rng = random.Random(seed)
    out = []
    for p in passages:
        text = str(p.get("text", ""))
        spans = list(_CAP_SPAN_RE.finditer(text))
        new_text = text
        for m in reversed(spans):  # reversed so earlier index replacements stay valid
            if rng.random() >= noise_rate:
                continue
            span_text = m.group()
            tokens = span_text.split()
            if len(tokens) < 2:
                continue
            if rng.random() < 0.5:
                alias = "".join(t[0] for t in tokens)  # acronym
            else:
                drop_idx = rng.randrange(len(tokens))
                alias = " ".join(t for i, t in enumerate(tokens) if i != drop_idx)
            new_text = new_text[: m.start()] + alias + new_text[m.end():]
        p = dict(p)
        p["text"] = new_text
        out.append(p)
    return out


def corrupt_numeric_and_date_mentions(
    passages: List[Dict[str, Any]],
    seed: int = 0,
    corruption_rate: float = 0.5,
    year_delta_range: int = 5,
    numeric_delta_fraction: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Perturbs a random subset of year mentions (by ±1..`year_delta_range`
    years) and other numeric mentions (by up to
    ±`numeric_delta_fraction` of their value) in each passage's text.
    Tests whether contradiction detection
    (:func:`graphverify.incompatibility.classify_incompatibility`) correctly
    flags evidence that now conflicts with an (unperturbed) claim, and
    whether numeric tolerance (`GraphVerifyConfig.numeric_tolerance`) avoids
    flagging small, non-corrupted rounding differences as contradictions.
    """
    rng = random.Random(seed)
    out = []
    for p in passages:
        text = str(p.get("text", ""))
        text = _corrupt_years(text, rng, corruption_rate, year_delta_range)
        text = _corrupt_numbers(text, rng, corruption_rate, numeric_delta_fraction)
        p = dict(p)
        p["text"] = text
        out.append(p)
    return out


def _corrupt_years(text: str, rng: random.Random, rate: float, delta_range: int) -> str:
    def repl(m: re.Match) -> str:
        if rng.random() >= rate:
            return m.group()
        year = int(m.group())
        delta = rng.choice([d for d in range(-delta_range, delta_range + 1) if d != 0])
        return str(year + delta)
    return _YEAR_RE.sub(repl, text)


def _corrupt_numbers(text: str, rng: random.Random, rate: float, delta_fraction: float) -> str:
    def repl(m: re.Match) -> str:
        # Skip bare 4-digit tokens in the plausible year range: those are
        # handled independently by `_corrupt_years` so the two corruption
        # types stay separately interpretable instead of compounding on the
        # same span.
        if _YEAR_RE.fullmatch(m.group()):
            return m.group()
        if rng.random() >= rate:
            return m.group()
        value = float(m.group())
        delta = value * delta_fraction * rng.choice([-1, 1])
        new_value = value + delta
        return str(int(round(new_value))) if "." not in m.group() else f"{new_value:.2f}"
    return _NUMBER_RE.sub(repl, text)
