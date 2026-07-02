from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

try:
    from dotenv import load_dotenv

    # Load `<repo_root>/.env` (sibling of this package) if present, without
    # overriding variables the caller already has set in the environment.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
    pass


# ---------------------------------------------------------------------------
# Path scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
LAMBDA_HEAD: float = 0.30
LAMBDA_REL:  float = 0.25
LAMBDA_TAIL: float = 0.30
LAMBDA_PROV: float = 0.15

assert abs(LAMBDA_HEAD + LAMBDA_REL + LAMBDA_TAIL + LAMBDA_PROV - 1.0) < 1e-9

# ---------------------------------------------------------------------------
# Verdict thresholds
# ---------------------------------------------------------------------------
SUPPORT_THRESHOLD:    float = 0.60
CONTRADICT_THRESHOLD: float = 0.55

# ---------------------------------------------------------------------------
# Matching thresholds
# ---------------------------------------------------------------------------
EMBED_COSINE_CUTOFF: float = 0.75
NUMERIC_TOLERANCE:   float = 0.05   # ±5% for quantity comparisons

# ---------------------------------------------------------------------------
# Graph search
# ---------------------------------------------------------------------------
L_MAX:        int = 3    # max hop depth per path search
TOP_K_PATHS:  int = 20   # candidate paths evaluated per claim
TOP_K_PASSAGES: int = 10

# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = 0.0   # deterministic decoding
LLM_MAX_TOKENS:  int   = 512

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
ECE_BINS: int = 15

# ---------------------------------------------------------------------------
# Text-evidence fallback (evidence_mode="text" / "hybrid")
# ---------------------------------------------------------------------------
TEXT_FALLBACK_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Relation aliases: surface forms that map to a single canonical relation name.
# Extend this table when adding new domains.
# ---------------------------------------------------------------------------
RELATION_ALIASES: Dict[str, List[str]] = {
    "birthPlace":      ["birthplace", "born in", "place of birth", "native of",
                        "birth city", "birth country", "birth location"],
    "deathPlace":      ["died in", "death place", "place of death", "passed away in"],
    "occupation":      ["works as", "profession", "job", "career", "employed as"],
    "nationality":     ["is from", "citizen of", "national of", "native country"],
    "almaMater":       ["studied at", "attended", "graduated from", "alumni of"],
    "employer":        ["works for", "employed by", "hired by", "company"],
    "spouse":          ["married to", "husband of", "wife of", "partner of"],
    "parent":          ["parent of", "father of", "mother of", "children"],
    "child":           ["child of", "son of", "daughter of"],
    "award":           ["won", "received", "awarded", "prize"],
    "genre":           ["type of", "kind of", "classified as", "category"],
    "locatedIn":       ["located in", "situated in", "found in", "based in",
                        "position", "place", "in the city of"],
    "foundedBy":       ["founded by", "established by", "created by", "started by"],
    "foundedDate":     ["founded in", "established in", "created in", "started in"],
    "partOf":          ["part of", "member of", "component of", "belongs to"],
    "hasProperty":     ["has", "contains", "features", "includes"],
    "releaseDate":     ["released in", "release year", "published in", "came out in",
                        "premiere", "debut"],
    "director":        ["directed by", "filmmaker", "helmed by"],
    "author":          ["written by", "authored by", "penned by"],
    "producedBy":      ["produced by", "producer"],
    "language":        ["spoken in", "official language", "written in"],
    "capital":         ["capital of", "capital city"],
    "population":      ["population of", "inhabitants", "residents"],
    "area":            ["area of", "size of", "covers"],
    "height":          ["height of", "tall", "stands at"],
    "age":             ["age of", "years old", "born"],
    "duration":        ["lasts", "duration of", "length of", "runtime"],
    "cost":            ["costs", "price of", "worth", "valued at"],
}

# Relations where (head, relation) uniquely determines the tail value.
# A different tail in the graph signals a contradiction.
FUNCTIONAL_RELATIONS: List[str] = [
    "birthPlace", "deathPlace", "nationality", "foundedDate",
    "releaseDate", "capital", "almaMater", "director", "author",
    "locatedIn", "area", "population",
]

# Relations whose values are dates or years (used for temporal comparison).
TEMPORAL_RELATIONS: List[str] = [
    "birthDate", "deathDate", "foundedDate", "releaseDate",
    "publicationYear", "year",
]

# Label sets where only one value can be true at a time.
MUTUALLY_EXCLUSIVE_SETS: List[List[str]] = [
    ["supports", "refutes", "not enough info"],
    ["true", "false"],
]


@dataclass
class GraphVerifyConfig:
    """
    Runtime configuration for the GraphVerify pipeline. Override fields per
    experiment; every field has a default matching the values reported in
    the method description (see the ``lambda_*``/``*_threshold`` constants
    above, tuned on a held-out development split).

    Three independent axes control which system variant a given config
    produces, and experiment scripts vary exactly one axis at a time:

    ``verdict_mode`` (final-verdict computation):
        - ``"score_only"``: the threshold-based path scorer
          (:mod:`graphverify.path_scorer`, :mod:`graphverify.verdict_assigner`)
          is the sole decision rule. This is GraphVerify-score: fully
          auditable, no LLM in the verdict loop beyond claim/triple
          extraction. Produced by :class:`graphverify.verifier.GraphVerify`.
        - ``"hybrid_llm"``: the score-only verdict is computed first as a
          prior, then :class:`graphverify.hybrid_verdict.HybridVerdictHead`
          reads the claim, triple, and top support/conflict paths and
          confirms or overrides it. This is GraphVerify-hybrid. Produced by
          :class:`graphverify.verifier.HybridGraphVerify`. Use
          :func:`graphverify.verifier.build_graphverify` to construct the
          right class from this field automatically.

    ``evidence_mode`` (what evidence feeds verdict computation, per claim):
        - ``"text"``: no graph/path search at all; each claim is checked
          directly against the concatenated retrieved-passage text via
          :func:`graphverify.text_evidence.text_entailment_verdict`. This is
          the weakest evidence condition in the evidence-composition
          ablation (Table 3 "text evidence" row).
        - ``"retrieved_graph"``: the provenance-linked graph is built only
          from the retrieved passages, and a claim with no support/conflict
          path clearing threshold is Unsupported with no fallback. This is
          the fairest, retrieved-only setting used for the main results.
        - ``"kg_paths"``: the retrieved-passage graph is augmented with
          triples from an external, curated KG file (``external_kg_path``,
          JSONL of ``{"head","relation","tail"}`` records) via
          :func:`graphverify.evidence_graph.merge_external_kg`. If
          ``external_kg_path`` is unset, this degrades to
          ``"retrieved_graph"`` with a runtime warning (there is no silent
          fallback).
        - ``"hybrid"`` (default): ``"kg_paths"`` graph construction, plus a
          text-evidence fallback (see ``text_fallback_threshold``) applied
          only to claims the graph pipeline could not resolve
          (Unsupported). This implements the "textual fallback matching"
          the method description calls out as an extension for claims that
          fail to map to a valid graph triple.

    ``evidence_source`` (which passages the graph is built from, set by the
    caller before invoking ``verify()`` — not branched on inside the
    verifier itself, since the passage pool is a data-loading concern):
        - ``"retrieved_only"``: passages are exactly what the shared
          retriever returned for the query (the fair, no-privileged-
          information main-results setting).
        - ``"corpus_no_gold"``: passages are expanded from the full corpus
          (larger recall) without using any gold/label information.
        - ``"gold_oracle"``: a gold evidence graph or gold passage set is
          supplied directly via the ``graph=`` argument of
          :meth:`GraphVerify.verify`, bypassing extraction entirely. Used
          only for the oracle-pipeline upper-bound experiment
          (``experiments/run_oracle_pipeline_decomposition.py``).
    """
    lambda_head: float = LAMBDA_HEAD
    lambda_rel:  float = LAMBDA_REL
    lambda_tail: float = LAMBDA_TAIL
    lambda_prov: float = LAMBDA_PROV

    support_threshold:    float = SUPPORT_THRESHOLD
    contradict_threshold: float = CONTRADICT_THRESHOLD

    embed_cosine_cutoff: float = EMBED_COSINE_CUTOFF
    numeric_tolerance:   float = NUMERIC_TOLERANCE

    l_max:          int = L_MAX
    top_k_paths:    int = TOP_K_PATHS
    top_k_passages: int = TOP_K_PASSAGES

    llm_backend: Literal["openai", "anthropic", "local"] = "openai"
    llm_model:   str = "gpt-4o-mini"
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens:  int   = LLM_MAX_TOKENS

    local_model_path: str = "Qwen/Qwen2.5-7B-Instruct"

    embed_model: str = "BAAI/bge-base-en-v1.5"

    verdict_mode:  Literal["score_only", "hybrid_llm"] = "score_only"
    evidence_mode: Literal["text", "retrieved_graph", "kg_paths", "hybrid"] = "hybrid"
    evidence_source: Literal["retrieved_only", "corpus_no_gold", "gold_oracle"] = "retrieved_only"

    external_kg_path: Optional[str] = None
    text_fallback_threshold: float = TEXT_FALLBACK_THRESHOLD

    ece_bins: int = ECE_BINS

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])

    # -------------------------------------------------------------------
    # Component-ablation switches (consumed by eval/ablation.py). Each is a
    # real behavioral switch, not a label: see the referenced module's
    # docstring for exactly what changes when it is set.
    # -------------------------------------------------------------------
    disable_claim_decomposition:    bool = False   # see graphverify/verifier.py: GraphVerify.verify()
    disable_relation_normalization: bool = False   # see graphverify/relation_normalizer.py
    entity_match_mode: Literal["exact_only", "exact_alias", "exact_alias_embed", "embed_only"] = "exact_alias_embed"
    # see graphverify/entity_linker.py and graphverify/path_scorer.py
