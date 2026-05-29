from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal


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
    """Runtime configuration. Override fields per experiment."""
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

    llm_backend: Literal["openai", "local"] = "openai"
    llm_model:   str = "gpt-4o-mini"
    llm_temperature: float = LLM_TEMPERATURE
    llm_max_tokens:  int   = LLM_MAX_TOKENS

    local_model_path: str = "Qwen/Qwen2.5-7B-Instruct"

    embed_model: str = "BAAI/bge-base-en-v1.5"

    evidence_mode: Literal["text", "retrieved_graph", "kg_paths", "hybrid"] = "hybrid"

    ece_bins: int = ECE_BINS

    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
