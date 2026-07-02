# 🔎 GraphVerify

**Claim-Level Post-Generation Verification for Retrieval-Augmented Generation**

> GraphVerify audits generated RAG answers claim-by-claim using provenance-linked knowledge graphs — catching unsupported inferences and active contradictions that answer-level faithfulness checks miss.

![Datasets](https://img.shields.io/badge/Datasets-5%20benchmarks-e74c3c) ![Paper](https://img.shields.io/badge/Paper-in%20preparation-2ecc71) ![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-yellow) ![Tests](https://img.shields.io/badge/tests-261%20passing-brightgreen) ![LLM Backends](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Anthropic%20%7C%20Local-8e44ad)

---

## ✨ Overview

**GraphVerify** is a research codebase for **claim-level, post-generation verification of retrieval-augmented generation (RAG) answers**. Given a generated answer and the passages that were retrieved for it, GraphVerify:

1. **Decomposes** the answer into atomic, independently-verifiable claims.
2. **Canonicalizes** each claim as a relation triple `(head, relation, tail)`.
3. **Builds** a provenance-linked evidence graph from the same retrieved passages the RAG system used.
4. **Searches** that graph for support paths (evidence that confirms the claim) and conflict paths (evidence that actively contradicts it).
5. **Assigns** one of three verdicts, with a full audit trail — instead of a single opaque faithfulness score.

| Verdict | Meaning | Returned trace |
|---|---|---|
| ✅ **Supported** | Sufficient graph evidence supports the claim | Support path with source spans |
| ❓ **Unsupported** | Retrieved graph lacks enough evidence | Empty / abstention signal |
| ❌ **Contradictory** | Graph contains active conflicting evidence | Conflict path with provenance |

Two verifier variants share the same rule-based path-scoring backbone:

- 🧮 **GraphVerify-score** (`verdict_mode="score_only"`) — the threshold-based path scorer is the entire decision procedure. Fully auditable, no LLM in the verdict loop beyond claim/triple extraction.
- 🤖 **GraphVerify-hybrid** (`verdict_mode="hybrid_llm"`) — the same rule-based pipeline runs first, then an LLM verdict head reads the claim, triple, and top scored support/conflict paths and confirms or overrides the rule-based prior.

The companion baseline suite (`baselines/`) reimplements 9 published/adjacent verification and graph-retrieval methods so GraphVerify can be compared fairly, on the exact same claims and evidence budget, out of the box.

---

## 📚 Datasets

GraphVerify evaluates on five public benchmarks. None are bundled in this repo — `dataset/loader.py` downloads and caches each one automatically on first use (see [Supplying Your Own Data](#-supplying-your-own-data) to bring your own instead).

🔗 **HotpotQA** *(multi-hop QA)*: https://hotpotqa.github.io/

🔗 **2WikiMultiHopQA** *(multi-hop QA)*: https://github.com/Alab-NII/2wikimultihop

🔗 **MuSiQue** *(compositional multi-hop QA)*: https://github.com/StonyBrookNLP/musique

🔗 **FEVER** *(fact verification)*: https://fever.ai/

🔗 **RAGTruth** *(RAG hallucination detection)*: https://github.com/ParticleMedia/RAGTruth

---

## 📖 Table of Contents

- [Datasets](#-datasets)
- [What GraphVerify Supports](#-what-graphverify-supports)
- [Example](#-example)
- [Why GraphVerify](#-why-graphverify)
- [Architecture](#-architecture)
- [Baselines](#-baselines)
- [Experiments](#-experiments)
- [Requirements](#-requirements)
- [API Keys & Environment Setup](#-api-keys--environment-setup)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Running the Test Suite](#-running-the-test-suite)
- [Full Evaluation Pipeline](#-full-evaluation-pipeline)
- [Running the Full Experiment Suite](#-running-the-full-experiment-suite)
- [Supplying Your Own Data](#-supplying-your-own-data)
- [Local Model Inference](#-local-model-inference-no-api-key-needed)
- [Project Structure](#-project-structure)
- [Reproducibility](#-reproducibility)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)

---

## 🌈 What GraphVerify Supports

- ✅ Claim-level, three-way verdicts (Supported / Unsupported / Contradictory) — not just a single faithfulness score
- ✅ Auditable evidence paths returned for every verdict, not just a confidence number
- ✅ Two verifier variants: rule-only (**GraphVerify-score**) and LLM-assisted (**GraphVerify-hybrid**)
- ✅ 9 baseline verifiers out of the box (SAFE, RARR, FIRE, CiteFix, GraphRAG, HippoRAG, GraphCheck, Hybrid KG-LLM, LLM-text-only), all sharing the same claims and evidence budget
- ✅ 5 benchmark datasets — HotpotQA, 2WikiMultiHopQA, MuSiQue, FEVER, RAGTruth
- ✅ OpenAI, Anthropic, or a fully local (Qwen2.5-7B-Instruct) LLM backend
- ✅ Retrieval-noise stress testing, threshold sensitivity sweeps, component ablations, calibration reporting
- ✅ Human-annotation tooling with inter-annotator agreement (Cohen's kappa, Krippendorff's alpha) and adjudication
- ✅ 261 offline unit tests — zero network calls, zero API cost, ~2 seconds to run
- ✅ Every hyperparameter and every LLM prompt externalized to config/`prompts/*.txt` — nothing hardcoded or hidden

---

## 🖼️ Example

Given a generated RAG answer and its retrieved passages:

```python
from graphverify import GraphVerify, GraphVerifyConfig

gv = GraphVerify(GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini"))

output = gv.verify(
    query="When did Albert Einstein win the Nobel Prize and where was he born?",
    passages=[
        {"id": "p1", "text": "Albert Einstein was a theoretical physicist born on March 14, 1879, "
                              "in Ulm, in the Kingdom of Württemberg in the German Empire. He received "
                              "the Nobel Prize in Physics in 1921 for his discovery of the law of the "
                              "photoelectric effect.", "rank": 1, "score": 0.95},
    ],
    answer="Albert Einstein won the Nobel Prize in Physics in 1922. He was born in Berlin, Germany.",
)

for rec in output.records:
    print(f"{rec['verdict']:<14} {rec['claim']}")
```

Because the answer gets the year and the birthplace wrong relative to the retrieved evidence, GraphVerify catches both — with the conflicting evidence path attached to each verdict (illustrative shape; exact scores depend on your LLM backend):

```text
Contradictory  Albert Einstein won the Nobel Prize in Physics in 1922.
    └─ evidence: Einstein --[award]--> Nobel Prize in Physics (1921)   reliability: 0.91

Contradictory  He was born in Berlin, Germany.
    └─ evidence: Einstein --[birthPlace]--> Ulm                        reliability: 0.88
```

Run it yourself:

```bash
export OPENAI_API_KEY=sk-...
python demo.py --show_errors
```

---

## 🌟 Why GraphVerify

- 🕸️ **Graph structure used *after* generation, not just for retrieval.** Most graph-RAG methods use graphs to fetch context. GraphVerify uses graphs to check what the model actually said.
- 🧾 **Auditable by default.** Every verdict comes with a returned evidence path — not just a scalar confidence number.
- ⚖️ **Fair, apples-to-apples baselines.** All 9 baselines verify the exact same shared claim list as GraphVerify (see [Baselines](#-baselines)), so comparisons aren't confounded by decomposition differences.
- 🧪 **No placeholders.** Every config field, every ablation variant, every baseline is a real, tested, working code path — none are declared but silently unimplemented.
- 🔓 **Bring your own LLM.** OpenAI, Anthropic, or a fully local Qwen2.5-7B-Instruct backend — swap with one config field.

## 🏗️ Architecture

```
GraphVerify/
├── graphverify/               🧠 Core library
│   ├── config.py              ⚙️  All hyperparameters and behavior switches (GraphVerifyConfig)
│   ├── llm_client.py          🔌 LLM abstraction: OpenAI / Anthropic / local Qwen2.5-7B-Instruct
│   ├── embedder.py            📐 BGE-base-en-v1.5 embedding wrapper (lazy-loaded, swappable)
│   ├── prompts.py             📄 Loads prompts/*.txt templates
│   ├── claim_decomposer.py    ✂️  Atomic claim decomposition
│   ├── triple_extractor.py    🔺 Canonical (head, relation, tail) triple extraction
│   ├── entity_linker.py       🔗 Exact / alias / embedding entity linking
│   ├── relation_normalizer.py 🏷️  Surface relation → canonical form
│   ├── evidence_graph.py      🕸️  Provenance-linked graph construction + external-KG merge
│   ├── path_searcher.py       🔍 Support/conflict path search (hop-limited, top-K)
│   ├── path_scorer.py         📊 Path scoring: entity/relation/provenance agreement
│   ├── incompatibility.py     ⚔️  Conflict detection + taxonomy classification
│   ├── verdict_assigner.py    🧮 Three-way verdict assignment from path scores
│   ├── text_evidence.py       📝 Text-only entailment check (fallback path)
│   ├── hybrid_verdict.py      🤖 LLM verdict head for GraphVerify-hybrid
│   ├── calibrator.py          🌡️  Temperature scaling, ECE computation
│   └── verifier.py            🚀 GraphVerify / HybridGraphVerify / build_graphverify()
├── baselines/                 🧩 9 baseline verifiers + shared interface
├── dataset/
│   ├── loader.py               📚 HotpotQA, 2WikiMultiHopQA, MuSiQue, FEVER, RAGTruth + schema validator
│   ├── retriever.py            🔎 Dense retriever (FAISS) / pass-through retriever
│   ├── answer_generation.py    ✍️  Shared RAG answer generator
│   ├── stress_test.py          🌪️  Retrieval-noise perturbations
│   ├── claim_annotation.py     🧑‍🔬 Cohen's kappa, Krippendorff's alpha, adjudication
│   └── splits.py               ✂️  Deterministic dev/test split construction
├── eval/
│   ├── metrics.py               📈 Claim Acc, per-class F1, macro-F1, Path Corr, ECE, AUROC/AUPRC, bootstrap CI
│   ├── significance.py          📉 Paired bootstrap significance testing with effect size
│   ├── contradiction_taxonomy.py ⚔️  Classifies Contradictory verdicts by mechanism
│   ├── coverage_reliability.py   🎯 Answer-level accept/reject policy, EM/F1, hallucination P/R
│   ├── calibration_report.py     🌡️  Per-class ECE tables + reliability diagrams
│   ├── error_analysis.py         🔬 Sampled error categorization
│   ├── evaluate.py               📊 Multi-dataset evaluation driver
│   └── ablation.py               🧪 Component ablation (real config-field-backed variants)
├── experiments/                🧫 17 runnable, accurately-named experiment entrypoints
├── prompts/                    💬 Every LLM prompt used anywhere, as plain text
├── tests/
│   ├── fakes.py                 🎭 Deterministic offline LLM/embedder test doubles
│   ├── unit/                    ✅ Full offline test suite — no network, no API keys
│   └── integration/             🌐 Real API/model tests, gated by RUN_INTEGRATION_TESTS=1
├── docs/
│   ├── HARDWARE_SOFTWARE_REQUIREMENTS.md   🖥️  What you need to run each stage
│   ├── REPRODUCIBILITY.md                  🔁 Maps every result to its exact command
│   └── ANNOTATION_GUIDELINES.md            📝 Claim-level labeling protocol
├── graph_build.py              🕸️  Build & cache evidence graphs
├── verify.py                   ✅ Run verification on a dataset
├── evaluate.py                 📊 Evaluate predictions
├── calibrate.py                🌡️  Fit temperature calibrator
├── run_all_seeds.sh            🚀 Full evaluation pipeline
└── demo.py                     👋 Quick end-to-end demo
```

## 🧩 Baselines

Nine methods compared against GraphVerify, each an **independent, from-description reimplementation** used as an evaluation control (see each module's docstring for the exact published method it approximates and how it differs — no baseline is presented as more "native" than it actually is):

| Method | Category | Uses graph? |
|---|---|---|
| 🔍 SAFE | native post-hoc verifier | no |
| 🔬 RARR | native post-hoc verifier | no |
| 🔄 FIRE | native post-hoc verifier | no |
| 🩹 CiteFix | native post-hoc verifier | no |
| 🕸️ GraphRAG *(adapted)* | adapted graph-retrieval control | yes |
| 🦛 HippoRAG *(adapted)* | adapted graph-retrieval control | yes |
| ✅ GraphCheck *(adapted)* | KG fact-checking | yes |
| 🤝 Hybrid KG-LLM | KG fact-checking | yes |
| 💬 LLM-text-only verifier | ablation control | no |

The GraphRAG/HippoRAG adaptations reuse GraphVerify's own path search and verdict assignment (a **shared verdict head**) — only their retrieval strategy differs — so any performance gap is attributable to retrieval, never to a separately implemented scoring rule.

## 🧫 Experiments

Every experiment below is a real, runnable CLI script under `experiments/`. Full commands and prerequisites for each: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

| Script | What it measures |
|---|---|
| `build_dataset_statistics.py` | 📚 Per-dataset counts: answers, claims, verdict distribution, manual labels, path labels |
| `run_main_verification_benchmark.py` | 🏆 Claim accuracy / F1 / macro-F1 / path correctness / ECE for all 11 methods, with bootstrap CIs |
| `run_per_dataset_comparison.py` | 📊 Per-dataset breakdown + significance test vs. the strongest baseline |
| `run_retrieval_noise_stress_test.py` | 🌪️ Robustness under top-k changes, distractors, bridge-evidence removal, entity/numeric noise |
| `run_component_ablation.py` | 🧪 What each pipeline component (decomposition, normalization, provenance, contradiction check, matching tier) actually contributes |
| `run_contradiction_taxonomy_breakdown.py` | ⚔️ Why claims were flagged Contradictory (entity, relation, numeric, temporal, multi-hop, mutually-exclusive) |
| `collect_qualitative_examples.py` | 🔎 Sampled real examples across all three verdicts, with evidence traces |
| `run_label_efficiency_experiment.py` | 🏷️ How little labeled dev data the thresholds actually need |
| `run_cross_dataset_threshold_transfer.py` | 🔀 Whether thresholds tuned on one dataset generalize to others |
| `run_threshold_sensitivity_sweep.py` | 🌡️ Full response surface over the threshold grid |
| `run_oracle_pipeline_decomposition.py` | 🎯 Upper bound with gold claims/graph, isolating verifier loss from pipeline-stage loss |
| `benchmark_runtime_and_compute.py` | ⏱️ Wall-clock latency and memory vs. the base RAG pipeline |
| `compute_annotation_agreement.py` | 🧑‍🔬 Cohen's kappa / Krippendorff's alpha from real human annotation files |
| `run_generator_transfer.py` | 🔁 Do results hold across different answer-generator backbones? |
| `run_retriever_transfer.py` | 🔎 Do results hold across different embedding-model retrievers? |
| `run_longform_citation_evaluation.py` | 📰 Long-form (Summary/Data2txt) hallucination detection on RAGTruth |
| `run_answer_level_reliability.py` | 🎯 Coverage/abstention curves: accepted-answer %, EM/F1, hallucination precision/recall |

## 🖥️ Requirements

Full details, per-stage hardware needs, and per-experiment cost estimates live in [`docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`](docs/HARDWARE_SOFTWARE_REQUIREMENTS.md). Short version:

| | Minimum | Notes |
|---|---|---|
| 🐍 Python | 3.9 – 3.12 | Tested on 3.9 and 3.11 |
| 🖥️ CPU | 2 cores | 4+ recommended |
| 🧠 RAM | 4 GB | 8–16 GB recommended (embedding model + FAISS indices) |
| 💾 Disk | 5 GB free | Up to 15 GB with all five dataset caches |
| 🎮 GPU | Not required | Only needed for the optional local Qwen2.5-7B-Instruct backend (8 GB+ VRAM quantized, 16–24 GB fp16) |
| 🌐 Network | Required | Dataset downloads, embedding-model download, and every OpenAI/Anthropic API call |

## 🔑 API Keys & Environment Setup

```bash
cp .env.example .env
```

Then fill in `.env`:

| Variable | Required? | Used for | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ **Yes**, unless using `--llm_backend local` | Claim decomposition, triple extraction, evidence-graph construction, GraphVerify-Hybrid, RAG answer generation, most text-driven baselines | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `OPENAI_BASE_URL` | ⬜ Optional | Point at a self-hosted OpenAI-compatible endpoint (e.g. local vLLM) instead of api.openai.com | — |
| `ANTHROPIC_API_KEY` | ⬜ Optional | Only if you use `--llm_backend anthropic` or include a Claude model in `run_generator_transfer.py` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `HF_TOKEN` | ⬜ Optional | Avoids anonymous rate limits on HuggingFace dataset/model downloads (public datasets used here don't require it) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `RUN_INTEGRATION_TESTS` | ⬜ Optional | Set to `1` to enable `tests/integration/` (real API + real embedding-model download) | — |

> 🔒 `.env` is git-ignored — never commit real keys. Only `.env.example` (with placeholder values) is tracked.

No key at all? You can still run the **entire offline test suite** (`pytest tests/unit`) and every module's core logic — the fake LLM/embedder test doubles in `tests/fakes.py` make the whole pipeline runnable with zero network access for development and CI.

## 📥 Installation

```bash
# 1. Clone and enter the repo
cd GraphVerify

# 2. (Recommended) create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# 4. Optional: NER fallback model
python -m spacy download en_core_web_sm

# 5. Set up your environment file
cp .env.example .env
# → edit .env and add your OPENAI_API_KEY

# 6. Verify the install
pytest tests/unit -q
```

If the last command prints something like `261 passed`, you're fully set up. ✅

## 🚀 Quick Start

```bash
export OPENAI_API_KEY=sk-...
python demo.py
python demo.py --show_errors   # also demonstrates contradiction detection
```

### Python API

```python
from graphverify import GraphVerify, GraphVerifyConfig

cfg = GraphVerifyConfig(llm_backend="openai", llm_model="gpt-4o-mini")
gv  = GraphVerify(cfg)

output = gv.verify(
    query="When did Einstein win the Nobel Prize?",
    passages=[
        {"id": "p1", "text": "Albert Einstein received the Nobel Prize in Physics in 1921.", "rank": 1, "score": 0.95},
    ],
    answer="Albert Einstein won the Nobel Prize in 1921.",
)

for rec in output.records:
    print(rec["claim"], "→", rec["verdict"])
```

Swap in the LLM-assisted hybrid verdict head:

```python
from graphverify import build_graphverify, GraphVerifyConfig

gv = build_graphverify(GraphVerifyConfig(verdict_mode="hybrid_llm"))
```

## ✅ Running the Test Suite

```bash
# Offline, deterministic, no API keys — runs in ~2 seconds
pytest tests/unit -q

# Real API + real embedding-model download (costs money, needs network)
RUN_INTEGRATION_TESTS=1 pytest tests/integration -m integration

# With coverage
pytest tests/unit --cov=graphverify --cov=baselines --cov=dataset --cov=eval --cov=experiments
```

The offline suite (`tests/unit/`) exercises every production module — core pipeline, all 9 baselines, all metrics, all experiment-script core functions — against deterministic fake LLM/embedder test doubles (`tests/fakes.py`), so it never makes a network call and never costs anything to run.

## 🔬 Full Evaluation Pipeline

### Step 1 — Build evidence graphs

```bash
python graph_build.py \
    --dataset hotpotqa --split validation \
    --output_dir output/graphs/hotpotqa \
    --llm_backend openai --llm_model gpt-4o-mini
```

### Step 2 — Run verification

```bash
python verify.py \
    --dataset hotpotqa --split validation \
    --graph_dir output/graphs/hotpotqa \
    --output_dir output/predictions/hotpotqa
```

### Step 3 — Calibrate

```bash
python calibrate.py \
    --pred_dir output/predictions/hotpotqa \
    --dataset hotpotqa --split validation --seed 0 \
    --output_dir output/calibrators
```

### Step 4 — Evaluate

```bash
python evaluate.py \
    --pred_dir output/predictions/hotpotqa \
    --dataset hotpotqa --split validation
```

Or run the whole thing in one shot:

```bash
./run_all_seeds.sh
```

## 🧪 Running the Full Experiment Suite

Compare GraphVerify against all 9 baselines on every dataset:

```bash
python experiments/run_main_verification_benchmark.py \
    --datasets hotpotqa,2wikimultihopqa,musique,fever,ragtruth \
    --split validation --max_samples 200
```

Then, for the deeper analyses:

```bash
python experiments/run_per_dataset_comparison.py --datasets hotpotqa,fever
python experiments/run_retrieval_noise_stress_test.py --datasets hotpotqa
python experiments/run_component_ablation.py --dataset hotpotqa
python experiments/run_contradiction_taxonomy_breakdown.py \
    --predictions_dir output/results/main_benchmark --datasets hotpotqa --methods graphverify_hybrid
```

📖 Every script, its exact command, and what it needs (API keys, prior-stage outputs, human annotation) is documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — start there before running anything you haven't run before.

## 📂 Supplying Your Own Data

Place JSONL files in a local data directory using the unified schema:

```json
{
  "id": "...",
  "query": "...",
  "answer": "...",
  "generated": "...",
  "passages": [{"id": "...", "text": "...", "rank": 1, "score": 0.9}],
  "gold_verdict": "Supported|Unsupported|Contradictory",
  "gold_path": "Entity → relation → Entity",
  "label": "..."
}
```

Then pass `--data_dir /path/to/data` to `graph_build.py`, `verify.py`, or any `experiments/` script.

```python
from dataset.loader import load_dataset, validate_schema

samples = load_dataset("hotpotqa", split="validation", data_dir="/path/to/data")
errors = validate_schema(samples)
assert not errors, errors   # catch schema drift before running anything expensive
```

## 🖥️ Local Model Inference (no API key needed)

```bash
pip install accelerate bitsandbytes
python verify.py \
    --llm_backend local \
    --local_model Qwen/Qwen2.5-7B-Instruct \
    --dataset hotpotqa --split validation
```

Needs a GPU with 8 GB+ VRAM (4-bit/8-bit quantized) or 16–24 GB (fp16) — see [`docs/HARDWARE_SOFTWARE_REQUIREMENTS.md`](docs/HARDWARE_SOFTWARE_REQUIREMENTS.md) for details.

## 🗂️ Project Structure

See [Architecture](#-architecture) above for the annotated tree. Key entry points at a glance:

| I want to... | Look here |
|---|---|
| Run a single verification call | `demo.py`, `graphverify/verifier.py` |
| Change hyperparameters or behavior | `graphverify/config.py` (`GraphVerifyConfig`) |
| Add or inspect a baseline | `baselines/` |
| Add a new dataset | `dataset/loader.py` (`load_dataset`, `validate_schema`) |
| Compute a metric | `eval/metrics.py` |
| Run a full experiment | `experiments/` (see table above) |
| Understand a prompt | `prompts/*.txt` |
| Write/run a test | `tests/unit/`, `tests/fakes.py` |

## 🔁 Reproducibility

Every result this codebase can produce maps to one script and one command — see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Nothing is generated by manual spreadsheet editing. Prompts live in `prompts/` as plain text, hyperparameters live in `graphverify/config.py`, and split files are produced deterministically by `dataset/splits.py` — there is no hidden configuration anywhere in the pipeline.

## 🩺 Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `openai.OpenAIError: Missing credentials` | `OPENAI_API_KEY` isn't set — check `.env` was copied and filled in, or `export OPENAI_API_KEY=...` |
| Tests hang or try to hit the network | You're running `tests/integration/` without meaning to — use `pytest tests/unit` for the offline suite |
| `ModuleNotFoundError` for `graphverify`/`baselines`/etc. | Run from the repo root, or make sure `pip install -r requirements.txt` completed; pytest picks up the repo root automatically via `pyproject.toml`'s `pythonpath` |
| RAGTruth loading is slow the first time | It downloads and caches `dataset/data/ragtruth_*.jsonl` (~35 MB total) on first use; subsequent loads are instant |
| `evidence_mode='kg_paths'` warning | You set `evidence_mode="kg_paths"` without `external_kg_path` — either supply one or use `evidence_mode="hybrid"`/`"retrieved_graph"` |

## 📚 Citation

```bibtex
@inproceedings{graphverify2026,
  title  = {GraphVerify: Claim-Level Post-Generation Verification for Retrieval-Augmented Generation},
  year   = {2026},
}
```

---

<p align="center">Made for reproducible, auditable RAG verification research. 🕸️✅</p>
