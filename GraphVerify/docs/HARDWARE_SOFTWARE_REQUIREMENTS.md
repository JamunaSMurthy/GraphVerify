# Hardware and Software Requirements

This document lists what is actually needed to install, test, and run every experiment in
this repository — from a single `pytest` invocation up to the full P0/P1/P2 benchmark suite
across all five datasets. Figures below are measured or conservatively estimated on the
reference configuration described in each section; your mileage will vary with dataset size,
`--max_samples`, and LLM provider latency.

## 1. Software

| Component | Required version | Notes |
|---|---|---|
| Python | 3.9 – 3.12 | Codebase uses `from __future__ import annotations` throughout for 3.9 compatibility. Tested on 3.9 and 3.11. |
| pip | >= 21 | For installing `requirements.txt` / `requirements-dev.txt`. |
| OS | Linux, macOS, or WSL2 on Windows | `faiss-cpu` and `torch` wheels are available for all three; no OS-specific code paths. |
| git | any recent version | Only needed to clone/track the repo. |

Install everything with:

```bash
pip install -r requirements.txt          # runtime
pip install -r requirements-dev.txt      # + pytest, coverage (adds -r requirements.txt)
python -m spacy download en_core_web_sm  # optional NER fallback, not required for tests
```

Optional, only if you run the local Qwen2.5-7B-Instruct backend (`--llm_backend local`):

```bash
pip install accelerate bitsandbytes
```

### External services

| Service | Used for | Required? |
|---|---|---|
| OpenAI API (or any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) | Claim decomposition, triple extraction, evidence-graph construction, GraphVerify-Hybrid verdict head, RAG answer generation, SAFE/RARR/FIRE/CiteFix/Hybrid-KG-LLM/LLM-text-only baselines | Yes, unless you run everything with `--llm_backend local` against a self-hosted model |
| Anthropic API | Optional Claude backbone for `experiments/run_generator_transfer.py` | No — only if you include a Claude generator in the transfer sweep |
| HuggingFace Hub | Dataset downloads (HotpotQA, 2WikiMultiHopQA, MuSiQue, FEVER), `BAAI/bge-base-en-v1.5` embedding model, optional local Qwen2.5-7B-Instruct weights | Yes, for the public datasets and the embedding model (always needed — the embedder is not optional) |
| RAGTruth (GitHub release, fetched by `dataset/loader.py`) | RAGTruth hallucination-detection benchmark | Yes, for RAGTruth experiments only |

Set credentials in `.env` (copy `.env.example` → `.env`); see that file for the full list and
what each key gates.

## 2. Hardware

### 2.1 Running the test suite (`pytest tests/unit`)

No GPU, no network, no API keys. All LLM and embedding calls are replaced by deterministic
fakes (`tests/fakes.py`). Runs in well under a minute on any laptop-class CPU with 2 GB free
RAM. `tests/integration/*` additionally needs network access and real credentials — see
Section 1.

### 2.2 Running the pipeline with the OpenAI backend (`--llm_backend openai`)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| CPU | 2 cores | 4+ cores | Graph search (`PathSearcher`), FAISS index build/query, and metric computation are CPU-bound. |
| RAM | 4 GB | 8–16 GB | `BAAI/bge-base-en-v1.5` (≈440 MB on disk, ~1–2 GB resident with batching) plus FAISS flat indices held in memory (a few hundred MB per dataset at the sizes used here — each passage/entity/relation embedding is 768-dim float32). |
| Disk | 5 GB free | 15 GB free | HuggingFace dataset caches (HotpotQA ≈ 600 MB, 2WikiMultiHopQA ≈ 500 MB, MuSiQue ≈ 200 MB, FEVER ≈ 700 MB, RAGTruth ≈ 50 MB), the embedding model, plus cached evidence graphs / predictions written under `output/`. |
| GPU | None required | None required | The OpenAI backend does no local inference; the embedding model can run on CPU (sentence-transformers auto-detects CUDA/MPS if present and uses it opportunistically, but it is not required). |
| Network | Required | Stable broadband | Every claim decomposition / triple extraction / verdict call is an API round trip. At `top_k_passages=10` and a few claims per answer, expect on the order of 4–8 LLM calls per verified answer. |

### 2.3 Running the local backend (`--llm_backend local`, Qwen2.5-7B-Instruct)

| Resource | Minimum | Recommended | Why |
|---|---|---|---|
| GPU VRAM | 8 GB (4-bit/8-bit quantized via `bitsandbytes`) | 16–24 GB (fp16, no quantization) | Qwen2.5-7B-Instruct at fp16 is ≈15 GB of weights; 4-bit quantization brings this to ≈5–6 GB plus activation overhead. |
| System RAM | 16 GB | 32 GB | Model loading, tokenizer, and `accelerate` device mapping overhead. |
| Disk | 20 GB free | 30 GB free | Model weights (~15 GB fp16) plus dataset/embedding caches from Section 2.2. |
| CPU | 4 cores | 8+ cores | Feeds the GPU, handles graph search / FAISS / metrics as in 2.2. |

CPU-only inference with the local backend is technically possible via `device_map="auto"` but
is impractically slow for anything beyond a handful of samples — use the OpenAI backend or a
GPU for real experiment runs.

## 3. Estimated cost / runtime per experiment

Estimates assume `gpt-4o-mini` pricing/latency, `top_k_passages=10`, `L_max=3` hops,
`top_k_paths=20`, and the five main datasets at moderate sample counts
(`--max_samples 500` per dataset unless noted — full-split runs scale roughly linearly).
Actual LLM cost depends on your provider's current pricing; treat these as order-of-magnitude
planning numbers, not guarantees.

| Script | What it does | Approx. LLM calls | Approx. wall-clock (OpenAI backend, 4 cores) |
|---|---|---|---|
| `graph_build.py` (per dataset, 500 samples) | Extract triples from every retrieved passage | ~500 × avg. passages/sample (≈5,000) | 20–40 min |
| `verify.py` (per dataset × seed, 500 samples) | Full claim-level verification | ~500 × (1 decompose + ~3 claims × (1 extract [+1 hybrid verdict if `verdict_mode=hybrid_llm`])) ≈ 2,500–4,000 | 15–30 min |
| `experiments/run_main_verification_benchmark.py` (all 5 datasets × 11 methods) | P0.2 main comparison table | sum of the above × 11 methods (text-only baselines are cheaper — no graph build step) | Several hours; budget a full day for all datasets at full scale |
| `experiments/run_retrieval_noise_stress_test.py` | P0.4 perturbation sweep | ~5 perturbation conditions × base verify cost | Proportional to `verify.py` × 5 |
| `experiments/run_component_ablation.py` / `eval/ablation.py` | P0.5 component ablation | ~8 variants × base verify cost | Proportional to `verify.py` × 8 |
| `experiments/run_threshold_sensitivity_sweep.py` | P1 threshold grid | 0 additional LLM calls (rescoring cached graphs/paths only) | Minutes, if graphs/paths are already cached |
| `experiments/benchmark_runtime_and_compute.py` | P1 runtime/memory profiling | Small fixed sample (default 50) | < 10 min |
| `experiments/compute_annotation_agreement.py` | P1 IAA from human-provided CSVs | 0 (no LLM calls — pure statistics over annotation files you supply) | Seconds |

Graph-building is the dominant one-time cost per dataset; once `output/graphs/<dataset>/*.jsonl`
is cached, `verify.py` and downstream experiment scripts reuse it via `--graph_dir` and do not
re-extract triples from passages.

## 4. What this document does not cover

Downloading and annotating new gold claim/path labels (for the oracle-pipeline and
human-annotation-agreement experiments) requires human annotators and is a process, not a
hardware requirement — see `docs/ANNOTATION_GUIDELINES.md`.
