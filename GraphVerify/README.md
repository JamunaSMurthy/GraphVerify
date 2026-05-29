# GraphVerify

**Claim-Level Post-Generation Verification for Retrieval-Augmented Generation**

> GraphVerify audits generated RAG answers claim-by-claim using provenance-linked knowledge graphs.

## Overview

GraphVerify is a post-generation verifier: given a RAG answer, it decomposes it into atomic claims, canonicalizes each as a relation triple `(h, r, t)`, and checks it against a provenance-linked evidence graph built from the same retrieved passages used by the RAG system.

Each claim receives one of three verdicts:
| Verdict | Meaning | Returned trace |
|---|---|---|
| ✓ Supported | Sufficient graph evidence supports the claim | Support path with source spans |
| ? Unsupported | Retrieved graph lacks enough evidence | Empty / abstention signal |
| ✗ Contradictory | Graph contains active conflicting evidence | Conflict path with provenance |

## Architecture

```
GraphVerify/
├── graphverify/          Core library
│   ├── config.py         Hyperparameters (λ, θ from paper footnote 1)
│   ├── llm_client.py     LLM abstraction (OpenAI API / local Qwen2.5-7B)
│   ├── embedder.py       BGE-base-en-v1.5 embedding wrapper
│   ├── claim_decomposer.py  Γ = D(ŷ) — atomic claim decomposition (Eq. 3)
│   ├── triple_extractor.py  τ_i = T(γ_i) = (h,r,t) canonical triple (Eq. 4)
│   ├── entity_linker.py     Exact / alias / embedding entity linking
│   ├── relation_normalizer.py  Surface relation → canonical form
│   ├── evidence_graph.py    G_q = (V,E,R,P) provenance-linked graph
│   ├── path_searcher.py     Top-20 support/conflict paths (L_max=3 hops)
│   ├── path_scorer.py       s(p,τ) = λ_e·a_h + λ_r·a_r + λ_t·a_t + λ_p·a_p (Eq. 5)
│   ├── incompatibility.py   Incompatibility predicate c(p,τ)
│   ├── verdict_assigner.py  Three-way verdict assignment (Eq. 7)
│   ├── calibrator.py        Temperature scaling, ECE computation (Eq. 8)
│   └── verifier.py          Main pipeline (GraphVerify class)
├── dataset/
│   ├── loader.py         HotpotQA, 2WikiMultiHopQA, MuSiQue, FEVER, RAGTruth
│   └── retriever.py      BGE-base dense retriever (FAISS)
├── eval/
│   ├── metrics.py        Claim Acc, Unsupp F1, Contr F1, Path Corr, ECE
│   ├── evaluate.py       Multi-dataset evaluation (Table 1/2)
│   └── ablation.py       Component ablation (Table 5)
├── graph_build.py        Build & cache evidence graphs
├── verify.py             Run verification on a dataset
├── evaluate.py           Evaluate predictions
├── calibrate.py          Fit temperature calibrator
├── run_all_seeds.sh      Full experiment pipeline (3 seeds)
└── demo.py               Quick end-to-end demo
```

## Key Hyperparameters (paper footnote 1)

| Parameter | Value | Meaning |
|---|---|---|
| λ_e = λ_t | 0.30 | Head / tail entity agreement weight |
| λ_r | 0.25 | Relation agreement weight |
| λ_p | 0.15 | Provenance confidence weight |
| θ_s | 0.60 | Support threshold |
| θ_c | 0.55 | Contradiction threshold |
| L_max | 3 | Max graph search hops |
| top-K paths | 20 | Candidate paths per claim |
| top-k passages | 10 | Retriever budget per query |
| embed cosine cutoff | 0.75 | Embedding match threshold |
| numeric tolerance | ±5% | Numeric incompatibility tolerance |

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, for NER fallback
```

## Quick Start

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

## Full Evaluation Pipeline

### Step 1: Build evidence graphs

```bash
python graph_build.py \
    --dataset hotpotqa --split validation \
    --output_dir output/graphs/hotpotqa \
    --llm_backend openai --llm_model gpt-4o-mini
```

### Step 2: Run verification (3 seeds)

```bash
for SEED in 0 1 2; do
  python verify.py \
      --dataset hotpotqa --split validation \
      --graph_dir output/graphs/hotpotqa \
      --output_dir output/predictions/hotpotqa \
      --seed $SEED
done
```

### Step 3: Calibrate

```bash
python calibrate.py \
    --pred_dir output/predictions/hotpotqa \
    --dataset hotpotqa --split validation --seed 0 \
    --output_dir output/calibrators
```

### Step 4: Evaluate (Table 1 format)

```bash
python evaluate.py \
    --pred_dir output/predictions/hotpotqa \
    --dataset hotpotqa --split validation \
    --seeds 0,1,2
```

### Or run everything at once

```bash
./run_all_seeds.sh
```

### Component ablation (Table 5)

```bash
python eval/ablation.py \
    --dataset hotpotqa --split validation \
    --graph_dir output/graphs/hotpotqa
```

## Supplying Your Own Datasets / Baselines

Place JSONL files in a local data directory using the unified schema:

```json
{
  "id": "...",
  "query": "...",
  "answer": "...",
  "generated": "...",
  "passages": [{"id":"...", "text":"...", "rank":1, "score":0.9}],
  "gold_verdict": "Supported|Unsupported|Contradictory",
  "gold_path": "Entity → relation → Entity",
  "label": "..."
}
```

Then pass `--data_dir /path/to/data` to `graph_build.py`, `verify.py`, etc.

## Local Model (Qwen2.5-7B-Instruct)

```bash
pip install accelerate bitsandbytes
python verify.py \
    --llm_backend local \
    --local_model Qwen/Qwen2.5-7B-Instruct \
    --dataset hotpotqa --split validation
```

## Citation

```bibtex
@inproceedings{graphverify2026,
  title  = {GraphVerify: Claim-Level Post-Generation Verification for Retrieval-Augmented Generation},
  year   = {2026},
}
```
