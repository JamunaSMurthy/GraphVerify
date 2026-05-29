"""
Quick end-to-end demonstration of GraphVerify.

Verifies a small hard-coded example so you can confirm the pipeline
is working before running the full benchmarks.

Usage:
  export OPENAI_API_KEY=sk-...
  python demo.py

  # Local model:
  python demo.py --llm_backend local --local_model Qwen/Qwen2.5-7B-Instruct

  # Also run the incorrect-answer example (demonstrates contradiction detection):
  python demo.py --show_errors
"""
from __future__ import annotations

import argparse

from graphverify import GraphVerify, GraphVerifyConfig


PASSAGES = [
    {
        "id":    "wiki_einstein_1",
        "text":  "Albert Einstein was a theoretical physicist born on March 14, 1879, "
                 "in Ulm, in the Kingdom of Württemberg in the German Empire. "
                 "He received the Nobel Prize in Physics in 1921 for his discovery "
                 "of the law of the photoelectric effect.",
        "rank":  1,
        "score": 0.95,
    },
    {
        "id":    "wiki_einstein_2",
        "text":  "Einstein's work on the photoelectric effect provided crucial "
                 "experimental evidence for quantum theory. The Nobel Committee "
                 "awarded him the prize in 1921, though he had theorized it in 1905.",
        "rank":  2,
        "score": 0.82,
    },
    {
        "id":    "wiki_einstein_3",
        "text":  "Einstein published four groundbreaking works in 1905. "
                 "His birthdate is disputed in some sources as 1880, "
                 "though most historians agree on 1879.",
        "rank":  3,
        "score": 0.61,
    },
]

CORRECT_EXAMPLE = {
    "query":   "When did Albert Einstein win the Nobel Prize and where was he born?",
    "answer":  "Albert Einstein won the Nobel Prize in Physics in 1921. "
               "He was born in Ulm, Germany in 1879.",
    "passages": PASSAGES,
}

INCORRECT_EXAMPLE = {
    "query":   "When did Einstein win the Nobel Prize?",
    "answer":  "Albert Einstein won the Nobel Prize in Physics in 1922. "
               "He was born in Berlin, Germany.",
    "passages": PASSAGES,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_backend",  type=str, default="openai", choices=["openai", "local"])
    p.add_argument("--llm_model",    type=str, default="gpt-4o-mini")
    p.add_argument("--local_model",  type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--show_errors",  action="store_true")
    return p.parse_args()


def print_result(out, title: str) -> None:
    SYMBOLS = {"Supported": "✓", "Unsupported": "?", "Contradictory": "✗"}
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Query:  {out.query[:80]}")
    print(f"  Answer: {out.answer[:100]}")
    print(f"\n  {len(out.records)} claim(s) verified:")
    for rec in out.records:
        symbol = SYMBOLS.get(rec["verdict"], " ")
        claim  = rec["claim"][:50]
        path   = rec.get("best_path") or []
        path_str = " → ".join(
            f"{e.get('src_label','')}→{e.get('dst_label','')}"
            for e in (path[:1] if path else [])
        )
        print(f"  {symbol} {claim:<52} {rec['verdict']:<15} {rec['reliability']:.3f}")
        if path_str:
            print(f"      └─ {path_str[:70]}")
    print(f"\n  {out.n_supported} supported  |  "
          f"{out.n_unsupported} unsupported  |  "
          f"{out.n_contradictory} contradictory")
    print(f"  Graph: {out.graph_stats['n_nodes']} nodes, {out.graph_stats['n_edges']} edges")


def main():
    args = parse_args()
    cfg  = GraphVerifyConfig(llm_backend=args.llm_backend, llm_model=args.llm_model,
                             local_model_path=args.local_model)
    gv   = GraphVerify(cfg)

    out = gv.verify(**CORRECT_EXAMPLE)
    print_result(out, "Correct answer")

    if args.show_errors:
        out2 = gv.verify(**INCORRECT_EXAMPLE)
        print_result(out2, "Answer with errors (should show contradictions)")


if __name__ == "__main__":
    main()
