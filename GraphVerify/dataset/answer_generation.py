"""
RAG answer generation: produces the generated answer that GraphVerify (and
every baseline) verifies.

Every dataset loader in `dataset/loader.py` carries a gold QA answer /
claim, not a model-generated answer to audit — that is exactly the
distinction the revision plan requires ("we generated answers using the
same RAG pipeline and retrieved-evidence budget across all methods").
This module is that shared generator: one function, reused by every
experiment script and every method under comparison, so a difference in
verification results is never explained by two methods verifying two
different generated answers for the same query.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from graphverify.llm_client import LLMClient
from graphverify.prompts import load_prompt


def generate_answer(
    llm_client: LLMClient,
    query: str,
    passages: List[Dict[str, Any]],
    max_passages: Optional[int] = None,
    max_passage_chars: int = 500,
) -> str:
    """
    Generates a RAG answer to `query` using only `passages` as context.
    Returns "" if there are no passages to answer from (an empty-context
    generation is not meaningful to verify).
    """
    if not passages:
        return ""

    used = passages[:max_passages] if max_passages else passages
    evidence_block = "\n".join(
        f"[{p.get('id', 'p')}] {str(p.get('text', ''))[:max_passage_chars]}" for p in used
    )
    messages = [
        {"role": "system", "content": load_prompt("answer_generation_system")},
        {"role": "user", "content": load_prompt("answer_generation_user").format(
            query=query, evidence=evidence_block,
        )},
    ]
    return llm_client.chat(messages).strip()


def generate_answers_for_dataset(
    llm_client: LLMClient,
    records: List[Dict[str, Any]],
    max_passages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns a new list of records (shallow copies) with the ``"generated"``
    field filled in via :func:`generate_answer`. Records that already carry
    a non-empty ``"generated"`` value are left unchanged, so this is safe to
    call repeatedly / resume a partially-generated dataset.
    """
    out = []
    for rec in records:
        rec = dict(rec)
        if not rec.get("generated"):
            rec["generated"] = generate_answer(
                llm_client, rec.get("query", ""), rec.get("passages", []), max_passages=max_passages,
            )
        out.append(rec)
    return out
