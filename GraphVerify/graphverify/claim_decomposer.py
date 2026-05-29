"""Decomposes a free-form answer into atomic, independently verifiable claims."""
from __future__ import annotations

import re
from typing import List


DECOMPOSE_SYSTEM = """You are a claim decomposition assistant.
Decompose a paragraph into atomic, self-contained factual claims.
Rules:
- Each claim must be independently verifiable.
- Each claim should contain exactly one assertion.
- Preserve all named entities exactly as stated.
- Do not infer or add information not present in the text.
- Return a JSON object with key "claims": a list of claim strings.
"""

DECOMPOSE_USER = """Decompose the following answer into atomic factual claims.
Return only a JSON object: {{"claims": ["claim1", "claim2", ...]}}

Answer:
{answer}"""


class ClaimDecomposer:

    def __init__(self, llm_client) -> None:
        self._llm = llm_client

    def decompose(self, answer: str) -> List[str]:
        """
        Returns a list of atomic claim strings.
        Falls back to sentence splitting if the LLM call fails.
        """
        if not answer or not answer.strip():
            return []

        messages = [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user",   "content": DECOMPOSE_USER.format(answer=answer.strip())},
        ]
        result = self._llm.chat_json(messages)

        if isinstance(result, dict) and "claims" in result:
            claims = [str(c).strip() for c in result["claims"] if str(c).strip()]
            if claims:
                return claims

        return _sentence_split(answer)


def _sentence_split(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
