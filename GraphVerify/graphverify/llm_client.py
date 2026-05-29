"""LLM abstraction supporting OpenAI-compatible API and local Qwen2.5-7B-Instruct."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .config import GraphVerifyConfig


class LLMClient:

    def __init__(self, cfg: GraphVerifyConfig) -> None:
        self.cfg = cfg
        self._backend = cfg.llm_backend
        self._model = cfg.llm_model
        self._temperature = cfg.llm_temperature
        self._max_tokens = cfg.llm_max_tokens

        if self._backend == "openai":
            self._init_openai()
        elif self._backend == "local":
            self._init_local()
        else:
            raise ValueError(f"Unknown LLM backend: {self._backend}")

    def _init_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", None),
        )

    def _init_local(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch accelerate")
        import torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.local_model_path, trust_remote_code=True
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._local_model.eval()

    def chat(self, messages: List[Dict[str, str]], *, json_mode: bool = False) -> str:
        if self._backend == "openai":
            return self._openai_chat(messages, json_mode=json_mode)
        return self._local_chat(messages)

    def chat_json(self, messages: List[Dict[str, str]]) -> Any:
        """Send chat and parse the response as JSON. Returns None on parse failure."""
        raw = self.chat(messages, json_mode=True)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for pattern in (r'\{.*\}', r'\[.*\]'):
                m = re.search(pattern, text, re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
            return None

    def _openai_chat(self, messages: List[Dict[str, str]], *, json_mode: bool) -> str:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def _local_chat(self, messages: List[Dict[str, str]]) -> str:
        import torch
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._local_model.device)
        with torch.no_grad():
            out = self._local_model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                temperature=None,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
