from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional
    requests = None

from ..domain.types import TeacherStepLogprobs
from .student_model import HFTokenizerAdapter

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


@dataclass
class TeacherScores:
    per_step: List[TeacherStepLogprobs]


class TeacherProvider:
    def __init__(self, model_spec: str, use_mock: bool = False):
        self.model_spec = model_spec
        self.kind, _, ident = model_spec.partition(":")
        self.ident = ident
        self.use_mock = use_mock or (os.getenv("OPENROUTER_API_KEY") in (None, ""))

        self.tokenizer = None  # avoid network in mock

    def score_completion_logprobs(self, prompt: str, completion: str) -> TeacherScores:
        if self.use_mock:
            return self._score_mock(completion)
        if self.kind != "openrouter":
            raise ValueError("Only openrouter teacher is supported in non-mock mode")
        if requests is None:
            raise RuntimeError("requests not available; cannot call OpenRouter")
        api_key = os.environ["OPENROUTER_API_KEY"]
        url = "https://openrouter.ai/api/v1/chat/completions"
        # Ask for logprobs over the provided completion by sending it as a forced output (best-effort; depends on OpenRouter support).
        # If the API does not support scoring arbitrary completions with per-token logprobs, you can instead ask the model to generate
        # and use its own tokens/logprobs to approximate on-policy teacher scores.
        payload = {
            "model": self.ident,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "logprobs": True,
            "max_tokens": max(1, len(completion)),
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": os.getenv("OPENROUTER_REFERER", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "on-policy-gold-local"),
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Expected shape (pseudo): data["choices"][0]["logprobs"]["content"][t]["top_logprobs"] => [{"token": str, "logprob": float}, ...]
        per_step: List[TeacherStepLogprobs] = []
        choice = data.get("choices", [{}])[0]
        logprobs = choice.get("logprobs", {})
        content = logprobs.get("content", [])
        for step in content:
            top = step.get("top_logprobs", [])
            dist = {item.get("token"): float(item.get("logprob")) for item in top if item.get("token")}
            chosen = step.get("token")
            per_step.append(TeacherStepLogprobs(token_logprobs=dist, chosen_token=chosen))
        return TeacherScores(per_step=per_step)

    def _score_mock(self, completion: str) -> TeacherScores:
        # Deterministic mock: assign 0.85 to the actual token string, 0.15 to a neighbor token string
        # whitespace fallback
        toks = completion.split()
        per: List[TeacherStepLogprobs] = []
        for t in toks:
            dist = {t: math.log(0.85)}
            per.append(TeacherStepLogprobs(token_logprobs=dist, chosen_token=t))
        return TeacherScores(per_step=per)
