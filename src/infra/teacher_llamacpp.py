from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List
import os

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional
    Llama = None

from ..domain.types import TeacherStepLogprobs


@dataclass
class TeacherScores:
    per_step: List[TeacherStepLogprobs]


class LlamaCppTeacher:
    """
    Local teacher backed by llama.cpp GGUF.
    Scores a given student completion by computing per-token logprobs using echo=True.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int | None = None,
        use_mock: bool = False,
    ):
        self.model_path = model_path
        self.use_mock = use_mock
        self.llm = None
        if not self.use_mock:
            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python not installed. Install it and provide a GGUF model."
                )
            # Env overrides (optional)
            env_n_ctx = int(os.getenv("TEACHER_N_CTX", n_ctx))
            env_n_threads = os.getenv("TEACHER_N_THREADS")
            env_n_gpu_layers = os.getenv("TEACHER_N_GPU_LAYERS")
            # Default to offload all layers to GPU on Metal if not specified
            default_n_gpu_layers = int(env_n_gpu_layers) if env_n_gpu_layers is not None else (
                int(n_gpu_layers) if n_gpu_layers is not None else -1
            )

            # logits_all=True is required to request per-token logprobs
            kwargs = {"model_path": model_path, "n_ctx": env_n_ctx, "logits_all": True, "n_gpu_layers": default_n_gpu_layers, "verbose": False}
            if n_threads is not None:
                kwargs["n_threads"] = int(n_threads)
            elif env_n_threads is not None:
                kwargs["n_threads"] = int(env_n_threads)
            self.llm = Llama(**kwargs)

    def score_completion_logprobs(self, prompt: str, completion: str) -> TeacherScores:
        if self.use_mock or self.llm is None:
            return self._score_mock(completion)
        # Tokenize prompt and completion to slice echo outputs correctly
        p_tokens = self.llm.tokenize(bytes(prompt, "utf-8"), add_bos=True)
        c_tokens = self.llm.tokenize(bytes(completion, "utf-8"), add_bos=False)
        # Ask llama.cpp to score the concatenation using echo=True
        top_k = int(os.getenv("TEACHER_TOP_K", "20"))
        result = self.llm.create_completion(
            prompt=prompt + completion,
            max_tokens=0,
            temperature=0.0,
            logprobs=top_k,
            echo=True,
        )
        choice = result.get("choices", [{}])[0]
        lp = choice.get("logprobs", {})
        top = lp.get("top_logprobs", [])
        tokens = lp.get("tokens") or []
        # Extract the portion that corresponds to completion
        # top is aligned with tokens in choice["text"], but llama-cpp also returns token list
        # We'll use token offsets by length of prompt tokens
        start = len(p_tokens)
        end = start + len(c_tokens)
        per: List[TeacherStepLogprobs] = []
        for i in range(start, min(end, len(top))):
            top_i = top[i] or {}
            # Normalize values to floats; ensure token strings present
            dist = {str(k): float(v) for k, v in top_i.items()}
            chosen = None
            if i < len(tokens):
                chosen = tokens[i]
            per.append(TeacherStepLogprobs(token_logprobs=dist, chosen_token=chosen))
        return TeacherScores(per_step=per)

    def _score_mock(self, completion: str) -> TeacherScores:
        toks = completion.split()
        per: List[TeacherStepLogprobs] = []
        for t in toks:
            per.append(TeacherStepLogprobs(token_logprobs={t: math.log(0.85)}, chosen_token=t))
        return TeacherScores(per_step=per)
