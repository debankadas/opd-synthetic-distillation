from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from ..domain.types import TeacherStepLogprobs


@dataclass
class TeacherScores:
    per_step: List[TeacherStepLogprobs]


class LocalHFTeacher:
    """
    Hugging Face Transformers-based teacher.
    Supports GPU (cuda), MPS (Apple), or CPU. Downloads models if missing.
    """

    def __init__(
        self,
        model_id_or_path: str,
        use_mock: bool = False,
        device: str | None = None,
        dtype: str | None = None,
        flash_attn: bool | None = None,
        top_k: int = 20,
    ):
        self.ident = model_id_or_path
        self.use_mock = use_mock
        self.device = self._resolve_device(device)
        self.top_k = top_k

        # dtype selection
        self.torch_dtype = None
        if torch is not None:
            if (dtype or "").lower() in ("bf16", "bfloat16") and hasattr(torch, "bfloat16"):
                self.torch_dtype = torch.bfloat16
            elif (dtype or "").lower() in ("fp16", "float16", "half") and hasattr(torch, "float16"):
                self.torch_dtype = torch.float16

        self.model = None
        self.tokenizer = None

        if not self.use_mock:
            if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
                raise RuntimeError("Transformers or torch not available for HF teacher")
            attn_impl = None
            if flash_attn or os.getenv("TEACHER_FLASH_ATTENTION", "0") == "1":
                attn_impl = "flash_attention_2"
            tok_kwargs = dict(trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(self.ident, **tok_kwargs)
            model_kwargs = dict(trust_remote_code=True)
            if self.torch_dtype is not None:
                model_kwargs["torch_dtype"] = self.torch_dtype
            if attn_impl is not None:
                model_kwargs["attn_implementation"] = attn_impl
            self.model = AutoModelForCausalLM.from_pretrained(self.ident, **model_kwargs)
            if self.device:
                self.model.to(self.device)
            self.model.eval()

    def score_completion_logprobs(self, prompt: str, completion: str) -> TeacherScores:
        if self.use_mock or self.model is None or self.tokenizer is None or torch is None:
            return self._score_mock(completion)
        tok = self.tokenizer
        enc_prompt = tok(prompt, add_special_tokens=False, return_tensors="pt")
        enc_comp = tok(completion, add_special_tokens=False, return_tensors="pt")
        input_ids = torch.cat([enc_prompt["input_ids"], enc_comp["input_ids"]], dim=1)
        attn = torch.ones_like(input_ids)
        if self.device == "cuda":
            input_ids = input_ids.cuda(non_blocking=True)
            attn = attn.cuda(non_blocking=True)
        elif self.device == "mps":
            input_ids = input_ids.to("mps")
            attn = attn.to("mps")
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # [1, seq_len, vocab]
            logprobs = torch.log_softmax(logits, dim=-1)
        prompt_len = enc_prompt["input_ids"].shape[1]
        comp_len = enc_comp["input_ids"].shape[1]
        per: List[TeacherStepLogprobs] = []
        comp_ids = enc_comp["input_ids"][0]
        k = max(1, min(self.top_k, logprobs.shape[-1]))
        for i in range(comp_len):
            pos = prompt_len + i
            dist = logprobs[0, pos - 1]
            topv, topi = torch.topk(dist, k=k)
            tokens = tok.convert_ids_to_tokens(topi.tolist())
            step_dist = {t: float(v) for t, v in zip(tokens, topv.tolist())}
            chosen_id = int(comp_ids[i])
            chosen_tok = tok.convert_ids_to_tokens([chosen_id])[0]
            per.append(TeacherStepLogprobs(token_logprobs=step_dist, chosen_token=chosen_tok))
        return TeacherScores(per_step=per)

    def _score_mock(self, completion: str) -> TeacherScores:
        toks = completion.split()
        per: List[TeacherStepLogprobs] = []
        for t in toks:
            per.append(TeacherStepLogprobs(token_logprobs={t: math.log(0.85)}, chosen_token=t))
        return TeacherScores(per_step=per)

    @staticmethod
    def _resolve_device(device: str | None) -> str | None:
        if device and device != "auto":
            return device
        if torch is None:
            return None
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
