from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
except Exception:  # pragma: no cover - optional import for offline
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None

from ..domain.tokenizer_align import TokenizerAdapter


class HFTokenizerAdapter(TokenizerAdapter):
    def __init__(self, hf_tokenizer):
        self.tok = hf_tokenizer

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def token_str(self, token_id: int) -> str:
        return self.tok.convert_ids_to_tokens([token_id])[0]

    def vocab_strings(self) -> List[str]:
        # Best-effort: many tokenizers expose get_vocab() mapping str->id
        if hasattr(self.tok, "get_vocab"):
            # Return in id order for deterministic mapping
            vocab = self.tok.get_vocab()
            inv = {v: k for k, v in vocab.items()}
            return [inv[i] for i in range(len(inv))]
        raise RuntimeError("Tokenizer does not expose get_vocab()")


@dataclass
class StudentOutput:
    completion_text: str
    token_ids: List[int]
    per_position_logprobs: List[Dict[int, float]]  # top-k logprobs over student ids


class StudentProvider:
    def __init__(self, model_spec: str, device: str | None = None):
        self.model_spec = model_spec
        self.kind, _, ident = model_spec.partition(":")
        self.ident = ident

        self.model = None
        self.tokenizer = None
        self.adapter = None
        self.device = device or "auto"

        if self.kind == "mock":
            return
        if self.kind == "local":
            if AutoModelForCausalLM is None:
                raise RuntimeError("Transformers not available for local model")
            self.tokenizer = AutoTokenizer.from_pretrained(self.ident, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.ident, trust_remote_code=True)
            # Device placement
            target_device = self._resolve_device()
            if target_device and hasattr(self.model, "to"):
                self.model.to(target_device)
            self.model.eval()
            self.adapter = HFTokenizerAdapter(self.tokenizer)
        else:
            raise ValueError(f"Unsupported student provider: {self.kind}")

    def get_tokenizer_adapter(self) -> TokenizerAdapter:
        if self.kind == "mock":
            # Provide a tiny fake adapter that splits on spaces for offline tests
            return _WhitespaceTokenizerAdapter()
        return self.adapter

    def generate_and_score(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        top_k: int = 50,
    ) -> StudentOutput:
        if self.kind == "mock":
            # Deterministic pseudo-completion: echo last sentence and produce pseudo-logprobs
            completion = "\nSure, here are the key points."
            adapter = self.get_tokenizer_adapter()
            ids = adapter.encode(completion)
            per_pos: List[Dict[int, float]] = []
            for tid in ids:
                # Assign most mass to the token itself, small tail to neighbors
                dist = {tid: math.log(0.9)}
                dist[(tid + 1) % max(len(adapter.vocab_strings()), 1) if adapter.vocab_strings() else tid] = math.log(0.1)
                # Renormalize safety
                lse = _logsumexp(dist.values())
                per_pos.append({k: v - lse for k, v in dist.items()})
            return StudentOutput(completion_text=completion, token_ids=ids, per_position_logprobs=per_pos)

        # HF path
        assert self.model is not None and self.tokenizer is not None
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self._resolve_device() == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=min(top_k, 50),
                return_dict_in_generate=True,
                output_scores=True,
            )
        # Extract completion tokens
        gen_ids = out.sequences[0][inputs["input_ids"].shape[1] :].tolist()
        # Convert scores (logits) -> logprobs top-k per position
        per_pos: List[Dict[int, float]] = []
        for step_scores in out.scores:
            logits = step_scores[0]
            logprobs = torch.log_softmax(logits, dim=-1)
            topv, topi = torch.topk(logprobs, k=min(top_k, logprobs.shape[-1]))
            per_pos.append({int(i): float(v) for i, v in zip(topi.tolist(), topv.tolist())})
        completion = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return StudentOutput(completion_text=completion, token_ids=gen_ids, per_position_logprobs=per_pos)

    # Helpers
    def _resolve_device(self) -> str | None:
        if self.device == "auto":
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return self.device


def _logsumexp(vals):
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


class _WhitespaceTokenizerAdapter(TokenizerAdapter):  # pragma: no cover - used in mock environments only
    def __init__(self):
        self._vocab = []  # filled lazily

    def _ensure_vocab(self, text: str):
        toks = sorted(set(text.split()))
        self._vocab = toks

    def encode(self, text: str) -> List[int]:
        if not self._vocab:
            self._ensure_vocab(text)
        return [self._vocab.index(t) if t in self._vocab else self._add(t) for t in text.split()]

    def _add(self, t: str) -> int:
        self._vocab.append(t)
        return len(self._vocab) - 1

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._vocab[i] for i in ids)

    def token_str(self, token_id: int) -> str:
        return self._vocab[token_id]

    def vocab_strings(self) -> List[str]:
        return list(self._vocab)

    
