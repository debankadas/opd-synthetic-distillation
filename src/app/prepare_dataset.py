from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional
import random


@dataclass
class PromptItem:
    system: str
    user: str
    assistant: str | None = None
    source: str | None = None


def _extract_from_conversations(row: dict[str, Any]) -> PromptItem | None:
    conv = row.get("conversations", []) or row.get("messages", [])
    if not isinstance(conv, list) or not conv:
        return None
    system = ""
    user = ""
    assistant = None
    for m in conv:
        role = m.get("role") if isinstance(m, dict) else None
        content = m.get("content") if isinstance(m, dict) else None
        if role == "system" and not system and content:
            system = content
        if role == "user" and content:
            user = content
        if role == "assistant" and content and assistant is None:
            assistant = content
    if not user:
        # try last user
        users = [m.get("content", "") for m in conv if isinstance(m, dict) and m.get("role") == "user"]
        user = users[-1] if users else ""
    if not user:
        return None
    if not system:
        system = "You are a helpful AI assistant."
    return PromptItem(system=system, user=user, assistant=assistant, source=row.get("source"))


def _extract_from_alpaca(row: dict[str, Any]) -> PromptItem | None:
    if "instruction" in row:
        instr = row.get("instruction", "")
        inp = row.get("input", "")
        system = row.get("system", "You are a helpful AI assistant.")
        user = instr if not inp else f"{instr}\n\n{inp}"
        assistant = row.get("output") or row.get("response")
        return PromptItem(system=system, user=user, assistant=assistant, source=row.get("source"))
    return None


def _extract_from_prompt_completion(row: dict[str, Any]) -> PromptItem | None:
    prompt = row.get("prompt") or row.get("question") or row.get("input")
    if isinstance(prompt, str) and prompt.strip():
        system = row.get("system", "You are a helpful AI assistant.")
        assistant = row.get("response") or row.get("completion") or row.get("answer")
        return PromptItem(system=system, user=prompt, assistant=assistant, source=row.get("source"))
    return None


def _extract_from_text(row: dict[str, Any]) -> PromptItem | None:
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        system = row.get("system", "You are a helpful AI assistant.")
        return PromptItem(system=system, user=text, source=row.get("source"))
    return None


def _extract_from_dpo(row: dict[str, Any]) -> PromptItem | None:
    # Common DPO shape: {prompt, chosen, rejected}
    prompt = row.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        system = row.get("system", "You are a helpful AI assistant.")
        # We could take 'chosen' as assistant label if desired
        assistant_val = row.get("chosen")
        assistant: str | None = None
        if isinstance(assistant_val, str):
            assistant = assistant_val
        elif isinstance(assistant_val, dict):
            assistant = assistant_val.get("content") or assistant_val.get("text")
        elif isinstance(assistant_val, list):
            # list of messages/shards
            parts: list[str] = []
            for m in assistant_val:
                if isinstance(m, dict):
                    if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                        parts.append(m["content"]) 
                    elif isinstance(m.get("text"), str):
                        parts.append(m["text"]) 
            assistant = "\n".join(parts) if parts else None
        return PromptItem(system=system, user=prompt, assistant=assistant, source=row.get("source"))
    return None


def extract_prompts(rows: List[dict[str, Any]], limit: int | None = None) -> List[PromptItem]:
    extractors = (
        _extract_from_conversations,
        _extract_from_alpaca,
        _extract_from_prompt_completion,
        _extract_from_dpo,
        _extract_from_text,
    )
    out: List[PromptItem] = []
    for row in rows:
        item: PromptItem | None = None
        for fn in extractors:
            item = fn(row)
            if item:
                break
        if not item:
            continue
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def load_prompts_any(path: str, limit: int | None = None) -> List[PromptItem]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("items") or payload.get("conversations") or []
        if not isinstance(rows, list):
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    return extract_prompts(rows, limit=limit)


def load_prompts_sampled(path: str, limit: Optional[int] = None, seed: int = 42) -> List[PromptItem]:
    """Load prompts from various JSON schemas and randomly sample up to `limit` rows.

    - If limit is None or >= dataset size, returns all rows.
    - Sampling happens before schema extraction to reduce top-of-file bias.
    """
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("items") or payload.get("conversations") or []
        if not isinstance(rows, list):
            rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    if limit is not None and limit > 0 and len(rows) > limit:
        rnd = random.Random(seed)
        idxs = rnd.sample(range(len(rows)), k=limit)
        rows = [rows[i] for i in idxs]
        limit = None  # we've already limited
    return extract_prompts(rows, limit=limit)


def render_prompt(item: PromptItem) -> str:
    # Simple, explicit chat-ish template suitable for many instruction-tuned models
    return f"System: {item.system}\nUser: {item.user}\nAssistant:"
