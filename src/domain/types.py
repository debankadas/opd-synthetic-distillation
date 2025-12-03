from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class Conversation:
    messages: List[ChatMessage]


@dataclass(frozen=True)
class TokenSpan:
    # Character offsets in the rendered text for a token.
    start: int
    end: int  # exclusive
    text: str
    token_id: int


@dataclass(frozen=True)
class SequenceAlignment:
    # For each aligned position i, lists of indices into student and teacher token lists.
    # Example: student_groups[i] = [2], teacher_groups[i] = [2,3] means teacher token 2+3 merged align to student token 2.
    student_groups: List[List[int]]
    teacher_groups: List[List[int]]


@dataclass(frozen=True)
class VocabMapping:
    # Map exact token string -> (student_ids, teacher_ids)
    common_token_to_ids: Dict[str, Tuple[List[int], List[int]]]
    # Token strings only in teacher or student
    teacher_only: List[str]
    student_only: List[str]
    jaccard_similarity: float


@dataclass
class TeacherStepLogprobs:
    # For position t (teacher tokenizer step), a distribution over token strings -> logprob.
    # Only a subset may be present if API returns top-k.
    token_logprobs: Dict[str, float]
    # The chosen token string by teacher at t (if generation occurred).
    chosen_token: Optional[str] = None


@dataclass
class MergedTeacherDist:
    # Distribution over token strings after merging teacher steps for one aligned position.
    token_logprobs: Dict[str, float]


@dataclass
class DistillationPosition:
    # GOLD per aligned position
    teacher_on_student_vocab: Dict[int, float]  # teacher dist mapped to student ids (logprobs)
    student_logprobs: Dict[int, float]  # student token id -> logprob (top-k or logits->logsoftmax)


@dataclass
class DistillationBatch:
    positions: List[DistillationPosition]


@dataclass
class LossBreakdown:
    # KL terms
    gkd_loss: float
    uld_loss: float
    total_loss: float
    mapped_token_fraction: float
