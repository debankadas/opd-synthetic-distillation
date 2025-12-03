from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import (
    DistillationBatch,
    DistillationPosition,
    MergedTeacherDist,
    SequenceAlignment,
    TokenSpan,
    VocabMapping,
    LossBreakdown,
)


class TokenizerAdapter:
    """
    Minimal interface wrapper around a tokenizer.
    Implementations must provide:
      - encode(text) -> List[int]
      - decode(ids) -> str
      - token_str(id) -> str
      - vocab_strings() -> List[str]
    """

    def encode(self, text: str) -> List[int]:  # pragma: no cover (interface)
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:  # pragma: no cover (interface)
        raise NotImplementedError

    def token_str(self, token_id: int) -> str:  # pragma: no cover (interface)
        raise NotImplementedError

    def vocab_strings(self) -> List[str]:  # pragma: no cover (interface)
        raise NotImplementedError


def compute_token_spans(text: str, tokenizer: TokenizerAdapter) -> List[TokenSpan]:
    ids = tokenizer.encode(text)
    spans: List[TokenSpan] = []
    cursor = 0
    for tid in ids:
        tstr = tokenizer.token_str(tid)
        # Greedy match tstr in text starting at cursor.
        # This is a heuristic for BPE-like tokenizers that retain token strings.
        # If not found, fallback to slice-by-length.
        next_idx = text.find(tstr, cursor)
        if next_idx == -1:
            # Fallback: assume length of token string and move cursor
            start = cursor
            end = min(len(text), cursor + len(tstr))
        else:
            start = next_idx
            end = next_idx + len(tstr)
        spans.append(TokenSpan(start=start, end=end, text=text[start:end], token_id=tid))
        cursor = end
    return spans


def align_sequences_by_char_spans(
    student_spans: List[TokenSpan], teacher_spans: List[TokenSpan]
) -> SequenceAlignment:
    # Build merged boundaries from both sets of spans
    boundaries = sorted({0, *[s.start for s in student_spans], *[s.end for s in student_spans], *[t.start for t in teacher_spans], *[t.end for t in teacher_spans]})
    student_groups: List[List[int]] = []
    teacher_groups: List[List[int]] = []

    # Index by span start/end to find overlaps
    def group_indices(spans: List[TokenSpan], start: int, end: int) -> List[int]:
        out: List[int] = []
        for i, sp in enumerate(spans):
            if not (sp.end <= start or sp.start >= end):
                out.append(i)
        return out

    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        s_idx = group_indices(student_spans, seg_start, seg_end)
        t_idx = group_indices(teacher_spans, seg_start, seg_end)
        if s_idx or t_idx:
            # Merge contiguous same groups with previous if unchanged
            if (
                student_groups
                and s_idx == student_groups[-1]
                and t_idx == teacher_groups[-1]
            ):
                # already represented
                continue
            student_groups.append(s_idx)
            teacher_groups.append(t_idx)

    # Remove empty entries that can occur due to non-overlap
    student_groups = [g or [] for g in student_groups]
    teacher_groups = [g or [] for g in teacher_groups]
    return SequenceAlignment(student_groups=student_groups, teacher_groups=teacher_groups)


def build_vocab_mapping(
    student_tok: TokenizerAdapter, teacher_tok: TokenizerAdapter
) -> VocabMapping:
    s_tokens = set(student_tok.vocab_strings())
    t_tokens = set(teacher_tok.vocab_strings())
    common = sorted(s_tokens.intersection(t_tokens))
    teacher_only = sorted(t_tokens - s_tokens)
    student_only = sorted(s_tokens - t_tokens)

    common_map: Dict[str, Tuple[List[int], List[int]]] = {}
    # In practice multiple ids can map to same string with added spaces; keep exact matches only.
    s_lookup: Dict[str, List[int]] = {}
    for sid in range(len(student_tok.vocab_strings())):
        sstr = student_tok.token_str(sid)
        s_lookup.setdefault(sstr, []).append(sid)
    t_lookup: Dict[str, List[int]] = {}
    for tid in range(len(teacher_tok.vocab_strings())):
        tstr = teacher_tok.token_str(tid)
        t_lookup.setdefault(tstr, []).append(tid)
    for tok in common:
        common_map[tok] = (s_lookup.get(tok, []), t_lookup.get(tok, []))

    union = len(s_tokens.union(t_tokens))
    iou = (len(common) / union) if union else 0.0
    return VocabMapping(
        common_token_to_ids=common_map,
        teacher_only=teacher_only,
        student_only=student_only,
        jaccard_similarity=iou,
    )


def logsumexp_values(items: Iterable[float]) -> float:
    xs = list(items)
    if not xs:
        return -float("inf")
    m = max(xs)
    s = sum(math.exp(x - m) for x in xs)
    return m + math.log(s)


def merge_teacher_steps_logprobs(
    teacher_steps: List[Dict[str, float]], groups: List[int]
) -> MergedTeacherDist:
    """
    Merge multiple teacher steps into a single distribution by summing token logprobs
    (log of joint probability) followed by normalization.

    teacher_steps: list of dict token_str->logprob for each teacher step.
    groups: indices of teacher steps to merge.
    """
    # Collect union of candidate tokens across merged steps
    all_tokens = set()
    for i in groups:
        all_tokens.update(teacher_steps[i].keys())
    # Sum logprobs per token (missing tokens treated as -inf)
    summed: Dict[str, float] = {}
    for tok in all_tokens:
        vals = []
        for i in groups:
            val = teacher_steps[i].get(tok, -float("inf"))
            vals.append(val)
        # joint log-probability approximation via sum
        summed[tok] = sum(vals)
    # Normalize to log-softmax
    lse = logsumexp_values(summed.values())
    if not math.isfinite(lse):
        return MergedTeacherDist(token_logprobs={})
    normalized = {k: (v - lse) for k, v in summed.items()}
    return MergedTeacherDist(token_logprobs=normalized)


def redistribute_teacher_to_student_vocab(
    merged: MergedTeacherDist, student_lookup: Dict[str, int]
) -> Dict[int, float]:
    """
    Build a student-vocab distribution from teacher token strings:
    - For exact token string matches, assign probability mass to the student token id.
    - For unmatched tokens, fallback to ULD-like rank sorting: ignore here for simplicity in domain; caller can combine with ULD loss separately.
    Returns: student_token_id -> logprob (log-softmaxed over the matched subset).
    """
    matched_pairs: List[Tuple[int, float]] = []
    for tok_str, lp in merged.token_logprobs.items():
        sid = student_lookup.get(tok_str)
        if sid is not None:
            matched_pairs.append((sid, lp))
    if not matched_pairs:
        return {}
    # If multiple teacher tokens map to same student id, logsumexp them
    acc: Dict[int, List[float]] = {}
    for sid, lp in matched_pairs:
        acc.setdefault(sid, []).append(lp)
    sid_log = {sid: logsumexp_values(vs) for sid, vs in acc.items()}
    # Renormalize over matched student ids
    lse = logsumexp_values(sid_log.values())
    return {sid: (lp - lse) for sid, lp in sid_log.items()}


def merge_student_steps_logprobs(
    student_steps: List[Dict[int, float]], groups: List[int]
) -> Dict[int, float]:
    """
    Merge student's per-step logprobs across multiple positions (sum of logprobs per id, then renormalize).
    Returns student_id->logprob.
    """
    all_ids = set()
    for i in groups:
        all_ids.update(student_steps[i].keys())
    summed: Dict[int, float] = {}
    for sid in all_ids:
        vals: List[float] = []
        for i in groups:
            vals.append(student_steps[i].get(sid, -float("inf")))
        summed[sid] = sum(vals)
    lse = logsumexp_values(summed.values())
    if not math.isfinite(lse):
        return {}
    return {k: (v - lse) for k, v in summed.items()}


def kl_divergence(p_log: Dict[int, float], q_log: Dict[int, float]) -> float:
    """
    D_KL(P || Q) where P and Q are dicts of logprobs over same support subset.
    Missing keys treated as zero mass in Q => contributes +inf; here we guard by limiting to intersection.
    """
    keys = set(p_log.keys()) & set(q_log.keys())
    if not keys:
        return 0.0
    kl = 0.0
    for k in keys:
        p = math.exp(p_log[k])
        q = math.exp(q_log[k])
        if q <= 0:
            continue
        kl += p * (p_log[k] - q_log[k])
    return kl


def jsd_beta(p_log: Dict[int, float], q_log: Dict[int, float], beta: float) -> float:
    # Generalized JSD with mixture pi = beta*p + (1-beta)*q
    keys = set(p_log.keys()) | set(q_log.keys())
    # Ensure log-space safety by filling missing with -inf
    def get_prob(d: Dict[int, float], k: int) -> float:
        return math.exp(d[k]) if k in d else 0.0
    pi: Dict[int, float] = {}
    for k in keys:
        ps = get_prob(p_log, k)
        qs = get_prob(q_log, k)
        pi[k] = beta * ps + (1 - beta) * qs
    # Convert pi to log
    total = sum(pi.values()) or 1.0
    pi_log = {k: (math.log(v / total) if v > 0 else -float("inf")) for k, v in pi.items()}
    return beta * kl_divergence(p_log, pi_log) + (1 - beta) * kl_divergence(q_log, pi_log)


def compute_gold_loss(
    batch: DistillationBatch,
    beta: float,
    w_gkd: float,
    w_uld: float,
) -> LossBreakdown:
    gkd_terms: List[float] = []
    uld_terms: List[float] = []
    mapped = 0
    total = 0
    for pos in batch.positions:
        total += 1
        teacher_sid_log = pos.teacher_on_student_vocab
        if teacher_sid_log:
            mapped += 1
            gkd_terms.append(jsd_beta(teacher_sid_log, pos.student_logprobs, beta))
        else:
            # ULD fallback (very rough): compare sorted lists by rank overlap
            # Build rank lists
            st_sorted = sorted(pos.student_logprobs.items(), key=lambda x: x[1], reverse=True)
            # No teacher support -> zero contribution here; in practice you'd sort teacher logits after padding
            uld_terms.append(0.0)

    gkd = sum(gkd_terms) / len(gkd_terms) if gkd_terms else 0.0
    uld = sum(uld_terms) / len(uld_terms) if uld_terms else 0.0
    total_loss = w_gkd * gkd + w_uld * uld
    frac = (mapped / total) if total else 0.0
    return LossBreakdown(gkd_loss=gkd, uld_loss=uld, total_loss=total_loss, mapped_token_fraction=frac)
