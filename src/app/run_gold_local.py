from __future__ import annotations

import argparse
from typing import Dict, List

from ..infra.config import DistillConfig
from ..infra.event_logger import log_event
from ..infra.student_model import StudentProvider
from ..infra.teacher_openrouter import TeacherProvider
from ..infra.teacher_local_hf import LocalHFTeacher
from ..domain.tokenizer_align import (
    TokenizerAdapter,
    compute_token_spans,
    align_sequences_by_char_spans,
    merge_teacher_steps_logprobs,
    redistribute_teacher_to_student_vocab,
    merge_student_steps_logprobs,
    compute_gold_loss,
)
from .prepare_dataset import load_prompts_any, render_prompt
from ..domain.types import DistillationBatch, DistillationPosition, TeacherStepLogprobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local GOLD-style on-policy distillation dry run")
    p.add_argument("--data-path", required=False, default="sample_training_data/retailco_training_data_sft_conversations.json")
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--student-model", default="mock", help="mock or local:<hf-id-or-path>")
    p.add_argument("--teacher-model", default="openrouter:qwen/qwen2.5-7b-instruct")
    p.add_argument("--use-mock-teacher", action="store_true")
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--w-gkd", type=float, default=1.0)
    p.add_argument("--w-uld", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-k-student", type=int, default=50)
    return p.parse_args()


def build_teacher_token_strings(per_step: List[TeacherStepLogprobs], completion_text: str) -> List[str]:
    # Prefer chosen_token; fallback to argmax of top_logprobs
    toks: List[str] = []
    for step in per_step:
        if step.chosen_token:
            toks.append(step.chosen_token)
        elif step.token_logprobs:
            best = max(step.token_logprobs.items(), key=lambda kv: kv[1])[0]
            toks.append(best)
        else:
            toks.append("")
    # As a guard, ensure concatenation approx equals completion prefix; otherwise, rely on alignment over char spans
    return toks


def teacher_spans_from_strings(token_strings: List[str], text: str):
    spans = []
    cursor = 0
    for t in token_strings:
        if not t:
            start = cursor
            end = cursor
        else:
            idx = text.find(t, cursor)
            if idx == -1:
                start = cursor
                end = min(len(text), cursor + len(t))
            else:
                start = idx
                end = idx + len(t)
        spans.append((start, end, t))
        cursor = end
    return spans


def main() -> None:
    args = parse_args()
    cfg = DistillConfig(
        data_path=args.data_path,
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        use_mock_teacher=bool(args.use_mock_teacher),
        num_samples=int(args.num_samples),
        beta=args.beta,
        w_gkd=args.w_gkd,
        w_uld=args.w_uld,
        max_new_tokens=args.max_new_tokens,
        top_k_student=args.top_k_student,
    )

    prompts = load_prompts_any(cfg.data_path, limit=cfg.num_samples)
    if not prompts:
        raise SystemExit("No prompts extracted from dataset. Check file or format.")

    student = StudentProvider(cfg.student_model, device=cfg.device)
    if args.teacher_model.startswith("local:"):
        teacher = LocalHFTeacher(cfg.teacher_model, use_mock=cfg.use_mock_teacher, device=cfg.device)
    else:
        teacher = TeacherProvider(cfg.teacher_model, use_mock=cfg.use_mock_teacher)

    s_tok: TokenizerAdapter = student.get_tokenizer_adapter()

    total_loss = 0.0
    total_mapped_frac = 0.0
    count = 0

    for i, item in enumerate(prompts):
        prompt_text = render_prompt(item)
        s_out = student.generate_and_score(prompt_text, max_new_tokens=cfg.max_new_tokens, top_k=cfg.top_k_student)
        t_scores = teacher.score_completion_logprobs(prompt_text, s_out.completion_text)

        # Spans for alignment over the student's completion text
        s_spans = compute_token_spans(s_out.completion_text, s_tok)
        t_tok_strings = build_teacher_token_strings(t_scores.per_step, s_out.completion_text)
        t_spans = [
            # Fabricate TokenSpan-like objects using student text
            # token_id is dummy (-1) since we only need char ranges
            type("TS", (), {"start": s, "end": e, "text": txt, "token_id": -1})
            for (s, e, txt) in teacher_spans_from_strings(t_tok_strings, s_out.completion_text)
        ]

        alignment = align_sequences_by_char_spans(s_spans, t_spans)  # groups of indices

        # Prepare vocab mapping for redistribution
        # Build lookup str->sid for student
        student_vocab = s_tok.vocab_strings()
        student_lookup: Dict[str, int] = {s_tok.token_str(i): i for i in range(len(student_vocab))}

        # Build per-position structures
        positions: List[DistillationPosition] = []
        # Teacher step distributions in strings
        teacher_step_str_dists: List[Dict[str, float]] = [step.token_logprobs for step in t_scores.per_step]
        for s_group, t_group in zip(alignment.student_groups, alignment.teacher_groups):
            if not s_group and not t_group:
                continue
            # Merge teacher steps over t_group
            merged_t = merge_teacher_steps_logprobs(teacher_step_str_dists, t_group)
            # Map to student vocab (id->logprob)
            t_on_s = redistribute_teacher_to_student_vocab(merged_t, student_lookup)
            # Merge student steps over s_group
            s_merged = merge_student_steps_logprobs(s_out.per_position_logprobs, s_group)
            positions.append(DistillationPosition(teacher_on_student_vocab=t_on_s, student_logprobs=s_merged))

        batch = DistillationBatch(positions=positions)
        loss = compute_gold_loss(batch, beta=cfg.beta, w_gkd=cfg.w_gkd, w_uld=cfg.w_uld)
        total_loss += loss.total_loss
        total_mapped_frac += loss.mapped_token_fraction
        count += 1

        log_event(
            "gold_step",
            sample_index=i,
            prompt_chars=len(prompt_text),
            completion_chars=len(s_out.completion_text),
            gkd_loss=loss.gkd_loss,
            uld_loss=loss.uld_loss,
            total_loss=loss.total_loss,
            mapped_token_fraction=loss.mapped_token_fraction,
        )

    avg_loss = total_loss / max(count, 1)
    avg_mapped = total_mapped_frac / max(count, 1)
    log_event("gold_summary", samples=count, avg_total_loss=avg_loss, avg_mapped_fraction=avg_mapped)


if __name__ == "__main__":
    main()
