from __future__ import annotations

import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

from ..infra.event_logger import log_event, init_event_log
from ..infra.student_model import StudentProvider
from ..infra.teacher_llamacpp import LlamaCppTeacher
from ..infra.teacher_local_hf import LocalHFTeacher
from ..infra.model_manager import (
    ensure_student_model_available,
    ensure_teacher_gguf_available,
    ensure_teacher_gguf_from_repo,
    ensure_teacher_gguf_simple,
)
from ..domain.tokenizer_align import (
    TokenizerAdapter,
    compute_token_spans,
    align_sequences_by_char_spans,
    merge_teacher_steps_logprobs,
    redistribute_teacher_to_student_vocab,
    merge_student_steps_logprobs,
    compute_gold_loss,
)
from ..app.prepare_dataset import load_prompts_any, render_prompt
from ..domain.types import DistillationBatch, DistillationPosition

# Training/loss params
MAX_NEW_TOKENS = 128
TOP_K_STUDENT = 50
BETA = 0.0
W_GKD = 1.0
W_ULD = 0.0


def run_once() -> None:
    # Load .env first
    load_dotenv()

    # Read config strictly from env/.env
    data_path = os.getenv("DATA_PATH")
    if not data_path:
        raise SystemExit("DATA_PATH not set in environment/.env")

    num_samples_env = os.getenv("NUM_SAMPLES", "all")
    num_samples: Optional[int]
    if str(num_samples_env).lower() in ("none", "all", "-1"):
        num_samples = None
    else:
        try:
            num_samples = int(num_samples_env)
        except ValueError:
            num_samples = None

    student_model_path = os.getenv("STUDENT_MODEL_PATH")
    student_model_id = os.getenv("STUDENT_MODEL_ID")

    # Single-var teacher configuration: TEACHER_GGUF can be a local .gguf path OR a HF repo id/URL
    teacher_spec = os.getenv("TEACHER_GGUF") or os.getenv("TEACHER_GGUF_REPO") or os.getenv("TEACHER_GGUF_HF_REPO_ID")
    teacher_gguf_path = os.getenv("TEACHER_GGUF_PATH")  # optional explicit path override
    teacher_filename = os.getenv("TEACHER_GGUF_FILENAME")  # optional filename hint (not required)

    log_file = init_event_log(output_dir="runs")
    log_event(
        "run_start",
        log_file=log_file,
        data_path=data_path,
        num_samples=num_samples_env,
        student_model_path=student_model_path,
        student_model_id=student_model_id,
        teacher_backend=os.getenv("TEACHER_BACKEND", "llama_cpp"),
        teacher_spec=teacher_spec,
        teacher_gguf_path=teacher_gguf_path,
        teacher_filename=teacher_filename,
    )

    prompts = load_prompts_any(data_path, limit=num_samples)
    if not prompts:
        raise SystemExit(f"No prompts extracted from dataset at {data_path}")

    # Ensure local student
    try:
        if student_model_path and os.path.isdir(student_model_path):
            resolved_student_dir = student_model_path
        else:
            if not student_model_id:
                raise RuntimeError("Set STUDENT_MODEL_PATH (preferred) or STUDENT_MODEL_ID in .env")
            resolved_student_dir = ensure_student_model_available(student_model_id, cache_root="models/hf")
        student = StudentProvider(f"local:{resolved_student_dir}")
        s_tok: TokenizerAdapter = student.get_tokenizer_adapter()
    except Exception as e:
        log_event("student_init_failed", error=str(e))
        student = StudentProvider("mock")
        s_tok = student.get_tokenizer_adapter()
        log_event("student_fallback_to_mock")

    teacher_backend = os.getenv("TEACHER_BACKEND", "llama_cpp").lower()
    try:
        if teacher_backend == "hf":
            teacher_model_id = os.getenv("TEACHER_MODEL_ID") or os.getenv("TEACHER_MODEL_PATH")
            if not teacher_model_id:
                raise RuntimeError("Set TEACHER_MODEL_ID (or TEACHER_MODEL_PATH) when TEACHER_BACKEND=hf")
            teacher_device = os.getenv("TEACHER_DEVICE", "auto")
            teacher_dtype = os.getenv("TEACHER_DTYPE", "bf16")
            teacher_top_k = int(os.getenv("TEACHER_TOP_K", "20"))
            teacher = LocalHFTeacher(
                model_id_or_path=teacher_model_id,
                use_mock=False,
                device=teacher_device,
                dtype=teacher_dtype,
                flash_attn=os.getenv("TEACHER_FLASH_ATTENTION", "1") == "1",
                top_k=teacher_top_k,
            )
        else:
            if teacher_gguf_path and os.path.isfile(teacher_gguf_path):
                resolved_gguf = teacher_gguf_path
            elif teacher_spec:
                # Single-var flow: local .gguf path or HF repo id/URL
                resolved_gguf = ensure_teacher_gguf_simple(teacher_spec, preferred_quant="Q4_K_M")
            else:
                raise RuntimeError("Set TEACHER_GGUF to a local .gguf path or a HF repo (id or URL)")
            teacher = LlamaCppTeacher(model_path=resolved_gguf, use_mock=False)
    except Exception as e:
        log_event("teacher_init_failed", error=str(e))
        if teacher_backend == "hf":
            teacher = LocalHFTeacher(model_id_or_path="mock", use_mock=True)
        else:
            teacher = LlamaCppTeacher(model_path=teacher_gguf_path or (teacher_spec or ""), use_mock=True)
        log_event("teacher_fallback_to_mock")

    total_loss = 0.0
    total_mapped = 0.0
    n = 0

    for i, item in enumerate(prompts):
        prompt_text = render_prompt(item)
        s_out = student.generate_and_score(prompt_text, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K_STUDENT)
        t_scores = teacher.score_completion_logprobs(prompt_text, s_out.completion_text)

        s_spans = compute_token_spans(s_out.completion_text, s_tok)
        # Teacher spans from token strings: approximate via greedy find on completion text
        t_str_steps = [step.token_logprobs for step in t_scores.per_step]
        # Build teacher TokenSpan-like via positions reusing student spans boundaries
        # We align via character merges anyway, so just slice using greedy char search
        t_spans = []
        cursor = 0
        for step in t_scores.per_step:
            tok = step.chosen_token or (max(step.token_logprobs.items(), key=lambda kv: kv[1])[0] if step.token_logprobs else "")
            if tok:
                idx = s_out.completion_text.find(tok, cursor)
                if idx == -1:
                    start = cursor
                    end = min(len(s_out.completion_text), cursor + len(tok))
                else:
                    start = idx
                    end = idx + len(tok)
            else:
                start = end = cursor
            t_spans.append(type("TS", (), {"start": start, "end": end, "text": s_out.completion_text[start:end], "token_id": -1}))
            cursor = end

        alignment = align_sequences_by_char_spans(s_spans, t_spans)

        student_vocab = s_tok.vocab_strings()
        student_lookup: Dict[str, int] = {s_tok.token_str(i): i for i in range(len(student_vocab))}

        positions: List[DistillationPosition] = []
        teacher_step_str_dists: List[Dict[str, float]] = t_str_steps
        for s_group, t_group in zip(alignment.student_groups, alignment.teacher_groups):
            if not s_group and not t_group:
                continue
            merged_t = merge_teacher_steps_logprobs(teacher_step_str_dists, t_group)
            t_on_s = redistribute_teacher_to_student_vocab(merged_t, student_lookup)
            s_merged = merge_student_steps_logprobs(s_out.per_position_logprobs, s_group)
            positions.append(
                DistillationPosition(
                    teacher_on_student_vocab=t_on_s,
                    student_logprobs=s_merged,
                )
            )

        batch = DistillationBatch(positions=positions)
        loss = compute_gold_loss(batch, beta=BETA, w_gkd=W_GKD, w_uld=W_ULD)
        total_loss += loss.total_loss
        total_mapped += loss.mapped_token_fraction
        n += 1
        log_event(
            "gold_step",
            sample_index=i,
            completion_chars=len(s_out.completion_text),
            gkd_loss=loss.gkd_loss,
            total_loss=loss.total_loss,
            mapped_token_fraction=loss.mapped_token_fraction,
        )

    log_event(
        "gold_summary",
        samples=n,
        avg_total_loss=(total_loss / max(n, 1)),
        avg_mapped_fraction=(total_mapped / max(n, 1)),
    )


if __name__ == "__main__":
    run_once()
