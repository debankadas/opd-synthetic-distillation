from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import contextlib
from datetime import datetime

import torch
from torch.optim import AdamW
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..infra.event_logger import init_event_log, log_event
from ..app.prepare_dataset import load_prompts_any, render_prompt
from ..infra.teacher_local_hf import LocalHFTeacher
from ..infra.teacher_llamacpp import LlamaCppTeacher
from ..infra.teacher_openrouter import TeacherProvider as OpenRouterTeacher
from ..infra.model_manager import ensure_student_model_available, ensure_teacher_gguf_simple
from ..domain.tokenizer_align import (
    TokenizerAdapter,
    compute_token_spans,
    align_sequences_by_char_spans,
    merge_teacher_steps_logprobs,
)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_student(model_id_or_path: str, device: str, dtype: Optional[str] = None):
    tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    # Auto-select dtype by device if not provided
    torch_dtype = None
    dtype_norm = (dtype or "").lower()
    if not dtype_norm:
        if device == "cuda" and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16
        elif device == "mps" and hasattr(torch, "float16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = None
    else:
        if dtype_norm in ("bf16", "bfloat16") and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype_norm in ("fp16", "float16", "half") and hasattr(torch, "float16"):
            torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        dtype=torch_dtype,
        trust_remote_code=True,
    )
    if device == "cuda":
        model.to("cuda")
    elif device == "mps":
        model.to("mps")
    model.train()
    return tok, model


def _logsoftmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1)


def _jsd_beta_torch(student_log: torch.Tensor, teacher_log: torch.Tensor, beta: float) -> torch.Tensor:
    """Numerically-stable JSD in log-space.

    - Casts to float32 to avoid half over/underflow (common on MPS)
    - Renormalizes logs before exponentiating
    """
    s = student_log.float()
    t = teacher_log.float()
    # Normalize logs defensively (they may already be normalized)
    s = s - torch.logsumexp(s, dim=-1)
    t = t - torch.logsumexp(t, dim=-1)
    p = s.exp()
    q = t.exp()
    pi = beta * p + (1.0 - beta) * q
    denom = (pi.sum() + 1e-12)
    pi = pi / denom
    pi_log = (pi + 1e-12).log()
    kl_p_pi = (p * (s - pi_log)).sum()
    kl_q_pi = (q * (t - pi_log)).sum()
    return beta * kl_p_pi + (1.0 - beta) * kl_q_pi


def train_once() -> None:
    load_dotenv()

    # Optional Weights & Biases init (single env: WANDB_API_KEY)
    wandb_run = None
    try:
        import wandb  # type: ignore
        api_key = os.getenv("WANDB_API_KEY")
        entity = os.getenv("WANDB_ENTITY")
        project = os.getenv("WANDB_PROJECT", "opdistill")
        if api_key:
            wandb.login(key=api_key)
            run_name = f"kd_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb_run = wandb.init(project=project, entity=entity, name=run_name, config={})
            log_event("wandb_init", run_name=run_name, project=project, entity=entity)
    except Exception:
        wandb_run = None

    data_path = os.getenv("DATA_PATH")
    output_dir = os.getenv("OUTPUT_DIR")
    if not data_path:
        raise SystemExit("DATA_PATH must be set in .env")
    if not output_dir:
        raise SystemExit("OUTPUT_DIR must be set in .env")

    num_samples_env = os.getenv("NUM_SAMPLES", "all")
    if num_samples_env.lower() in ("all", "none", "-1"):
        num_samples: Optional[int] = None
    else:
        try:
            num_samples = int(num_samples_env)
        except ValueError:
            num_samples = None

    max_steps = int(os.getenv("MAX_STEPS", "200"))
    save_every = int(os.getenv("SAVE_EVERY", "50"))
    lr = float(os.getenv("LEARNING_RATE", "5e-5"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "128"))
    # Encourage longer generations by default (no new envs required)
    # At least half the cap, up to 256 tokens, and ignore EOS to reduce early stops.
    min_new_tokens = min(256, max_new_tokens)
    ignore_eos = True
    beta = float(os.getenv("BETA", "0.0"))
    # KD/SFT blend and gating
    lambda_sft = float(os.getenv("LAMBDA_SFT", "0.0"))
    lambda_kd = float(os.getenv("LAMBDA_KD", "1.0"))
    kd_min_teacher_logp = float(os.getenv("KD_MIN_TEACHER_LOGP", "-3.0"))
    kd_debug = os.getenv("KD_DEBUG", "0") == "1"
    teacher_top_k = int(os.getenv("TEACHER_TOP_K", "20"))
    # Validation / early stopping
    val_fraction = float(os.getenv("VAL_FRACTION", "0.1"))
    eval_every = int(os.getenv("EVAL_EVERY", "50"))
    patience = int(os.getenv("PATIENCE", "3"))
    monitor_metric = os.getenv("MONITOR_METRIC", "val_total_loss")  # or val_top1_agreement
    monitor_mode = os.getenv("MONITOR_MODE", "min")  # min or max
    val_max_samples_env = os.getenv("VAL_MAX_SAMPLES")
    val_max_samples: Optional[int] = int(val_max_samples_env) if val_max_samples_env else None

    # Models
    student_model_id = os.getenv("STUDENT_MODEL_PATH") or os.getenv("STUDENT_MODEL_ID")
    if not student_model_id:
        raise SystemExit("Set STUDENT_MODEL_PATH or STUDENT_MODEL_ID in .env")
    if os.getenv("STUDENT_MODEL_PATH") is None and "," not in student_model_id and "/" in student_model_id:
        # download to local cache if repo id
        student_model_id = ensure_student_model_available(student_model_id, cache_root="models/hf")

    teacher_backend = os.getenv("TEACHER_BACKEND", "hf").lower()
    teacher_model_id = os.getenv("TEACHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

    device = _device()
    log_file = init_event_log("runs")
    log_event("train_start", log_file=log_file, data_path=data_path, output_dir=output_dir, device=device, max_steps=max_steps)
    if wandb_run:
        try:
            import wandb
            wandb.config.update({
                "data_path": data_path,
                "output_dir": output_dir,
                "max_steps": max_steps,
                "save_every": save_every,
                "lr": lr,
                "max_new_tokens": max_new_tokens,
                "lambda_sft": os.getenv("LAMBDA_SFT", "0.0"),
                "lambda_kd": os.getenv("LAMBDA_KD", "1.0"),
                "kd_min_teacher_logp": os.getenv("KD_MIN_TEACHER_LOGP", "-30"),
                "teacher_top_k": os.getenv("TEACHER_TOP_K", "50"),
                "val_fraction": os.getenv("VAL_FRACTION", "0.1"),
                "eval_every": os.getenv("EVAL_EVERY", "50"),
                "patience": os.getenv("PATIENCE", "3"),
                "monitor_metric": os.getenv("MONITOR_METRIC", "val_top1_agreement"),
                "monitor_mode": os.getenv("MONITOR_MODE", "max"),
            }, allow_val_change=True)
        except Exception:
            pass

    prompts = load_prompts_any(data_path, limit=num_samples)
    if not prompts:
        raise SystemExit("No prompts loaded")
    # Split train/val
    split_at = max(1, int(len(prompts) * (1.0 - val_fraction))) if len(prompts) > 1 else 1
    train_prompts = prompts[:split_at]
    val_prompts = prompts[split_at:] if val_fraction > 0 and len(prompts) > split_at else prompts

    # Helper: auto-resume from OUTPUT_DIR/best -> latest step-* -> final
    def _find_resume_dir(base_out: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        outp = Path(base_out)
        if not outp.exists():
            return None, None, None
        # best
        best = outp / "best"
        if best.exists():
            if (best / "adapter_config.json").exists() or (best / "adapter_model.safetensors").exists():
                return str(best), "peft", "best"
            if (best / "config.json").exists():
                return str(best), "full", "best"
        # step-*
        step_dirs = []
        for p in outp.iterdir():
            if p.is_dir() and p.name.startswith("step-"):
                try:
                    n = int(p.name.split("-", 1)[1])
                except Exception:
                    n = -1
                step_dirs.append((n, p))
        if step_dirs:
            step_dirs.sort(key=lambda x: x[0], reverse=True)
            cand = step_dirs[0][1]
            if (cand / "adapter_config.json").exists() or (cand / "adapter_model.safetensors").exists():
                return str(cand), "peft", cand.name
            if (cand / "config.json").exists():
                return str(cand), "full", cand.name
        # final
        final = outp / "final"
        if final.exists():
            if (final / "adapter_config.json").exists() or (final / "adapter_model.safetensors").exists():
                return str(final), "peft", "final"
            if (final / "config.json").exists():
                return str(final), "full", "final"
        return None, None, None

    def _find_latest_step_num(base_out: str) -> int:
        outp = Path(base_out)
        latest = 0
        if not outp.exists():
            return latest
        for p in outp.iterdir():
            if p.is_dir() and p.name.startswith("step-"):
                try:
                    n = int(p.name.split("-", 1)[1])
                except Exception:
                    n = 0
                if n > latest:
                    latest = n
        return latest

    resume_dir, resume_kind, resume_tag = _find_resume_dir(output_dir)
    resume_step = _find_latest_step_num(output_dir)
    # Seed best_metric from a persisted file if present to avoid overwriting "best" on new runs
    best_metric_file = Path(output_dir) / "best_metrics.json"
    persisted_best_metric: Optional[float] = None
    persisted_best_step: Optional[int] = None
    try:
        if best_metric_file.exists():
            import json as _json
            meta = _json.loads(best_metric_file.read_text())
            if isinstance(meta, dict):
                v = meta.get("value")
                if isinstance(v, (int, float)):
                    persisted_best_metric = float(v)
                s = meta.get("step")
                if isinstance(s, int):
                    persisted_best_step = s
    except Exception:
        persisted_best_metric = None
        persisted_best_step = None
    if resume_kind == "full" and resume_dir is not None:
        student_model_id = resume_dir
        log_event("resume_checkpoint", path=resume_dir, kind=resume_kind, tag=resume_tag)

    # Build student and teacher
    s_tok, s_model = _build_student(student_model_id, device=device, dtype=os.getenv("STUDENT_DTYPE"))
    try:
        if hasattr(s_model, "config"):
            s_model.config.use_cache = False
        if hasattr(s_model, "gradient_checkpointing_enable"):
            s_model.gradient_checkpointing_enable()
    except Exception:
        pass

    # Optional: LoRA for KD to reduce memory (particularly on MPS)
    use_kd_lora = os.getenv("KD_LORA", "0") == "1"
    if use_kd_lora:
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise RuntimeError("peft not installed but KD_LORA=1 is set. Install 'peft'.")
        kd_lora_r = int(os.getenv("KD_LORA_R", "16"))
        kd_lora_alpha = int(os.getenv("KD_LORA_ALPHA", "32"))
        kd_lora_dropout = float(os.getenv("KD_LORA_DROPOUT", "0.05"))
        kd_lora_targets = os.getenv("KD_LORA_TARGET", "q_proj,k_proj,v_proj,o_proj").split(",")
        if resume_kind == "peft" and resume_dir is not None and PeftModel is not None:
            s_model = PeftModel.from_pretrained(s_model, resume_dir)
            log_event("resume_checkpoint", path=resume_dir, kind=resume_kind, tag=resume_tag)
        else:
            lcfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=kd_lora_r,
                lora_alpha=kd_lora_alpha,
                lora_dropout=kd_lora_dropout,
                target_modules=kd_lora_targets,
                bias="none",
            )
            s_model = get_peft_model(s_model, lcfg)
    # Teacher backend selection
    if teacher_backend == "openrouter":
        openrouter_model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-7b-instruct")
        model_spec = f"openrouter:{openrouter_model}"
        use_mock = os.getenv("OPENROUTER_API_KEY") in (None, "")
        if use_mock:
            log_event("teacher_openrouter_mock_mode", reason="no_api_key")
        teacher = OpenRouterTeacher(model_spec=model_spec, use_mock=use_mock)
        log_event("teacher_init", backend="openrouter", model=openrouter_model, mock=use_mock)
    elif teacher_backend == "gguf":
        teacher_spec = os.getenv("TEACHER_GGUF") or os.getenv("TEACHER_GGUF_REPO") or os.getenv("TEACHER_GGUF_HF_REPO_ID")
        teacher_path = os.getenv("TEACHER_GGUF_PATH")
        if teacher_path and os.path.isfile(teacher_path):
            resolved_gguf = teacher_path
        elif teacher_spec:
            resolved_gguf = ensure_teacher_gguf_simple(teacher_spec, preferred_quant=os.getenv("TEACHER_PREFERRED_QUANT", "Q4_K_M"))
        else:
            raise SystemExit("Set TEACHER_GGUF (local .gguf or HF repo/URL) or switch TEACHER_BACKEND to 'openrouter' or 'hf'")
        teacher = LlamaCppTeacher(model_path=resolved_gguf, use_mock=False)
        log_event("teacher_init", backend="gguf", model_path=resolved_gguf)
    elif teacher_backend == "hf":
        teacher = LocalHFTeacher(
            model_id_or_path=teacher_model_id,
            use_mock=False,
            device=("cuda" if device == "cuda" else ("mps" if device == "mps" else "cpu")),
            dtype=os.getenv("TEACHER_DTYPE", "bf16"),
            flash_attn=os.getenv("TEACHER_FLASH_ATTENTION", "1") == "1",
            top_k=teacher_top_k,
        )
        log_event("teacher_init", backend="hf", model_id=teacher_model_id)
    else:
        raise SystemExit(f"Invalid TEACHER_BACKEND={teacher_backend}. Use 'openrouter', 'gguf', or 'hf'")

    # Optimize only trainable params (LoRA if enabled). Guard against empty list on resume.
    trainable_params = [p for p in s_model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        # If we resumed a PEFT adapter and params are frozen, unfreeze LoRA params
        try:
            unfroze = 0
            for n, p in s_model.named_parameters():
                if ("lora_" in n) or ("adapter" in n):
                    p.requires_grad = True
                    unfroze += 1
            if unfroze > 0:
                trainable_params = [p for p in s_model.parameters() if p.requires_grad]
        except Exception:
            pass
    if len(trainable_params) == 0:
        # Fallback: train full model to avoid empty optimizer (last resort on MPS)
        for p in s_model.parameters():
            p.requires_grad = True
        trainable_params = list(s_model.parameters())
        log_event("kd_warning_full_ft", reason="no_trainable_lora_params_found")
    optimizer = AdamW(trainable_params, lr=lr)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate() -> Dict[str, float]:
        # Compute val metrics without grad
        s_model.eval()
        total_loss = 0.0
        total_groups = 0
        mapped = 0
        agree = 0
        tokens_mapped = 0
        teacher_logp_sum = 0.0
        teacher_logp_count = 0
        eval_items = val_prompts[: (val_max_samples or len(val_prompts))]

        class _TokAdapterEval(TokenizerAdapter):
            def __init__(self, tok): self.tok = tok
            def encode(self, text: str): return self.tok.encode(text, add_special_tokens=False)
            def decode(self, ids: List[int]): return self.tok.decode(ids, skip_special_tokens=True)
            def token_str(self, token_id: int): return self.tok.convert_ids_to_tokens([token_id])[0]
            def vocab_strings(self) -> List[str]:
                vocab = self.tok.get_vocab(); inv = {v: k for k, v in vocab.items()}; return [inv[i] for i in range(len(inv))]

        s_adapter = _TokAdapterEval(s_tok)
        student_vocab = s_adapter.vocab_strings()
        student_lookup: Dict[str, int] = {s_adapter.token_str(i): i for i in range(len(student_vocab))}

        for item in eval_items:
            prompt_text = render_prompt(item)
            with torch.no_grad():
                enc_prompt = s_tok(prompt_text, return_tensors="pt")
                if device == "cuda": enc_prompt = {k: v.cuda() for k, v in enc_prompt.items()}
                elif device == "mps": enc_prompt = {k: v.to("mps") for k, v in enc_prompt.items()}
                do_sample_flag = (device == "cuda" and os.getenv("DO_SAMPLE", "1") == "1")
                gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample_flag, return_dict_in_generate=True, output_scores=False)
                if do_sample_flag: gen_kwargs.update(dict(top_k=50))
                gen_out = s_model.generate(**enc_prompt, **gen_kwargs)
                gen_ids = gen_out.sequences[0][enc_prompt["input_ids"].shape[1] :]
                completion_text = s_tok.decode(gen_ids, skip_special_tokens=True)
            t_scores = teacher.score_completion_logprobs(prompt_text, completion_text)

            # teacher logp over student's completion tokens (if available)
            for step_i in t_scores.per_step:
                if step_i.chosen_token and step_i.chosen_token in step_i.token_logprobs:
                    teacher_logp_sum += step_i.token_logprobs[step_i.chosen_token]
                    teacher_logp_count += 1

            s_spans = compute_token_spans(completion_text, s_adapter)
            # teacher spans
            t_tok_strings = []
            for step_i in t_scores.per_step:
                if step_i.chosen_token: t_tok_strings.append(step_i.chosen_token)
                elif step_i.token_logprobs: t_tok_strings.append(max(step_i.token_logprobs.items(), key=lambda kv: kv[1])[0])
                else: t_tok_strings.append("")
            t_spans = []
            cursor = 0
            for tok in t_tok_strings:
                if tok:
                    idx = completion_text.find(tok, cursor)
                    if idx == -1: start = cursor; end = min(len(completion_text), cursor + len(tok))
                    else: start = idx; end = idx + len(tok)
                else: start = end = cursor
                t_spans.append(type("TS", (), {"start": start, "end": end, "text": completion_text[start:end], "token_id": -1}))
                cursor = end
            alignment = align_sequences_by_char_spans(s_spans, t_spans)

            # forward student to get logprobs for agreement/loss
            with torch.no_grad():
                enc_comp = s_tok(completion_text, add_special_tokens=False, return_tensors="pt")
                if device == "cuda":
                    enc_comp = {k: v.cuda() for k, v in enc_comp.items()}
                elif device == "mps":
                    enc_comp = {k: v.to("mps") for k, v in enc_comp.items()}
                input_ids = torch.cat([enc_prompt["input_ids"], enc_comp["input_ids"]], dim=1)
                attn = torch.ones_like(input_ids)
                if device == "cuda": input_ids = input_ids.cuda(); attn = attn.cuda()
                elif device == "mps": input_ids = input_ids.to("mps"); attn = attn.to("mps")
                logits = s_model(input_ids=input_ids, attention_mask=attn).logits
                slog = torch.log_softmax(logits, dim=-1)[0]
            prompt_len = enc_prompt["input_ids"].shape[1]

            # compute per-group loss and agreement
            teacher_step_str_dists: List[Dict[str, float]] = [step.token_logprobs for step in t_scores.per_step]
            for s_group, t_group in zip(alignment.student_groups, alignment.teacher_groups):
                if not s_group or not t_group:
                    continue
                total_groups += 1
                merged_t = merge_teacher_steps_logprobs(teacher_step_str_dists, t_group)
                # teacher best token -> student id
                if merged_t.token_logprobs:
                    t_best_str = max(merged_t.token_logprobs.items(), key=lambda kv: kv[1])[0]
                    teacher_sid = student_lookup.get(t_best_str)
                else:
                    teacher_sid = None

                # student merged distribution over ids
                idxs = [prompt_len + i - 1 for i in s_group if (prompt_len + i - 1) >= 0]
                if not idxs:
                    continue
                s_merged = slog[idxs, :].sum(dim=0)
                s_merged = s_merged - torch.logsumexp(s_merged, dim=-1)
                # loss on intersection of ids present from teacher mapping (GKD-like over mapped id only)
                if teacher_sid is not None:
                    tokens_mapped += 1
                    mapped += 1
                    student_top_id = int(torch.argmax(s_merged).item())
                    if student_top_id == teacher_sid:
                        agree += 1
                    # simple loss: JS divergence with a delta at teacher_sid
                    # Use a dtype-safe finite log(0) proxy to avoid overflow on float16 (MPS)
                    if s_merged.dtype in (getattr(torch, "float16", None), getattr(torch, "bfloat16", None)):
                        fill = -1e4
                    else:
                        fill = -1e9
                    t_vec = torch.full_like(s_merged, fill_value=fill)
                    t_vec[teacher_sid] = 0.0  # log(1)
                    t_vec = t_vec - torch.logsumexp(t_vec, dim=-1)
                    jsd = _jsd_beta_torch(s_merged, t_vec, beta)
                    total_loss += float(jsd.detach().cpu())

        n = max(1, total_groups)
        val = {
            "val_total_loss": total_loss / n,
            "val_top1_agreement": (agree / max(1, mapped)),
            "val_mapped_fraction": (mapped / n),
            "val_avg_tokens_mapped": (tokens_mapped / max(1, len(eval_items))),
            "val_teacher_logp_student": (teacher_logp_sum / max(1, teacher_logp_count)),
        }
        log_event("val", **val)
        return val

    # Start from the latest saved step to avoid overwriting earlier checkpoints
    step = int(resume_step)
    best_metric = persisted_best_metric
    bad_evals = 0
    while step < max_steps:
        item = train_prompts[step % len(train_prompts)]
        prompt_text = render_prompt(item)

        # 1) Sample student completion (no grad)
        s_model.eval()
        with torch.no_grad():
            enc_prompt = s_tok(prompt_text, return_tensors="pt")
            if device == "cuda":
                enc_prompt = {k: v.cuda() for k, v in enc_prompt.items()}
            elif device == "mps":
                enc_prompt = {k: v.to("mps") for k, v in enc_prompt.items()}
            do_sample_flag = (device == "cuda" and os.getenv("DO_SAMPLE", "1") == "1")
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample_flag,
                return_dict_in_generate=True,
                output_scores=False,
            )
            # Optional: force a minimum length and/or ignore EOS to avoid early stops
            gen_kwargs["min_new_tokens"] = min_new_tokens
            if ignore_eos:
                gen_kwargs["eos_token_id"] = None
            if do_sample_flag:
                gen_kwargs.update(dict(top_k=50))
            gen_out = s_model.generate(**enc_prompt, **gen_kwargs)
            gen_ids = gen_out.sequences[0][enc_prompt["input_ids"].shape[1] :]
            completion_text = s_tok.decode(gen_ids, skip_special_tokens=True)
            student_token_strs = s_tok.convert_ids_to_tokens(gen_ids)
            student_token_strs = s_tok.convert_ids_to_tokens(gen_ids)

        # 2) Score teacher logprobs on the sampled completion
        t_scores = teacher.score_completion_logprobs(prompt_text, completion_text)

        # 3) Build alignment groups on completion text
        class _TokAdapter(TokenizerAdapter):
            def __init__(self, tok):
                self.tok = tok
            def encode(self, text: str) -> List[int]:
                return self.tok.encode(text, add_special_tokens=False)
            def decode(self, ids: List[int]) -> str:
                return self.tok.decode(ids, skip_special_tokens=True)
            def token_str(self, token_id: int) -> str:
                return self.tok.convert_ids_to_tokens([token_id])[0]
            def vocab_strings(self) -> List[str]:
                vocab = self.tok.get_vocab()
                inv = {v: k for k, v in vocab.items()}
                return [inv[i] for i in range(len(inv))]

        s_adapter = _TokAdapter(s_tok)
        s_spans = compute_token_spans(completion_text, s_adapter)
        t_tok_strings = []
        for step_i in t_scores.per_step:
            if step_i.chosen_token:
                t_tok_strings.append(step_i.chosen_token)
            elif step_i.token_logprobs:
                t_tok_strings.append(max(step_i.token_logprobs.items(), key=lambda kv: kv[1])[0])
            else:
                t_tok_strings.append("")
        # Build teacher spans from strings
        t_spans = []
        cursor = 0
        for tok in t_tok_strings:
            if tok:
                idx = completion_text.find(tok, cursor)
                if idx == -1:
                    start = cursor
                    end = min(len(completion_text), cursor + len(tok))
                else:
                    start = idx
                    end = idx + len(tok)
            else:
                start = end = cursor
            t_spans.append(type("TS", (), {"start": start, "end": end, "text": completion_text[start:end], "token_id": -1}))
            cursor = end
        alignment = align_sequences_by_char_spans(s_spans, t_spans)

        # 4) Forward student on prompt+completion to get differentiable logits
        s_model.train()
        enc_comp = s_tok(completion_text, add_special_tokens=False, return_tensors="pt")
        if device == "cuda":
            enc_comp = {k: v.cuda() for k, v in enc_comp.items()}
        elif device == "mps":
            enc_comp = {k: v.to("mps") for k, v in enc_comp.items()}
        input_ids = torch.cat([enc_prompt["input_ids"], enc_comp["input_ids"]], dim=1)
        attn = torch.ones_like(input_ids)
        if device == "cuda":
            input_ids = input_ids.cuda()
            attn = attn.cuda()
        elif device == "mps":
            input_ids = input_ids.to("mps")
            attn = attn.to("mps")
        t0 = time.time()
        with (torch.amp.autocast("cuda") if device == "cuda" else contextlib.nullcontext()):
            outputs = s_model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits  # [1, seq_len, vocab]
            logprobs = _logsoftmax(logits)

        prompt_len = enc_prompt["input_ids"].shape[1]

        # 5) Build teacher merged distributions per aligned position and compute JSD loss
        teacher_step_str_dists: List[Dict[str, float]] = [step.token_logprobs for step in t_scores.per_step]
        # student lookup: token string -> id
        vocab = s_adapter.vocab_strings()
        student_lookup: Dict[str, int] = {s_adapter.token_str(i): i for i in range(len(vocab))}

        kd_losses: List[torch.Tensor] = []
        # Step metrics
        step_groups = 0
        step_mapped = 0
        step_agree = 0
        step_tokens_mapped = 0
        # Teacher logp of student's tokens (approx by position-wise match up to min length)
        min_len = min(len(student_token_strs), len(teacher_step_str_dists))
        teacher_logp_sum = 0.0
        teacher_logp_count = 0
        for i in range(min_len):
            lp = teacher_step_str_dists[i].get(student_token_strs[i])
            if lp is not None:
                teacher_logp_sum += lp
                teacher_logp_count += 1
        for s_group, t_group in zip(alignment.student_groups, alignment.teacher_groups):
            if not s_group or not t_group:
                continue
            step_groups += 1
            # Merge teacher steps
            merged_t = merge_teacher_steps_logprobs(teacher_step_str_dists, t_group)
            # Map to student ids
            t_ids: List[int] = []
            t_logvals: List[float] = []
            for tok_str, lp in merged_t.token_logprobs.items():
                sid = student_lookup.get(tok_str)
                if sid is not None:
                    t_ids.append(sid)
                    t_logvals.append(lp)
            if not t_ids:
                if kd_debug:
                    log_event(
                        "kd_group_no_map",
                        step=step,
                        s_group_size=len(s_group),
                        t_group_size=len(t_group),
                    )
                continue
            step_mapped += 1
            t_log = torch.tensor(t_logvals, device=input_ids.device)
            # Merge student steps: sum logprobs at each step, then renormalize
            # Positions in logprobs predicting token at pos t are at index t-1
            idxs = [prompt_len + i - 1 for i in s_group if (prompt_len + i - 1) >= 0]
            if not idxs:
                continue
            s_logs = logprobs[0, idxs, :]  # [G, V]
            s_merged = s_logs.sum(dim=0)
            s_merged = s_merged - torch.logsumexp(s_merged, dim=-1)
            s_sel = s_merged[t_ids]
            # Agreement on this group
            student_top_id = int(torch.argmax(s_merged).item())
            teacher_best_id = t_ids[int(torch.argmax(t_log).item())]
            step_tokens_mapped += 1
            if student_top_id == teacher_best_id:
                step_agree += 1
            # KD gating: ensure teacher shows sufficient confidence on any mapped token
            max_teacher_logp = float(torch.max(t_log).item()) if t_log.numel() > 0 else float("-inf")
            if (max_teacher_logp < kd_min_teacher_logp) and lambda_kd > 0.0:
                if kd_debug:
                    log_event(
                        "kd_group_gated",
                        step=step,
                        reason="low_teacher_conf",
                        max_teacher_logp=max_teacher_logp,
                        threshold=kd_min_teacher_logp,
                        mapped_ids=len(t_ids),
                        s_group_size=len(s_group),
                        t_group_size=len(t_group),
                    )
            else:
                # JSD loss
                jsd = _jsd_beta_torch(s_sel, t_log, beta)
                kd_losses.append(jsd)

        # Build blended loss
        kd_loss_val = torch.stack(kd_losses).mean() if kd_losses else torch.tensor(0.0, device=input_ids.device)
        sft_loss_val = torch.tensor(0.0, device=input_ids.device)
        if lambda_sft > 0.0 and getattr(item, "assistant", None):
            # Supervised CE on assistant
            s_model.train()
            enc_ass = s_tok(item.assistant, add_special_tokens=False, return_tensors="pt")
            if device == "cuda": enc_ass = {k: v.cuda() for k, v in enc_ass.items()}
            elif device == "mps": enc_ass = {k: v.to("mps") for k, v in enc_ass.items()}
            s_inputs = torch.cat([enc_prompt["input_ids"], enc_ass["input_ids"]], dim=1)
            s_attn = torch.ones_like(s_inputs)
            # Build labels: -100 for prompt, assistant ids for CE
            labels = torch.full_like(s_inputs, fill_value=-100)
            labels[:, enc_prompt["input_ids"].shape[1] :] = s_inputs[:, enc_prompt["input_ids"].shape[1] :]
            out = s_model(input_ids=s_inputs, attention_mask=s_attn, labels=labels)
            sft_loss_val = out.loss

        # If no KD terms and no SFT label, skip this step to avoid zero-grad loss
        if (len(kd_losses) == 0) and not (lambda_sft > 0.0 and getattr(item, "assistant", None)):
            log_event("train_skip_no_signal", step=step)
            step += 1
            continue
        loss = lambda_kd * kd_loss_val + lambda_sft * sft_loss_val
        if scaler is not None:
            scaler.scale(loss).backward()
            try:
                torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=float(os.getenv("KD_GRAD_CLIP", "1.0")))
            except Exception:
                pass
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(s_model.parameters(), max_norm=float(os.getenv("KD_GRAD_CLIP", "1.0")))
            except Exception:
                pass
            optimizer.step()
        s_model.zero_grad(set_to_none=True)
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        step_ms = (time.time() - t0) * 1000.0
        # Build a compact teacher scoring preview (first 10 steps)
        teacher_preview = []
        try:
            preview_n = min(10, len(teacher_step_str_dists), len(student_token_strs))
            for i in range(preview_n):
                dist = teacher_step_str_dists[i]
                stud_tok = student_token_strs[i]
                lp_stud = dist.get(stud_tok)
                top_tok, top_lp = (None, None)
                if dist:
                    top_tok, top_lp = max(dist.items(), key=lambda kv: kv[1])
                teacher_preview.append({
                    "i": i,
                    "student_tok": stud_tok,
                    "teacher_lp_student": lp_stud,
                    "teacher_top": top_tok,
                    "teacher_lp_top": top_lp,
                })
        except Exception:
            teacher_preview = []

        log_event(
            "train_step",
            step=step,
            loss=float(loss.detach().cpu()),
            kd_loss=float(kd_loss_val.detach().cpu()) if kd_losses else 0.0,
            sft_loss=float(sft_loss_val.detach().cpu()) if lambda_sft > 0.0 and getattr(item, "assistant", None) else 0.0,
            step_ms=round(step_ms, 2),
            prompt_len=int(enc_prompt["input_ids"].shape[1]),
            completion_tokens=int(len(gen_ids)),
            completion_chars=int(len(completion_text)),
            mapped_fraction_step=(step_mapped / max(1, step_groups)),
            top1_agreement_step=(step_agree / max(1, step_tokens_mapped)),
            avg_tokens_mapped_step=float(step_tokens_mapped),
            teacher_logp_student_step=(teacher_logp_sum / max(1, teacher_logp_count)),
            lr=float(optimizer.param_groups[0].get("lr", lr)),
            prompt_user=getattr(item, "user", None),
            student_completion_preview=completion_text[:512],
            teacher_scoring_preview=teacher_preview,
        )
        if wandb_run:
            try:
                import wandb
                wandb.log({
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "kd_loss": float(kd_loss_val.detach().cpu()) if kd_losses else 0.0,
                    "sft_loss": float(sft_loss_val.detach().cpu()) if (lambda_sft > 0.0 and getattr(item, "assistant", None)) else 0.0,
                    "mapped_fraction_step": (step_mapped / max(1, step_groups)),
                    "top1_agreement_step": (step_agree / max(1, step_tokens_mapped)),
                    "teacher_logp_student_step": (teacher_logp_sum / max(1, teacher_logp_count)),
                    "completion_tokens": int(len(gen_ids)),
                })
            except Exception:
                pass

        if (step + 1) % save_every == 0:
            ckpt_dir = out_dir / f"step-{step+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            s_model.eval()
            s_model.save_pretrained(ckpt_dir)
            s_tok.save_pretrained(ckpt_dir)
            log_event("checkpoint_saved", path=str(ckpt_dir))
            s_model.train()

        step += 1

        # Periodic evaluation and early stopping
        if (step % eval_every) == 0:
            val = evaluate()
            current = val.get(monitor_metric)
            if wandb_run:
                try:
                    import wandb
                    v = {f"val/{k}": v for k, v in val.items()}
                    wandb.log(v)
                except Exception:
                    pass
            if current is not None:
                improved = False
                if best_metric is None:
                    improved = True
                else:
                    if monitor_mode == "min":
                        improved = current < best_metric
                    else:
                        improved = current > best_metric
                if improved:
                    best_metric = current
                    bad_evals = 0
                    best_dir = out_dir / "best"
                    best_dir.mkdir(parents=True, exist_ok=True)
                    s_model.eval()
                    s_model.save_pretrained(best_dir)
                    s_tok.save_pretrained(best_dir)
                    log_event("best_checkpoint_saved", metric=monitor_metric, value=current, path=str(best_dir))
                    # Persist best metric for future resumes
                    try:
                        import json as _json
                        best_metric_file.write_text(_json.dumps({
                            "metric": monitor_metric,
                            "mode": monitor_mode,
                            "value": float(current),
                            "step": int(step),
                            "timestamp": datetime.now().isoformat(),
                        }, indent=2))
                    except Exception:
                        pass
                    if wandb_run:
                        try:
                            import wandb
                            wandb.summary["best_metric"] = current
                        except Exception:
                            pass
                    s_model.train()
                else:
                    bad_evals += 1
                    log_event("early_stopping_counter", bad_evals=bad_evals, patience=patience)
                    if bad_evals >= patience:
                        log_event("early_stopping_triggered", best_metric=best_metric, monitor=monitor_metric)
                        break

    # Final save
    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    s_model.eval()
    s_model.save_pretrained(final_dir)
    s_tok.save_pretrained(final_dir)
    log_event("train_complete", path=str(final_dir))
    if wandb_run:
        try:
            import wandb
            wandb.summary["final_path"] = str(final_dir)
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    train_once()
