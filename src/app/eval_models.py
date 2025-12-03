from __future__ import annotations

import argparse
import json
import os
# (no dataclasses used directly here)
import time
import re
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore

from .prepare_dataset import load_prompts_sampled, render_prompt

# Fast-eval tuning knobs (no CLI):
# - Adjust here to trade accuracy for speed during quick iterations
EVAL_GLOBAL_LIMIT = 100            # examples per task (approx; tasks may expand per-choice)
EVAL_GLOBAL_NUM_FEWSHOT = 0        # force zero-shot
EVAL_GLOBAL_MAX_GEN_TOKS = 64      # cap generations for generative tasks
EVAL_GLOBAL_BATCH_SIZE = 4         # batch size used by lm-eval

# RetailCo quick settings
RETAIL_EVAL_MAX_SAMPLES = 100      # random sample size from your Retail file
RETAIL_EVAL_MAX_NEW_TOKENS = 256   # generation cap for RetailCo completions
RETAIL_EVAL_RANDOM_SEED = 42       # sampling seed for reproducibility

# Reduce tokenizers fork warnings that can stall on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cleanup_device_memory(device: str) -> None:
    # Proactively release cached memory between model runs
    try:
        if device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()  # type: ignore[attr-defined]
            except Exception:
                pass
        elif device == "mps" and hasattr(torch, "mps"):
            try:
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    # Encourage Python to free tensors and tokenizer/model objects
    try:
        gc.collect()
    except Exception:
        pass


def load_model(model_id_or_path: str, device: str, dtype: Optional[str] = None):
    # Detect PEFT adapter directory (no config.json but has adapter_config.json)
    model_path = Path(model_id_or_path)
    is_adapter = model_path.exists() and model_path.is_dir() and (model_path / "adapter_config.json").exists() and not (model_path / "config.json").exists()

    base_path = model_id_or_path
    adapter_path: Optional[str] = None
    if is_adapter:
        try:
            acfg = json.loads((model_path / "adapter_config.json").read_text())
            base_path = acfg.get("base_model_name_or_path") or base_path
            adapter_path = str(model_path)
        except Exception:
            adapter_path = None

    # Prefer tokenizer from adapter if present, else from base
    tok_src = adapter_path if (adapter_path and (model_path / "tokenizer_config.json").exists()) else base_path
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    torch_dtype = None
    dtype_norm = (dtype or "").lower()
    if not dtype_norm:
        if device == "cuda" and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16
        elif device == "mps":
            torch_dtype = None
    else:
        if dtype_norm in ("bf16", "bfloat16") and hasattr(torch, "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype_norm in ("fp16", "float16") and hasattr(torch, "float16"):
            torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch_dtype, trust_remote_code=True)
    # Attach adapter if provided
    if adapter_path and PeftModel is not None:
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception:
            pass
    if device == "cuda":
        model.to("cuda")
    elif device == "mps":
        model.to("mps")
    model.eval()
    try:
        if hasattr(model, "config"):
            model.config.use_cache = True
    except Exception:
        pass
    return tok, model


def gen_completion(tok, model, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    inp = tok(prompt, return_tensors="pt")
    if device == "cuda":
        inp = {k: v.cuda() for k, v in inp.items()}
    elif device == "mps":
        inp = {k: v.to("mps") for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
        )
    text = tok.decode(out.sequences[0][inp["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def eval_retailco(
    model_id: str,
    data_path: str,
    device: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    rows = load_prompts_sampled(data_path, limit=max_samples, seed=RETAIL_EVAL_RANDOM_SEED)
    tok, model = load_model(model_id, device=device)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    n = 0
    exact = 0
    exact_norm = 0
    rougeL = 0.0
    t_start = time.time()

    def _norm(s: str) -> str:
        # Lowercase, collapse inner whitespace, and trim common punct at ends
        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.strip("\"'`“”‘’.,!?;:()[]{}<>")
        return s
    for idx, it in enumerate(rows):
        if not it.user:
            continue
        # prefer explicit assistant if present
        gold = (it.assistant or "").strip()
        if not gold:
            # skip samples without labels in SFT/DPO format
            continue
        prompt = render_prompt(it)
        pred = gen_completion(tok, model, prompt, device=device, max_new_tokens=max_new_tokens)
        n += 1
        if pred == gold:
            exact += 1
        if _norm(pred) == _norm(gold):
            exact_norm += 1
        rL = scorer.score(pred, gold)["rougeL"].fmeasure
        rougeL += rL
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"  retail progress: {idx+1}/{len(rows)} processed in {elapsed:.1f}s", flush=True)
            # Help MPS avoid memory fragmentation during long loops
            if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    result = {
        "n": n,
        "exact_match": (exact / max(1, n)),
        "exact_match_norm": (exact_norm / max(1, n)),
        "rougeL_f1": (rougeL / max(1, n)),
    }
    # Explicitly drop model and tokenizer to free VRAM before global eval
    try:
        del tok
        del model
    except Exception:
        pass
    cleanup_device_memory(device)
    return result


def eval_global(
    model_id: str,
    tasks: List[str],
    device: str,
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    max_gen_toks: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    # Use lm-eval harness programmatically
    try:
        from lm_eval import evaluator
    except Exception:
        raise RuntimeError("lm-eval not installed. pip install lm-eval==0.4.3")
    # Support PEFT/LoRA directories by passing adapter to lm-eval (if supported)
    adapter_dir = None
    base_model_for_adapter = None
    try:
        p = Path(model_id)
        if p.exists() and p.is_dir():
            ac = p / "adapter_config.json"
            cfg = p / "config.json"
            if ac.exists() and not cfg.exists():
                adapter_dir = str(p)
                with open(ac, "r") as fh:
                    acfg = json.load(fh)
                base_model_for_adapter = acfg.get("base_model_name_or_path")
    except Exception:
        adapter_dir = None
        base_model_for_adapter = None

    if adapter_dir and base_model_for_adapter:
        base_args = f"pretrained={base_model_for_adapter},peft={adapter_dir},device={device},dtype=auto"
    else:
        base_args = f"pretrained={model_id},device={device},dtype=auto"
    try_args = base_args + ",apply_chat_template=True"
    # Some lm-eval versions forward unknown args to HF model.init, causing an error.
    # Try with chat template first; on failure, fall back to base args.
    # Apply defaults from module-level knobs when not provided
    if limit is None:
        limit = EVAL_GLOBAL_LIMIT
    if num_fewshot is None:
        num_fewshot = EVAL_GLOBAL_NUM_FEWSHOT
    if max_gen_toks is None:
        max_gen_toks = EVAL_GLOBAL_MAX_GEN_TOKS

    try:
        res = evaluator.simple_evaluate(
            model="hf",
            model_args=try_args,
            tasks=tasks,
            batch_size=EVAL_GLOBAL_BATCH_SIZE,
            num_fewshot=num_fewshot,
            limit=limit,
            gen_kwargs={"max_gen_toks": max_gen_toks} if max_gen_toks is not None else None,
        )
    except Exception as e:
        # If PEFT arg not supported by this lm-eval version, fall back to merged temporary model
        if adapter_dir and base_model_for_adapter:
            try:
                import tempfile
                from peft import PeftModel
                tmpdir = tempfile.mkdtemp(prefix="opd_eval_merge_")
                tok = AutoTokenizer.from_pretrained(adapter_dir if (Path(adapter_dir)/"tokenizer_config.json").exists() else base_model_for_adapter, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(base_model_for_adapter, trust_remote_code=True)
                peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
                try:
                    peft_model = peft_model.merge_and_unload()
                except Exception:
                    pass
                peft_model.save_pretrained(tmpdir)
                try:
                    tok.save_pretrained(tmpdir)
                except Exception:
                    pass
                merged_args = f"pretrained={tmpdir},device={device},dtype=auto"
                res = evaluator.simple_evaluate(
                    model="hf",
                    model_args=merged_args,
                    tasks=tasks,
                    batch_size=EVAL_GLOBAL_BATCH_SIZE,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    gen_kwargs={"max_gen_toks": max_gen_toks} if max_gen_toks is not None else None,
                )
                try:
                    import shutil
                    shutil.rmtree(tmpdir, ignore_errors=True)
                except Exception:
                    pass
                # proceed to summarization below
                pass
            except Exception as e2:
                # Fall back to base args without PEFT
                res = evaluator.simple_evaluate(
                    model="hf",
                    model_args=f"pretrained={base_model_for_adapter or model_id},device={device},dtype=auto",
                    tasks=tasks,
                    batch_size=EVAL_GLOBAL_BATCH_SIZE,
                    num_fewshot=num_fewshot,
                    limit=limit,
                    gen_kwargs={"max_gen_toks": max_gen_toks} if max_gen_toks is not None else None,
                )
        else:
            res = evaluator.simple_evaluate(
                model="hf",
                model_args=base_args,
                tasks=tasks,
                batch_size=EVAL_GLOBAL_BATCH_SIZE,
                num_fewshot=num_fewshot,
                limit=limit,
                gen_kwargs={"max_gen_toks": max_gen_toks} if max_gen_toks is not None else None,
            )
    # Extract primary metrics per task
    summary: Dict[str, Dict[str, float]] = {}
    metric_priority = [
        "acc_norm",
        "acc",
        "exact",
        "f1",
        "mc2",
        "mc1",
        "rougeL",
        "bleu",
        "score",
    ]
    for task, vals in res.get("results", {}).items():
        if not isinstance(vals, dict):
            continue
        chosen: Optional[tuple[str, float]] = None
        for k in metric_priority:
            v = vals.get(k)
            if isinstance(v, (int, float)):
                chosen = (k, float(v))
                break
        if chosen is None:
            # pick first numeric non-stderr value
            numeric_items = [(k, v) for k, v in vals.items() if isinstance(v, (int, float))]
            numeric_items = [kv for kv in numeric_items if "stderr" not in kv[0]]
            if numeric_items:
                k, v = numeric_items[0]
                chosen = (k, float(v))
        if chosen is not None:
            summary[task] = {chosen[0]: chosen[1]}
    return summary


def plot_bars(df: pd.DataFrame, title: str, out_path: Path, value_col: str, hue_col: str, x_col: str):
    # Work on a copy so we can prettify labels
    data = df.copy()
    if x_col in data.columns:
        data[x_col] = data[x_col].astype(str).str.replace("checkpoint_", "ckpt_", regex=False)
        data[x_col] = data[x_col].str.replace("checkpoint-", "ckpt-", regex=False)
    # Keep the first-seen order for categories
    order = list(dict.fromkeys(data[x_col].tolist())) if x_col in data.columns else None

    # Dynamic width: more categories -> wider fig
    n_cats = len(order) if order is not None else 6
    width = max(8.0, 1.2 * n_cats)
    plt.figure(figsize=(width, 5), dpi=150)
    # Avoid duplicate bars when hue equals x
    use_hue = (hue_col != x_col)
    if use_hue:
        ax = sns.barplot(data=data, x=x_col, y=value_col, hue=hue_col, order=order)
    else:
        ax = sns.barplot(data=data, x=x_col, y=value_col, order=order, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(value_col)
    # Rotate crowded x labels
    ax.set_xlabel("")
    for tick in ax.get_xticklabels():
        tick.set_rotation(28)
        tick.set_horizontalalignment("right")
    ax.margins(x=0.01)
    # If metric looks like a probability, lock to [0, 1]
    try:
        vmin = float(data[value_col].min())
        vmax = float(data[value_col].max())
        if 0.0 <= vmin and vmax <= 1.0:
            ax.set_ylim(0.0, 1.0)
    except Exception:
        pass
    # Show labels even for zero-height bars; annotate when all zeros
    try:
        all_zero = True
        for container in ax.containers:
            heights = []
            for bar in container:
                h = getattr(bar, "get_height", None)
                h = float(h()) if callable(h) else float(getattr(bar, "height", 0.0) or 0.0)
                heights.append(h)
                if abs(h) > 1e-9:
                    all_zero = False
            if heights:
                ax.bar_label(container, labels=[f"{h:.2f}" for h in heights], padding=3, fontsize=9)
        if all_zero:
            # Make room for labels at zero and add annotation
            ax.set_ylim(-0.05, 1.0)
            ax.annotate(
                "All values are 0.00",
                xy=(0.5, 0.5), xycoords="axes fraction",
                ha="center", va="center", fontsize=11, color="#666",
            )
    except Exception:
        pass
    # Simplify legend when hue duplicates x or is small-value
    leg = ax.get_legend()
    if leg and not use_hue:
        leg.remove()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    load_dotenv()
    # No external logging; keep eval script minimal
    p = argparse.ArgumentParser(description="Evaluate base, SFT, and KD models on global + custom RetailCo evals")
    p.add_argument("--base", default=os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct"))
    p.add_argument("--sft", default=os.getenv("STUDENT_MODEL_PATH"), help="Optional SFT model path")
    p.add_argument("--kd", default=os.getenv("KD_MODEL_PATH"), help="OPD/KD model path")
    p.add_argument("--kd-root", default=os.getenv("KD_SCAN_ROOT"), help="If set, scan this folder for KD checkpoints (best, final, step-*) and include them")
    p.add_argument("--retail-data", default=os.getenv("DATA_PATH", "sample_training_data/retailco_training_data_sft_conversations.json"))
    p.add_argument("--tasks", default=os.getenv("EVAL_TASKS", "arc_easy,arc_challenge,hellaswag,truthfulqa,ifeval"))
    # CLI args kept for backward-compat but not used for speed knobs
    p.add_argument("--max-samples", type=int, default=int(os.getenv("EVAL_MAX_SAMPLES", "200")))
    p.add_argument("--out", default=os.getenv("EVAL_OUTPUT_DIR", "reports"))
    p.add_argument("--retail-max-new", type=int, default=int(os.getenv("EVAL_RETAIL_MAX_NEW_TOKENS", "256")), help="Max new tokens for retail completions")
    p.add_argument("--skip-global", action="store_true", help="Skip lm-eval global tasks for speed")
    args = p.parse_args()

    device = get_device()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / f"eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    if args.base:
        models.append(("base", args.base))
    if args.sft:
        models.append(("sft", args.sft))
    # Always include the singular KD path if it exists
    if args.kd:
        models.append(("best-kd" if str(args.kd).endswith("best") else "kd", args.kd))

    # Optionally scan a root folder for KD checkpoints (best/final/step-*)
    if args.kd_root:
        root = Path(args.kd_root)
        if root.exists() and root.is_dir():
            seen_paths = {str(Path(p).resolve()) for _, p in models}
            # Search typical experiment subfolders one level deep
            exp_dirs = [d for d in root.iterdir() if d.is_dir()]
            # Also allow scanning the root itself if it directly contains checkpoints
            exp_dirs.append(root)
            found: list[tuple[str, str, int]] = []  # (label, path, sort_key)
            for exp in exp_dirs:
                for sub in exp.iterdir() if exp.is_dir() else []:
                    if not sub.is_dir():
                        continue
                    name = sub.name
                    # Only include typical checkpoint dirs
                    if name == "best" and (sub / "adapter_config.json").exists():
                        lbl = f"best-kd"
                        sortk = 10_000_000
                    elif name == "final" and ((sub / "config.json").exists() or (sub / "adapter_config.json").exists()):
                        lbl = f"final-kd"
                        sortk = 9_000_000
                    elif name.startswith("step-") and (sub / "adapter_config.json").exists():
                        try:
                            n = int(name.split("-", 1)[1])
                        except Exception:
                            n = -1
                        if n >= 0:
                            lbl = f"checkpoint_{n}-kd"
                            sortk = n
                        else:
                            continue
                    else:
                        continue
                    sp = str(sub.resolve())
                    if sp in seen_paths:
                        continue
                    seen_paths.add(sp)
                    found.append((lbl, sp, sortk))
            # Sort by step ascending, then final, then best
            found.sort(key=lambda t: t[2])
            for lbl, sp, _ in found:
                models.append((lbl, sp))

    global_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    report = {"device": device, "global_tasks": global_tasks, "retail_data": args.retail_data, "models": {}}

    # Evaluate each model
    for name, mid in models:
        print(f"Evaluating {name}: {mid}")
        # RetailCo (fast) with progress
        # Use module-level speed knobs regardless of CLI to avoid long runs
        retail = eval_retailco(
            mid,
            args.retail_data,
            device=device,
            max_samples=RETAIL_EVAL_MAX_SAMPLES,
            max_new_tokens=RETAIL_EVAL_MAX_NEW_TOKENS,
        )
        # Global (optional, can be slow)
        glob: Dict[str, float] | Dict[str, str]
        if (not args.skip_global) and global_tasks:
            try:
                glob = eval_global(mid, tasks=global_tasks, device=device)
            except Exception as e:
                glob = {"error": str(e)}
        else:
            glob = {"skipped": True}
        report["models"][name] = {"id": mid, "retail": retail, "global": glob}
        # Cleanup between models
        cleanup_device_memory(device)

    # Save JSON report
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    # Build comparison frames
    # Retail metrics
    retail_rows = []
    for name, _ in models:
        r = report["models"][name]["retail"]
        retail_rows.append({"model": name, "metric": "exact_match", "value": r.get("exact_match", 0.0)})
        retail_rows.append({"model": name, "metric": "exact_match_norm", "value": r.get("exact_match_norm", 0.0)})
        retail_rows.append({"model": name, "metric": "rougeL_f1", "value": r.get("rougeL_f1", 0.0)})
    df_retail = pd.DataFrame(retail_rows)
    if not df_retail.empty:
        for metric in df_retail["metric"].unique():
            sub = df_retail[df_retail["metric"] == metric]
            outp = out_dir / f"retail_{metric}.png"
            plot_bars(sub, f"RetailCo {metric}", outp, value_col="value", hue_col="model", x_col="model")
            # No external logging

    # Global metrics: one plot per task
    if isinstance(report["models"].get("base", {}).get("global"), dict):
        base_global = report["models"]["base"]["global"]
        tasks = [t for t in base_global.keys() if isinstance(base_global.get(t), dict) and base_global.get(t)]
        for task in tasks:
            rows = []
            metric_name = None
            for name, _ in models:
                g = report["models"][name]["global"]
                if isinstance(g, dict) and task in g and isinstance(g[task], dict):
                    # choose first numeric key
                    numeric = [(k, v) for k, v in g[task].items() if isinstance(v, (int, float))]
                    if not numeric:
                        continue
                    k, v = numeric[0]
                    metric_name = metric_name or k
                    rows.append({"model": name, "task": task, "value": float(v)})
            if rows and metric_name:
                df = pd.DataFrame(rows)
                outp = out_dir / f"global_{task}.png"
                plot_bars(df, f"{task} ({metric_name})", outp, value_col="value", hue_col="model", x_col="model")
                # No external logging

    print(f"Saved report and plots to {out_dir}")
    # No external logging


if __name__ == "__main__":
    main()
