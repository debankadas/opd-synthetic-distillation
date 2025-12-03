#!/bin/bash

# run_full_comparison.sh
# 1. Trains OPD model on synthetic data
# 2. Evaluates Base vs OPD model on benchmarks
# 3. Generates comparison plots

set -e  # Exit on error

# --- Configuration ---
# You can override these with environment variables
export DATA_PATH="${DATA_PATH:-synthetic_datasets/combined_100.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/synthetic_opd_qwen0.5b}"
export BASE_MODEL_ID="${BASE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}"
export TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
export MAX_STEPS="${MAX_STEPS:-300}"
export SAVE_EVERY="${SAVE_EVERY:-100}"
export EVAL_TASKS="${EVAL_TASKS:-arc_easy,arc_challenge,hellaswag,truthfulqa_mc2,winogrande,mmlu}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"

echo "=== Configuration ==="
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Base Model: $BASE_MODEL_ID"
echo "Teacher Model: $TEACHER_MODEL_ID"
echo "Eval Tasks: $EVAL_TASKS"
echo "====================="

# --- Step 1: Training ---
echo ""
echo ">>> Step 1: Training OPD Model..."
# Ensure .env exists or variables are exported. 
# We export key variables here to override .env or ensure they are set.
export LEARNING_RATE=5e-5
export MAX_NEW_TOKENS=128
export BETA=0.0

# Teacher backend selection: "openrouter" (API), "gguf" (local quantized), or "hf" (local full)
# Only set default if not already set (respects .env configuration)
if [ -z "$TEACHER_BACKEND" ]; then
    # Load from .env if available
    if [ -f .env ] && grep -q "^TEACHER_BACKEND=" .env; then
        export TEACHER_BACKEND=$(grep "^TEACHER_BACKEND=" .env | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' ')
    else
        # Default to gguf if .env doesn't specify
        export TEACHER_BACKEND=gguf
    fi
fi

echo "Teacher Backend: $TEACHER_BACKEND"

# Configure based on backend
if [ "$TEACHER_BACKEND" = "hf" ]; then
    export TEACHER_DTYPE=bf16
fi
# Optional LoRA
# export KD_LORA=1 

python -m src.app.train_gold

# --- Step 2: Evaluation ---
echo ""
echo ">>> Step 2: Running Comparison Evaluation..."

# The best checkpoint is usually saved in $OUTPUT_DIR/best
KD_MODEL_PATH="$OUTPUT_DIR/best"

if [ ! -d "$KD_MODEL_PATH" ]; then
    echo "Warning: Best checkpoint not found at $KD_MODEL_PATH. Checking for final..."
    if [ -d "$OUTPUT_DIR/final" ]; then
        KD_MODEL_PATH="$OUTPUT_DIR/final"
    else
        echo "Error: No checkpoint found in $OUTPUT_DIR"
        exit 1
    fi
fi

echo "Comparing:"
echo "  Base: $BASE_MODEL_ID"
echo "  OPD:  $KD_MODEL_PATH"

python -m src.app.eval_models \
    --base "$BASE_MODEL_ID" \
    --kd "$KD_MODEL_PATH" \
    --tasks "$EVAL_TASKS" \
    --max-samples "$EVAL_MAX_SAMPLES" \
    --out "reports/comparison_$(date +%Y%m%d_%H%M%S)"

echo ""
echo ">>> Complete! Check the reports/ directory for results and plots."
