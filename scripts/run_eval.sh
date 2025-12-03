#!/usr/bin/env bash
set -euo pipefail

# Load .env if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Defaults (override via env or flags)
BASE_MODEL_ID=${BASE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
SFT_MODEL_PATH=${SFT_MODEL_PATH:-job_fe048090773e4a01803e13ff22da3a86}
KD_MODEL_PATH=${KD_MODEL_PATH:-outputs_mac/qwen0_5b_gold}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-reports}
EVAL_TASKS=${EVAL_TASKS:-arc_easy,arc_challenge,hellaswag,truthfulqa,ifeval}
EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-200}

# If KD_MODEL_PATH contains a specific experiment, include its best if present;
# otherwise scan the root for checkpoints.
KD_ROOT_ARG=("--kd-root" "$KD_MODEL_PATH")
KD_BEST_ARG=()
if [ -d "$KD_MODEL_PATH/best" ]; then
  KD_BEST_ARG=("--kd" "$KD_MODEL_PATH/best")
fi

python -m src.app.eval_models \
  --base "$BASE_MODEL_ID" \
  --sft "$SFT_MODEL_PATH" \
  "${KD_ROOT_ARG[@]}" \
  "${KD_BEST_ARG[@]}" \
  --retail-data "${DATA_PATH:-sample_training_data/retailco_training_data_sft_conversations.json}" \
  --tasks "$EVAL_TASKS" \
  --max-samples "$EVAL_MAX_SAMPLES" \
  --out "$EVAL_OUTPUT_DIR"
