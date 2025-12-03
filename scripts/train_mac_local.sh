#!/usr/bin/env bash
set -euo pipefail

# Load .env into the environment if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Training with llama.cpp teacher (Metal) and saving checkpoints.

export TEACHER_BACKEND=llama_cpp
export LLAMA_LOG_LEVEL=${LLAMA_LOG_LEVEL:-40}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}


# Training config
: "${DATA_PATH:?Set DATA_PATH in .env or before calling this script}"
export OUTPUT_DIR=${OUTPUT_DIR:-outputs_mac/qwen0_5b_gold}
export MAX_STEPS=${MAX_STEPS:-200}
export SAVE_EVERY=${SAVE_EVERY:-50}

ACCELERATE_USE_MPS_DEVICE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
python -m src.app.train_gold
