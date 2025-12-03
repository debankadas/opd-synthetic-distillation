#!/usr/bin/env bash
set -euo pipefail

# Load .env into the environment if present
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# Training with a HF teacher (CUDA) and saving checkpoints.

export TEACHER_MODEL_ID=${TEACHER_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}
export TEACHER_BACKEND=hf
export TEACHER_DEVICE=${TEACHER_DEVICE:-cuda}
export TEACHER_DTYPE=${TEACHER_DTYPE:-bf16}
export TEACHER_FLASH_ATTENTION=${TEACHER_FLASH_ATTENTION:-1}
export TEACHER_TOP_K=${TEACHER_TOP_K:-20}

# Student
export STUDENT_DTYPE=${STUDENT_DTYPE:-bf16}

# Training config
: "${DATA_PATH:?Set DATA_PATH in .env or before calling this script}"
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen0_5b_gold}
export MAX_STEPS=${MAX_STEPS:-200}
export SAVE_EVERY=${SAVE_EVERY:-50}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}

python -m src.app.train_gold

