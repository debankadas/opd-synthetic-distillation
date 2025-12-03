#!/usr/bin/env bash
set -euo pipefail

# Create a Miniconda environment and install llama-cpp-python.
# Usage: ./scripts/setup_conda_llamacpp.sh gold-llamacpp 3.10

ENV_NAME=${1:-opdisitillation}
PY_VERSION=${2:-3.10}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

conda create -y -n "$ENV_NAME" python="$PY_VERSION"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Determine if Apple Silicon (arm64). Enable Metal only on arm64.
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  export CMAKE_ARGS="-DGGML_METAL=on"
  export LLAMA_METAL=on
  echo "Enabling GGML_METAL for Apple Silicon (arm64)."
else
  echo "Building CPU-only llama-cpp-python (no Metal)."
fi

python -m pip install --upgrade pip wheel setuptools
python -m pip install "llama-cpp-python>=0.2.90"
python -m pip install -r requirements.txt

echo "OK: conda env '$ENV_NAME' ready with llama-cpp-python installed."
echo "Activate with: conda activate $ENV_NAME"
echo "Next, place a 7B GGUF at models/teacher-7b.Q4_K_M.gguf (or edit TEACHER_GGUF_PATH)."

