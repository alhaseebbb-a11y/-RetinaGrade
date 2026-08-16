#!/usr/bin/env bash
# Activate the venv + expose the venv-bundled NVIDIA CUDA libraries to TensorFlow.
# Usage:  source setenv.sh
# (Must be run at the shell level — LD_LIBRARY_PATH cannot be set from inside Python
#  before tensorflow imports its CUDA plugins.)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [[ -z "${VIRTUAL_ENV:-}" || "$VIRTUAL_ENV" != "$VENV_DIR" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

NVIDIA_LIB_DIRS="$(find "$VENV_DIR/lib" -type d -path '*/nvidia/*/lib' 2>/dev/null | paste -sd: -)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"

# Use the venv's CUDA 12.3 ptxas (system CUDA 13 cannot compile for V100 / CC 7.0)
NVCC_BIN="$(find "$VENV_DIR/lib" -type d -path '*/nvidia/cuda_nvcc/bin' 2>/dev/null | head -1)"
if [[ -n "$NVCC_BIN" ]]; then
    export PATH="$NVCC_BIN:$PATH"
fi

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
echo "✓ venv: $(which python)"
echo "✓ NVIDIA libs: ${NVIDIA_LIB_DIRS}"
