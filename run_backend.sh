#!/usr/bin/env bash
# Start the DR-Grade FastAPI inference backend on port 8000.
#
#   ./run_backend.sh           # GPU (default: first visible card)
#   BACKEND_DEVICE=cpu ./run_backend.sh   # CPU fallback
#   CUDA_VISIBLE_DEVICES=1 ./run_backend.sh  # pin to physical GPU 1 (free card)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/backend"
# shellcheck disable=SC1091
source "$ROOT_DIR/setenv.sh"

PORT="${PORT:-8002}"
echo "🚀 Starting DR-Grade backend on :$PORT (device: ${BACKEND_DEVICE:-gpu})"
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT"
