#!/usr/bin/env bash
# Start the DR-Grade React frontend dev server (Vite).
# Talks to the FastAPI backend via the /api proxy (default http://localhost:8002).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/frontend"

if [[ ! -d node_modules ]]; then
    echo "📦 Installing dependencies ..."
    npm install
fi

echo "🚀 Starting frontend dev server ..."
exec npm run dev
