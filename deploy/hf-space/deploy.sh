#!/usr/bin/env bash
# Upload the app to the Hugging Face Space.
# Requires the hf CLI and a login with a write token: `hf auth login`.
set -euo pipefail

SPACE="${SPACE:-perfectlyx/indonesian-food-scanner}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$ROOT"
mkdir -p "$STAGE/model" "$STAGE/static/uploads" "$STAGE/static/results"
cp app.py space_app.py nutrition_data.csv "$STAGE/"
cp model/best.pt "$STAGE/model/"
cp -R templates "$STAGE/"
cp static/styles.css static/app.js "$STAGE/static/"
touch "$STAGE/static/uploads/.gitkeep" "$STAGE/static/results/.gitkeep"
cp deploy/hf-space/README.md deploy/hf-space/requirements.txt deploy/hf-space/packages.txt "$STAGE/"

HF="$(command -v hf || echo "$ROOT/.venv/bin/hf")"
"$HF" upload "$SPACE" "$STAGE" . --repo-type=space --commit-message "${1:-Deploy from GitHub repo}"
echo "Deployed: https://huggingface.co/spaces/$SPACE"
