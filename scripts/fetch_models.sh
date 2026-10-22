#!/usr/bin/env bash
# Weights are never committed: this script is the only way to get them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

# name | url | sha256
ENTRIES=(
  "Qwen3-8B-Q4_K_M.gguf|https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf|TO_FILL"
)

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r name url sha <<< "${entry}"
  if [ -f "models/${name}" ] && echo "${sha}  models/${name}" | sha256sum -c --status -; then
    echo "ok: models/${name}"
    continue
  fi
  curl -fL --retry 3 -o "models/${name}" "${url}"
  echo "${sha}  models/${name}" | sha256sum -c -
done

echo
echo "vLLM pulls Qwen/Qwen3-8B-AWQ into ~/.cache/huggingface on first 'make serve-vllm'."
