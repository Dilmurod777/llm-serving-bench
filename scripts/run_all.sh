#!/usr/bin/env bash
# Start each engine alone, sweep concurrency 1/8/32, stop it, next engine.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LLAMA="$HOME/engines/llama.cpp/llama-b11819/llama-server"
VLLM="$HOME/engines/vllm-venv/bin/vllm"

stop_all() {
  sudo systemctl stop ollama >/dev/null 2>&1 || true
  pkill -f llama-server >/dev/null 2>&1 || true
  pkill -f "vllm serve" >/dev/null 2>&1 || true
  sleep 4
}

wait_http() {
  for _ in $(seq 1 180); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "engine did not become ready: $1" >&2
  return 1
}

sweep() {
  for c in 1 8 32; do
    "$PY" -m bench --engine "$1" --concurrency "$c"
  done
}

stop_all

echo "=== ollama ==="
sudo systemctl start ollama
wait_http http://127.0.0.1:11434/api/version
sweep ollama
stop_all

echo "=== llama.cpp ==="
"$LLAMA" -m models/Qwen3-8B-Q4_K_M.gguf -c 32768 --parallel 32 -ngl 99 --port 8080 >/tmp/llamacpp.log 2>&1 &
wait_http http://127.0.0.1:8080/health
sweep llamacpp
stop_all

echo "=== vllm ==="
"$VLLM" serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000 >/tmp/vllm.log 2>&1 &
wait_http http://127.0.0.1:8000/health
sweep vllm
stop_all

echo "=== done ==="
"$PY" -m bench.plot
