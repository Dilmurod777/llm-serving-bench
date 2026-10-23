#!/usr/bin/env bash
# Start each engine alone, sweep concurrency 1/8/32, stop it, next engine.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
LLAMA="$HOME/engines/llama.cpp/llama-b11819/llama-server"
VLLM="$HOME/engines/vllm-venv/bin/vllm"
# The Windows Ollama keeps the embedding model resident and steals ~4.6 GB.
WIN_OLLAMA=/mnt/c/Users/RAIDER/AppData/Local/Programs/Ollama/ollama.exe

evict_windows_models() {
  [ -x "$WIN_OLLAMA" ] || return 0
  for m in $("$WIN_OLLAMA" ps 2>/dev/null | awk 'NR>1 {print $1}' || true); do
    "$WIN_OLLAMA" stop "$m" >/dev/null 2>&1 || true
  done
}

stop_all() {
  sudo systemctl stop ollama >/dev/null 2>&1 || true
  pkill -f llama-server >/dev/null 2>&1 || true
  pkill -f "vllm serve" >/dev/null 2>&1 || true
  evict_windows_models
  sleep 5
}

wait_http() {
  for _ in $(seq 1 600); do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "engine did not become ready: $1" >&2
  return 1
}

ENGINES="${ENGINES:-ollama,llamacpp,vllm}"
want() { case ",$ENGINES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Servers that preload the model must be checked before they start, not after.
check_idle() { "$PY" -c 'from bench.vram import assert_idle; assert_idle()'; }

sweep() {
  # SHORT=1 adds the 32-token pass, SHORT=only runs nothing but that pass.
  if [ "${SHORT:-0}" != "only" ]; then
    for c in 1 8 32; do
      "$PY" -m bench --engine "$1" --concurrency "$c" --skip-idle-check
    done
    # Time to first token: a one-token completion is prefill plus one decode step.
    for c in 1 8 32; do
      "$PY" -m bench --engine "$1" --concurrency "$c" --max-tokens 1 --skip-idle-check
    done
  fi
  if [ "${SHORT:-0}" != "0" ]; then
    for c in 1 8 32; do
      "$PY" -m bench --engine "$1" --concurrency "$c" --max-tokens 32 \
        --out "results/short/$1-c$c.json" --skip-idle-check
    done
  fi
}

serve_vllm() {
  local name=$1 model=$2
  shift 2
  echo "=== $name ==="
  check_idle
  # flashinfer's JIT sampler needs nvcc; the torch sampler avoids a CUDA toolkit install
  export VLLM_USE_FLASHINFER_SAMPLER=0
  "$VLLM" serve "$model" --max-model-len 4096 --gpu-memory-utilization 0.55 \
    --max-num-batched-tokens 8192 \
    --port 8000 "$@" >"results/engine-logs/$name.log" 2>&1 &
  wait_http http://127.0.0.1:8000/health
  sweep "$name"
  stop_all
}

mkdir -p results/engine-logs results/short
stop_all

if want ollama; then
echo "=== ollama ==="
check_idle
sudo systemctl start ollama
wait_http http://127.0.0.1:11434/api/version
sweep ollama
stop_all
fi

if want llamacpp; then
echo "=== llama.cpp ==="
export LD_LIBRARY_PATH="$HOME/engines/llama.cpp/llama-b11819"
check_idle
"$LLAMA" -m models/Qwen3-8B-Q4_K_M.gguf -c 32768 --parallel 32 -ngl 99 --port 8080 >results/engine-logs/llamacpp.log 2>&1 &
wait_http http://127.0.0.1:8080/health
sweep llamacpp
stop_all
fi

if want vllm; then
serve_vllm vllm Qwen/Qwen3-8B-AWQ
fi

if want vllm-fp8; then
# DeepGEMM JITs its FP8 kernels through deep_jit, which wants a CUDA toolkit;
# the fallback kernel needs none.
export VLLM_USE_DEEP_GEMM=0
serve_vllm vllm-fp8 Qwen/Qwen3-8B-FP8
fi

if want vllm-spec; then
serve_vllm vllm-spec Qwen/Qwen3-8B-AWQ \
  --speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'
fi

echo "=== done ==="
"$PY" -m bench.plot
