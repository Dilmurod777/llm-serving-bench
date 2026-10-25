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

# baseline = 1/8/32 at 32k context; wide = 64/128 at 128 slots; long = 4k-token prompts.
PROFILE="${PROFILE:-baseline}"
case "$PROFILE" in
  wide) CONCS="64 128"; SLOTS=128; CTX=512 ;;
  long) CONCS="1 4 8"; SLOTS=8; CTX=8192 ;;
  verify) CONCS="1"; SLOTS=32; CTX=1024 ;;
  *)    CONCS="1 8 32"; SLOTS=32; CTX=1024 ;;
esac

OUTDIR=results
if [ "$PROFILE" = verify ] || [ -n "${MODEL:-}" ]; then
  OUTDIR=results/local
fi
SHORTDIR="$OUTDIR/short"
WIDEDIR="$OUTDIR/wide"
LONGDIR="$OUTDIR/long"

# Ollama takes its slot count and per-slot context from the service unit.
set_ollama_slots() {
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment="OLLAMA_NUM_PARALLEL=$2"
Environment="OLLAMA_CONTEXT_LENGTH=$1"
EOF
  sudo systemctl daemon-reload
}

# Servers that preload the model must be checked before they start, not after.
check_idle() { "$PY" -c 'from bench.vram import assert_idle; assert_idle()'; }

bench() {
  local engine=$1
  local extra=()
  shift
  if [ -n "${MODEL:-}" ]; then extra=(--model "$MODEL"); fi
  "$PY" -m bench --engine "$engine" --slots "$SLOTS" --ctx "$CTX" \
    "${extra[@]}" "$@"
}

sweep() {
  case "$PROFILE" in
    verify)
      for c in $CONCS; do
        bench "$1" --concurrency "$c" --skip-idle-check \
          --out "$OUTDIR/$1-c$c.json"
      done
      return 0 ;;
    wide)
      for c in $CONCS; do
        bench "$1" --concurrency "$c" --skip-idle-check \
          --out "$WIDEDIR/$1-c$c.json"
      done
      return 0 ;;
    long)
      for c in $CONCS; do
        bench "$1" --concurrency "$c" --max-tokens 1 \
          --prompts data/prompts_long.jsonl --out "$LONGDIR/$1-c$c.json" \
          --skip-idle-check
      done
      return 0 ;;
  esac
  # SHORT=1 adds the 32-token pass, SHORT=only runs nothing but that pass.
  if [ "${SHORT:-0}" != "only" ]; then
    for c in $CONCS; do
      bench "$1" --concurrency "$c" --skip-idle-check \
        --out "$OUTDIR/$1-c$c.json"
    done
    # Time to first token: a one-token completion is prefill plus one decode step.
    for c in $CONCS; do
      bench "$1" --concurrency "$c" --max-tokens 1 --skip-idle-check \
        --out "$OUTDIR/$1-c$c-ttft.json"
    done
  fi
  if [ "${SHORT:-0}" != "0" ]; then
    for c in $CONCS; do
      bench "$1" --concurrency "$c" --max-tokens 32 \
        --out "$SHORTDIR/$1-c$c.json" --skip-idle-check
    done
  fi
}

serve_vllm() {
  local name=$1 model=${MODEL:-$2}
  shift 2
  echo "=== $name ==="
  check_idle
  # flashinfer's JIT sampler needs nvcc; the torch sampler avoids a CUDA toolkit install
  export VLLM_USE_FLASHINFER_SAMPLER=0
  "$VLLM" serve "$model" --max-model-len "$CTX" --gpu-memory-utilization 0.55 \
    --max-num-batched-tokens 8192 --max-num-seqs "$SLOTS" \
    --port 8000 "$@" >"results/engine-logs/$name.log" 2>&1 &
  wait_http http://127.0.0.1:8000/health
  sweep "$name"
  stop_all
}

mkdir -p results/engine-logs "$OUTDIR" "$SHORTDIR" "$WIDEDIR" "$LONGDIR"
stop_all

if want ollama; then
echo "=== ollama ==="
check_idle
set_ollama_slots "$CTX" "$SLOTS"
sudo systemctl start ollama
wait_http http://127.0.0.1:11434/api/version
sweep ollama
stop_all
fi

if want llamacpp; then
echo "=== llama.cpp ==="
export LD_LIBRARY_PATH="$HOME/engines/llama.cpp/llama-b11819"
check_idle
"$LLAMA" -m models/Qwen3-8B-Q4_K_M.gguf -c $((SLOTS * CTX)) --parallel "$SLOTS" -ngl 99 --port 8080 >results/engine-logs/llamacpp.log 2>&1 &
wait_http http://127.0.0.1:8080/health
if [ -n "${MODEL:-}" ]; then
  echo "MODEL is ignored for llamacpp: the server loads models/Qwen3-8B-Q4_K_M.gguf"
  MODEL= sweep llamacpp
else
  sweep llamacpp
fi
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

if [ "$PROFILE" = verify ]; then
  for engine in ${ENGINES//,/ }; do
    "$PY" -m bench.compare "results/$engine-c1.json" "$OUTDIR/$engine-c1.json"
  done
  exit 0
fi

echo "=== done ==="
"$PY" -m bench.plot
