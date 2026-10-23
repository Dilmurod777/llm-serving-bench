#!/usr/bin/env bash
# Engines run inside this Linux (WSL2) machine so they share one driver stack.
set -euo pipefail

TAG=b11819
LLAMA_DIR="${HOME}/engines/llama.cpp"
BASE="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}"
CUDA=13.4

if ! command -v ollama >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq zstd curl ca-certificates
  curl -fsSL https://ollama.com/install.sh | sh
fi

# vLLM runs torch.compile/inductor, which shells out to a C compiler
command -v cc >/dev/null 2>&1 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential

mkdir -p "${LLAMA_DIR}"
if [ ! -x "${LLAMA_DIR}/llama-${TAG}/llama-server" ]; then
  cd "${LLAMA_DIR}"
  curl -fL -o llama.tar.gz "${BASE}/llama-${TAG}-bin-ubuntu-cuda-${CUDA}-x64.tar.gz"
  curl -fL -o cudart.tar.gz "${BASE}/cudart-llama-${TAG}-bin-ubuntu-cuda-${CUDA}-x64.tar.gz"
  tar xzf llama.tar.gz
  tar xzf cudart.tar.gz
  # ggml-cuda.so links against libcudart/libcublas; keep them next to the binary
  cp "${LLAMA_DIR}"/cudart-llama-*/libc*.so.13 "${LLAMA_DIR}/llama-${TAG}/"
  rm -f llama.tar.gz cudart.tar.gz
  rm -rf "${LLAMA_DIR}"/cudart-llama-*/
fi

ollama --version
"${LLAMA_DIR}/llama-${TAG}/llama-server" --version | head -1
echo "engines installed: ollama, llama.cpp ${TAG}"
