PY := .venv/bin/python
LLAMA := $(HOME)/engines/llama.cpp/llama-b11819/llama-server
VLLM := $(HOME)/engines/vllm-venv/bin/vllm

.PHONY: install models ollama-model serve-ollama serve-llamacpp serve-vllm bench chart test

install:
	bash scripts/install_engines.sh

models:
	bash scripts/fetch_models.sh

ollama-model:
	ollama create qwen3-8b-q4 -f scripts/Modelfile.ollama

serve-ollama:
	ollama serve

serve-llamacpp:
	$(LLAMA) -m models/Qwen3-8B-Q4_K_M.gguf -c 32768 --parallel 32 -ngl 99 --port 8080

serve-vllm:
	$(VLLM) serve Qwen/Qwen3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.85 --port 8000

bench:
	bash scripts/run_all.sh

chart:
	$(PY) -m bench.plot

test:
	$(PY) -m pytest -q
