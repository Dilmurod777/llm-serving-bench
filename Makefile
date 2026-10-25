PY := .venv/bin/python
LLAMA := $(HOME)/engines/llama.cpp/llama-b11819/llama-server
VLLM := $(HOME)/engines/vllm-venv/bin/vllm

.PHONY: setup venv install models ollama-model serve-ollama serve-llamacpp serve-vllm serve-vllm-fp8 serve-vllm-spec bench verify chart test

setup: venv install models ollama-model

venv:
	python3 -m venv .venv
	.venv/bin/pip install -e . pytest

install:
	bash scripts/install_engines.sh

models:
	bash scripts/fetch_models.sh

ollama-model:
	sudo systemctl start ollama
	ollama create qwen3-8b-q4 -f scripts/Modelfile.ollama

serve-ollama:
	OLLAMA_NUM_PARALLEL=32 OLLAMA_CONTEXT_LENGTH=1024 ollama serve

serve-llamacpp:
	$(LLAMA) -m models/Qwen3-8B-Q4_K_M.gguf -c 32768 --parallel 32 -ngl 99 --port 8080

serve-vllm:
	$(VLLM) serve Qwen/Qwen3-8B-AWQ --max-model-len 1024 --max-num-seqs 32 --gpu-memory-utilization 0.55 --port 8000

serve-vllm-fp8:
	$(VLLM) serve Qwen/Qwen3-8B-FP8 --max-model-len 1024 --max-num-seqs 32 --gpu-memory-utilization 0.55 --port 8000

serve-vllm-spec:
	$(VLLM) serve Qwen/Qwen3-8B-AWQ --max-model-len 1024 --max-num-seqs 32 --gpu-memory-utilization 0.55 --port 8000 \
		--speculative-config '{"model":"Qwen/Qwen3-0.6B","num_speculative_tokens":5}'

bench:
	bash scripts/run_all.sh

verify:
	PROFILE=verify ENGINES=$(or $(ENGINE),llamacpp) bash scripts/run_all.sh

chart:
	$(PY) -m bench.plot

test:
	$(PY) -m pytest -q
