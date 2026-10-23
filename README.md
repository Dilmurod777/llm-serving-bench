# llm-serving-bench

Same model, same GPU, same 200 prompts. Three serving stacks. One chart.

![throughput by engine and concurrency](docs/chart.svg)

<!-- RESULTS_TABLE -->

At single-stream all three are within 8% of each other (105–114 tok/s): one 8B model
on one GPU costs what it costs. Batching is where they part. At 32 concurrent
requests vLLM serves **1655 tok/s** ($0.06 per 1M output tokens), llama.cpp 951
($0.10) and Ollama 743 ($0.13) — a 2.2× spread for the same weights on the same
card.

| engine | concurrency | output tok/s | p50 ms | p95 ms | peak VRAM MiB | $/1M output tokens |
|---|---|---|---|---|---|---|
| ollama | 1 | 105.46 | 2371.8 | 2393.4 | 12735 | 0.92 |
| llamacpp | 1 | 109.74 | 2306.4 | 2327.8 | 12635 | 0.89 |
| vllm | 1 | 113.56 | 2229.5 | 2244.5 | 15742 | 0.86 |
| ollama | 8 | 195.29 | 9201.0 | 16053.9 | 12741 | 0.50 |
| llamacpp | 8 | 216.25 | 7738.1 | 12806.5 | 12645 | 0.45 |
| vllm | 8 | 771.68 | 2520.8 | 2607.7 | 15906 | 0.13 |
| ollama | 32 | 742.78 | 10059.0 | 10402.9 | 12741 | 0.13 |
| llamacpp | 32 | 950.55 | 7716.4 | 8176.9 | 12645 | 0.10 |
| vllm | 32 | 1655.34 | 4183.7 | 4572.8 | 16050 | 0.06 |

Regenerate this table and the chart with `make chart` (or `python -m bench.plot`).

## What this measures

One 8B model served by three stacks on one RTX 5090 Laptop (24 GB), inside WSL2
Ubuntu 24.04 so the kernel, driver and VRAM accounting are identical for all three:

| engine | version | how it serves |
|---|---|---|
| Ollama | 0.40.8 | `ollama serve`, model built from the GGUF below |
| llama.cpp | b11819, CUDA 13.4 prebuilt | `llama-server -c 32768 --parallel 32 -ngl 99` |
| vLLM | 0.32.0 (torch 2.13.0+cu130) | `vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096` |

Every request is a **raw completion** — no chat template, `temperature=0`,
`max_tokens=256`, seed 0 — against the same 200-passage prompt set
(`data/prompts.jsonl`, sha256 pinned in `bench/prompts.py`). Prompt text comes from
the FAA *AC 43.13-1B* (US government work, public domain): 200 passages of
150–400 characters, sampled deterministically.

Each of the 9 runs is 200 requests at concurrency 1, 8 and 32. The GPU is required
to be idle (≥14 GB free, i.e. no model of ours resident) before a run starts, and
engines are stopped between runs so no two stacks are ever resident together. Peak
VRAM is the absolute `nvidia-smi` peak per sweep, so the desktop's own ~7 GB are
included in every number.

## Models

| model | size | licence | source |
|---|---|---|---|
| `Qwen3-8B-Q4_K_M.gguf` (Ollama, llama.cpp) | 4.7 GB | Apache-2.0 | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| `Qwen/Qwen3-8B-AWQ` (vLLM) | 5.5 GB | Apache-2.0 | [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ) |

The quantisation differs on purpose: vLLM's fast path on sm_120 is AWQ/Marlin,
while the GGUF engines share the same bytes. So the vLLM row is *not* a pure
serving-stack comparison — it is "vLLM at its best quantisation" against
"llama.cpp/Ollama at a 4-bit GGUF". The README says it instead of hiding it.

Weights are never in this repo. `make models` downloads and verifies them into
`models/`, which is git-ignored.

## Cost

```
$ / 1M output tokens = GPU price per hour / (output tokens per second × 3600) × 1e6
```

with `GPU_PRICE_PER_HOUR = 0.35` in `bench/config.py` (a rented 5090). It is the
only number in the table that is an assumption rather than a measurement, and it is
printed so you can substitute your own.

## Reproduce

Three commands on a machine with an RTX 5090 and WSL2 Ubuntu 24.04:

```bash
git clone https://github.com/Dilmurod777/llm-serving-bench && cd llm-serving-bench
make install models ollama-model
make bench
```

`make bench` runs `scripts/run_all.sh`: start one engine, sweep concurrency 1/8/32,
stop it, next engine, then render `docs/chart.svg` and print the Markdown table.
`make test` runs the 19 unit tests (summary math, the three HTTP adapters against
mocked servers, the VRAM gate, the sweep runner, prompt-set integrity, the chart).

## Layout

```
bench/       config, engine adapters, sweep runner, summary math, VRAM sampler, chart
data/        200 fixed prompts (sha256-pinned)
scripts/     install_engines.sh, fetch_models.sh, run_all.sh, Modelfile
results/     one json + 200-record jsonl per (engine, concurrency)
docs/        chart.svg, design and plan
tests/       19 pytest tests
```

## Caveats

- Laptop 5090, not a datacentre card: clocks throttle, and the Windows desktop
  holds ~7 GB of the card.
- VRAM compares each engine's own memory **policy** (Ollama's 32k-context
  reservation, llama.cpp's per-slot cache, vLLM's utilisation cap), not just weights.
- Token counts are each server's own accounting (`eval_count`,
  `timings.predicted_n`, `usage.completion_tokens`).
- Numbers are one machine on one day. The point is the shape of the curve and the
  cost per token, not a leaderboard position.
- The curve is not monotonic: for both GGUF engines concurrency 8 is the least
  efficient point (about 2× the single-stream rate), while 32 slots reach 7–9×
  (vLLM: 15×). That is the servers' step batching growing with the number of active
  slots, not a measurement error — two independent Ollama runs agree within 10%.
- vLLM runs with `VLLM_USE_FLASHINFER_SAMPLER=0`. Its flashinfer sampler JIT-compiles
  C++ on first use and demands a full CUDA toolkit; the torch sampler avoids that.
  It is the sampling kernel, not the model path, and at `temperature=0` it costs
  nothing measurable.
