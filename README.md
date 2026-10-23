# llm-serving-bench

Same model, same GPU, same 200 prompts. Three serving stacks, seven charts.

![throughput by engine and concurrency](docs/chart.svg)

At single-stream all three are within 8% of each other (94–114 tok/s): one 8B model
on one GPU costs what it costs. Batching is where they part. At 32 concurrent
requests vLLM serves **1648 tok/s** ($0.06 per 1M output tokens), llama.cpp 971
($0.10) and Ollama 727 ($0.13) — a 2.3× spread for the same weights on the same
card.

| engine | concurrency | output tok/s | p50 ms | p95 ms | peak VRAM MiB | $/1M output tokens |
|---|---|---|---|---|---|---|
| ollama | 1 | 94.35 | 2593.4 | 2827.6 | 14116 | 1.03 |
| llamacpp | 1 | 109.41 | 2308.7 | 2359.3 | 13143 | 0.89 |
| vllm | 1 | 113.98 | 2221.1 | 2236.8 | 16088 | 0.85 |
| ollama | 8 | 187.01 | 10964.1 | 13830.4 | 14142 | 0.52 |
| llamacpp | 8 | 208.19 | 10040.8 | 12924.5 | 13098 | 0.47 |
| vllm | 8 | 772.34 | 2515.6 | 2604.5 | 16258 | 0.13 |
| ollama | 32 | 727.24 | 10311.0 | 10709.4 | 13271 | 0.13 |
| llamacpp | 32 | 970.65 | 7535.4 | 7897.5 | 13100 | 0.10 |
| vllm | 32 | 1647.92 | 4192.2 | 4660.0 | 16432 | 0.06 |

Regenerate this table and the chart with `make chart` (or `python -m bench.plot`).

## Time to first token

![p50 time to first token by engine and concurrency](docs/ttft.svg)

Same prompt set, one token per request: this is prefill plus a single decode step,
i.e. what a streaming client waits for before any text appears.

| engine | concurrency | p50 ms | p95 ms |
|---|---|---|---|
| ollama | 1 | 47.8 | 65.4 |
| llamacpp | 1 | 42.3 | 59.0 |
| vllm | 1 | 33.7 | 48.0 |
| ollama | 8 | 367.5 | 470.6 |
| llamacpp | 8 | 367.5 | 466.6 |
| vllm | 8 | 190.0 | 257.1 |
| ollama | 32 | 1304.1 | 1693.9 |
| llamacpp | 32 | 1247.7 | 1609.9 |
| vllm | 32 | 777.9 | 1042.4 |

Unloaded, all three are in the same 34–48 ms band; vLLM is fastest because its
scheduler admits the whole batch in one step. Under load the GGUF engines queue
behind their slots (1.25–1.30 s at 32) while vLLM's continuous batching keeps the
wait at 0.78 s. A 256-token benchmark hides this completely — the first token is
what a user actually feels.

## Power and energy

![joules per 1M output tokens by engine and concurrency](docs/energy.svg)

Board power from `nvidia-smi`, averaged over each sweep (desktop draw included, so
read the deltas between engines rather than the absolute watts).

| engine | concurrency | avg W | J/1M output tokens |
|---|---|---|---|
| ollama | 1 | 138.1 | 1,463,691 |
| llamacpp | 1 | 137.8 | 1,259,523 |
| vllm | 1 | 131.2 | 1,151,071 |
| ollama | 8 | 131.3 | 702,105 |
| llamacpp | 8 | 131.7 | 632,582 |
| vllm | 8 | 128.5 | 166,377 |
| ollama | 32 | 94.3 | 129,669 |
| llamacpp | 32 | 108.7 | 111,987 |
| vllm | 32 | 142.4 | 86,412 |

Energy per token falls ~17× from single stream to 32 concurrent on vLLM and ~13× on
Ollama. Note the direction at the top end: vLLM draws the *most* watts at 32 slots
(142 W) and still uses the least energy per token, because it finishes 2.3× more
tokens in the same second. Watts are a power bill; joules per token are the bill for
the work.


## FP8 vs AWQ

![FP8 against AWQ at the same load](docs/fp8.svg)

| engine | concurrency | output tok/s | p50 ms | p95 ms | peak VRAM MiB | $/1M output tokens |
|---|---|---|---|---|---|---|
| vllm (AWQ) | 1 | 113.98 | 2221.1 | 2236.8 | 16088 | 0.85 |
| vllm-fp8 | 1 | 73.85 | 3440.9 | 3456.0 | 16739 | 1.32 |
| vllm (AWQ) | 8 | 772.34 | 2515.6 | 2604.5 | 16258 | 0.13 |
| vllm-fp8 | 8 | 534.26 | 3710.1 | 3781.9 | 16882 | 0.18 |
| vllm (AWQ) | 32 | 1647.92 | 4192.2 | 4660.0 | 16432 | 0.06 |
| vllm-fp8 | 32 | 1451.85 | 4712.6 | 4991.0 | 17118 | 0.07 |

FP8 loses at every point: −35% throughput at one stream, −31% at 8, −12% at 32, and it
costs $1.32 per million tokens against $0.85. Two reasons, both visible in the table.
FP8 weights are 9.5 GB against the 4-bit AWQ's 5.5 GB, so each token reads *more*
bytes, and on this card the fast FP8 path is DeepGEMM — which needs a CUDA toolkit
this machine does not have. What ran is vLLM's Cutlass block-scaled fallback. FP8 is
a data-centre win (Hopper/Blackwell with the full kernel set, and against an
unquantised BF16 baseline); against AWQ on a 24 GB laptop without the toolkit it is
not.

## Speculative decoding

![Speculative decoding against the same AWQ model](docs/spec.svg)

| engine | concurrency | output tok/s | p50 ms | p95 ms | peak VRAM MiB | $/1M output tokens |
|---|---|---|---|---|---|---|
| vllm | 1 | 113.98 | 2221.1 | 2236.8 | 16088 | 0.85 |
| vllm-spec | 1 | 157.31 | 1488.9 | 2369.4 | 16086 | 0.62 |
| vllm | 8 | 772.34 | 2515.6 | 2604.5 | 16258 | 0.13 |
| vllm-spec | 8 | 980.11 | 1861.4 | 2896.0 | 16082 | 0.10 |
| vllm | 32 | 1647.92 | 4192.2 | 4660.0 | 16432 | 0.06 |
| vllm-spec | 32 | 1674.47 | 4141.0 | 6487.9 | 16348 | 0.06 |

A 0.6B draft model proposes 5 tokens per step and the 8B target verifies them in one
forward pass. That is worth **+38% at one stream** and **+27% at 8**, then nothing:
**+1.6% at 32**. The reason is in vLLM's own metrics — the draft accepted 76.6% of
its proposals at one stream, mean acceptance length 4.8 of 5, which only pays while
the GPU has spare compute to run the draft. Once 32 sequences saturate the batch,
the verifier is the bottleneck and the draft is nearly free work that no longer buys
tokens. It also widens the tail: p95 at 32 slots is 6.49 s against 4.66 s, because
acceptance varies per prompt and a rejected draft wastes the step. VRAM is unchanged
(the 1.5 GB draft fits inside the same reservation), so the trade is latency tail for
throughput, and only at low concurrency.


## What this measures

One 8B model served by three stacks on one RTX 5090 Laptop (24 GB), inside WSL2
Ubuntu 24.04 so the kernel, driver and VRAM accounting are identical for all three:

| engine | version | how it serves |
|---|---|---|
| Ollama | 0.40.8 | `ollama serve`, model built from the GGUF below |
| llama.cpp | b11819, CUDA 13.4 prebuilt | `llama-server -c 32768 --parallel 32 -ngl 99` |
| vLLM | 0.32.0 (torch 2.13.0+cu130) | `vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096` |

Two more vLLM configurations reuse the same server: `vllm-fp8` serves the FP8
weights instead of AWQ, and `vllm-spec` adds a 0.6B draft model
(`--speculative-config`, 5 speculative tokens). They are opt-in, not part of the
baseline sweep.

Every request is a **raw completion** — no chat template, `temperature=0`,
`max_tokens=256`, seed 0 — against the same 200-passage prompt set
(`data/prompts.jsonl`, sha256 pinned in `bench/prompts.py`). Prompt text comes from
the FAA *AC 43.13-1B* (US government work, public domain): 200 passages of
150–400 characters, sampled deterministically.

Each baseline run is 200 requests at concurrency 1, 8 and 32; every engine also gets
a one-token sweep at the same concurrencies for time-to-first-token. The GPU must be
idle (≥14 GB free, i.e. no model of ours resident) before an engine starts, and
engines are stopped between runs so no two stacks are ever resident together. VRAM
and board power are sampled every 0.5 s during a sweep; VRAM is the absolute
`nvidia-smi` peak, so the desktop's own ~7 GB are included in every number.

## Models

| model | size | licence | source |
|---|---|---|---|
| `Qwen3-8B-Q4_K_M.gguf` (Ollama, llama.cpp) | 4.7 GB | Apache-2.0 | [Qwen/Qwen3-8B-GGUF](https://huggingface.co/Qwen/Qwen3-8B-GGUF) |
| `Qwen/Qwen3-8B-AWQ` (vLLM, spec target) | 5.5 GB | Apache-2.0 | [Qwen/Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ) |
| `Qwen/Qwen3-8B-FP8` (vLLM, FP8 run) | 9.5 GB | Apache-2.0 | [Qwen/Qwen3-8B-FP8](https://huggingface.co/Qwen/Qwen3-8B-FP8) |
| `Qwen/Qwen3-0.6B` (draft for speculative decoding) | 1.5 GB | Apache-2.0 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) |

The quantisation differs on purpose: vLLM's fast path on sm_120 is AWQ/Marlin (or
FP8), while the GGUF engines share the same 4-bit bytes. So the vLLM row is *not* a
pure serving-stack comparison — it is "vLLM at its best quantisation" against
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
repeat with one-token requests for TTFT, stop it, next engine, then render the charts
and print the Markdown tables. The two extra vLLM configurations are opt-in:

```bash
ENGINES=ollama,llamacpp,vllm,vllm-fp8,vllm-spec make bench
```

`make test` runs the 21 unit tests (summary and energy math, the three HTTP adapters
against mocked servers, the VRAM gate, the sweep runner, prompt-set integrity, the
charts).

## Layout

```
bench/       config, engine adapters, sweep runner, summary math, VRAM sampler, chart
data/        200 fixed prompts (sha256-pinned)
scripts/     install_engines.sh, fetch_models.sh, run_all.sh, Modelfile
results/     one json + 200-record jsonl per (engine, concurrency)
docs/        chart.svg, ttft.svg, energy.svg, fp8.svg, spec.svg, design and plan
tests/       21 pytest tests
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
- The FP8 run also sets `VLLM_USE_DEEP_GEMM=0`. vLLM's first choice for block-scaled
  FP8 is DeepGEMM, whose bundled JIT also wants a CUDA toolkit; the Cutlass FP8 kernel
  it falls back to needs none. FP8 numbers here are the Cutlass kernel.
