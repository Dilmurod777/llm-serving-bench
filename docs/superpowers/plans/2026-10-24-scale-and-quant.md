# Scale and quantisation experiments — implementation plan

**Goal:** close the three known gaps in `llm-serving-bench`: engine-vs-engine on identical
weights, behaviour beyond 32 concurrent requests, and time-to-first-token on realistic
4k-token prompts.

**Spec:** the existing design doc `docs/superpowers/specs/2026-10-23-llm-serving-bench-design.md`
still governs; these are additions in its established shape (engine block in
`scripts/run_all.sh`, three sweeps, one chart, one README section).

**Order:** 1 → 3 → 2. GGUF is isolated (one new engine), concurrency changes shared
config, long prompts need a new prompt set *and* a context change, so it goes last.

---

## Task 1: `vllm-gguf` — the same bytes in all three engines

The current vLLM row is AWQ; the GGUF engines share Q4_K_M. This task removes the
confound: vLLM serving the exact GGUF file llama.cpp loads.

**Files**
- `bench/__main__.py`: `DEFAULTS["vllm-gguf"] = ("http://127.0.0.1:8000", "models/Qwen3-8B-Q4_K_M.gguf")`
- `scripts/run_all.sh`: block `vllm-gguf` using `serve_vllm vllm-gguf models/Qwen3-8B-Q4_K_M.gguf --quantization gguf --tokenizer Qwen/Qwen3-8B`
- `bench/plot.py`: `COLORS["vllm-gguf"] = "#6b4f8a"`; `main()` renders `docs/gguf.svg` as vllm-gguf vs llamacpp vs ollama
- `README.md`: `## Same weights, three engines` section with chart, table, prose

**Smoke test first (decides the task):** serve on port 8003, wait 600 s, post one
16-token completion. vLLM 0.32's GGUF path is unoptimised and may refuse sm_120.

**Acceptance:** 3 sweeps × 200 requests, 0 unexpected errors, a chart where all three
bars share one quantisation; README says which kernel vLLM picked (log line
`Using ... kernel`) and what it cost in speed.

**If it fails:** record the failure verbatim in the README as a finding ("vLLM 0.32
cannot serve GGUF on sm_120: <error>"), keep the task closed, move on. No engine swap.

---

## Task 2: concurrency 64 and 128

**The constraint:** KV cache. Qwen3-8B is ~0.15 MB/token/sequence-slot; 1,024 tokens
× 128 slots = 131k tokens ≈ 20 GB, which will not fit beside 5 GB of weights and a
7 GB desktop. 512 tokens/slot × 128 = 65k ≈ 9.8 GB, which does.

**Decision:** at 64 and 128, every engine runs **512 tokens per slot** (a request
needs ~120 prompt + 256 output = 376, so 512 is enough). At 1/8/32 the existing
1,024 stays, so those numbers are untouched. README states the halved slot context
for the two new points.

**Files**
- `bench/config.py`: `CONCURRENCIES = (1, 8, 32, 64, 128)`
- `scripts/run_all.sh`: `sweep()` iterates the tuple; llama.cpp `-c 65536 --parallel 128`; Ollama drop-in `OLLAMA_NUM_PARALLEL=128` + `OLLAMA_CONTEXT_LENGTH=512`; vLLM unchanged (`--max-num-seqs` default 256, `--gpu-memory-utilization 0.55`)
- `README.md`: two extra rows per engine in every affected table; one caveat sentence

**Run:** `ENGINES=ollama,llamacpp,vllm bash scripts/run_all.sh` (all five points, plus TTFT).

**Acceptance:** all five concurrencies produce results for all three engines; if vLLM
preempts or queues at 128, that is recorded as a finding, not hidden. If an engine
OOMs at 128, drop that point for that engine and say so.

**Cost:** ~30 min GPU. Watch the idle gate — 15 GB free is needed before the run.

---

## Task 3: long prompts (≈4k tokens) — TTFT only

**Why TTFT only:** decode speed does not depend on prompt length; prefill does. A
4k prompt at c1 costs ~1–2 s of prefill, which is what a user feels.

**Files**
- `scripts/make_long_prompts.py`: concatenate FAA passages into 100 prompts of
  3,800–4,200 tokens (same corpus, seed 0), write `data/prompts_long.jsonl`
- `bench/prompts.py`: `PROMPTS_LONG_PATH`, `PROMPTS_LONG_SHA256`, `load_long_prompts()`
  (same guard shape as the short set)
- `scripts/run_all.sh`: `LONG=only` mode; context 8192/slot for all engines
  (Ollama `OLLAMA_CONTEXT_LENGTH=8192` + `OLLAMA_NUM_PARALLEL=8`, llama.cpp
  `-c 65536 --parallel 8`, vLLM `--max-model-len 8192`), concurrencies 1/4/8,
  `--max-tokens 1 --prompts data/prompts_long.jsonl --out results/long/<engine>-c<N>.json`
- `bench/plot.py`: `docs/ttft-long.svg` from `results/long/`
- `README.md`: `## Long prompts` section with chart, table, prose

**Acceptance:** 100 requests per point, TTFT p50 rises by an order of magnitude over
the short-prompt table, no truncation warnings in any engine log (a truncated prompt
would silently invalidate the number — check `prompt_eval_count` ≈ 4k in the JSONL).

**Cost:** ~30 min GPU including three engine restarts at the larger context.

---

## Risks

- **vLLM GGUF may not run at all** — accepted, reported as a finding.
- **128 concurrency may OOM llama.cpp/Ollama** even at 512/slot; fallback is 64 only,
  documented.
- **Long prompts need 8k context** in engines currently configured for 1k; a missed
  restart silently truncates prompts. The `prompt_eval_count` check above is the guard.
- Every new number goes through the same gate: 3 commands, a chart, a number, one
  caveat line. No new dependencies, no new abstractions.
