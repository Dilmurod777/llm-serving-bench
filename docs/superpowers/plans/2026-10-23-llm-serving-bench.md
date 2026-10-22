# llm-serving-bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One command measures Ollama, llama.cpp and vLLM on the same 5090 and renders a chart plus a cost table that becomes the README.

**Architecture:** A Python driver talks HTTP to whichever engine is running, so engines are swapped by URL, not by import. Every engine answers the same 200 raw-completion prompts at concurrency 1/8/32; each request's token count and latency are written as JSONL, summarised into one record per (engine, concurrency), and plotted to SVG.

**Tech Stack:** Python 3.12 (mise), httpx, pytest, asyncio; Ollama 0.40.8, llama.cpp b11819, vLLM 0.32.0; WSL2 Ubuntu-24.04.

**Spec:** `docs/superpowers/specs/2026-10-23-llm-serving-bench-design.md`

## Global Constraints

- Raw completions only: no chat template, `temperature=0`, `max_tokens=256`, seed 0.
- Concurrencies exactly `(1, 8, 32)`; prompt set exactly 200 lines in `data/prompts.jsonl`.
- Every run refuses to start unless free VRAM ≥ 22 000 MiB.
- `GPU_PRICE_PER_HOUR = 0.35` in `bench/config.py`, printed next to the cost column.
- No model weights in git: `models/` is ignored, `scripts/fetch_models.sh` is the only way to get them.
- Ollama and llama.cpp must load the same `Qwen3-8B-Q4_K_M.gguf`; vLLM loads `Qwen/Qwen3-8B-AWQ`. The README states this.
- All commands in this plan run inside WSL2: `wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/d/Projects/dev/llm-serving-bench && <cmd>'`.

## Review Focus

- **Engine returns HTTP 500 or dies mid-sweep** — the run must fail loudly with the engine's message, never record a partial row as if it completed.
- **Concurrency 32 on a 24 GB card** — vLLM may OOM during warmup; the error must name the engine and the setting, not a traceback alone.
- **Prompt longer than the served context** — the engine truncates or errors; either way the record must carry `error`, never a silently short token count.
- **Engine not yet listening when the sweep starts** — wait/retry briefly, then fail; never record 200 connection errors as latency numbers.
- **A stalled request** — per-request timeout (120 s) turns into an `error` record so the sweep cannot hang forever.

---

### Task 1: Package skeleton and the summary math

**Files:**
- Create: `bench/__init__.py`, `bench/config.py`, `bench/summarize.py`, `tests/test_summarize.py`, `.gitignore`, `pyproject.toml`

**Interfaces:**
- Produces: `config.GPU_PRICE_PER_HOUR`, `config.MAX_TOKENS`, `config.CONCURRENCIES`, `config.MODELS_DIR`
- Produces: `summarize.summarize(records: list[dict], concurrency: int, duration_s: float, peak_vram_mb: int) -> dict` with keys `engine_records`, `requests`, `output_tokens`, `tokens_per_s`, `p50_ms`, `p95_ms`, `peak_vram_mb`, `cost_per_1m_tokens`.

- [ ] **Step 1: Write the failing test**

```python
from bench.summarize import summarize

def test_summarize_known_numbers():
    records = [{"latency_ms": ms, "completion_tokens": 100, "error": None} for ms in (1000, 2000, 3000, 4000)]
    s = summarize(records, concurrency=1, duration_s=10.0, peak_vram_mb=20000)
    assert s["requests"] == 4 and s["output_tokens"] == 400
    assert s["tokens_per_s"] == 40.0
    assert s["p50_ms"] == 2500 and s["p95_ms"] == 4000
    assert round(s["cost_per_1m_tokens"], 2) == 2.43   # 0.35 / (40*3600) * 1e6

def test_errors_are_excluded_from_latency_but_counted():
    records = [{"latency_ms": 1000, "completion_tokens": 100, "error": None},
               {"latency_ms": 0, "completion_tokens": 0, "error": "timeout"}]
    s = summarize(records, 1, 1.0, 1)
    assert s["requests"] == 2 and s["errors"] == 1
    assert s["p95_ms"] == 1000
```

- [ ] **Step 2: Run it and watch it fail** — `python -m pytest tests/test_summarize.py -v` → `ModuleNotFoundError: bench.summarize`.

- [ ] **Step 3: Implement** `config.py` with the four constants (above) and `summarize.py`: nearest-rank percentiles on successful latencies (`sorted(...)[ceil(q*n)-1]`), `tokens_per_s = output_tokens / duration_s`, `cost_per_1m_tokens = GPU_PRICE_PER_HOUR / (tokens_per_s * 3600) * 1e6`, `errors` = count of records with a truthy `error`.

- [ ] **Step 4: Run tests** — expect PASS. Add `.gitignore` (`models/`, `.venv/`, `__pycache__/`) and a `pyproject.toml` declaring `httpx` and `pytest` and `[tool.pytest.ini_options] pythonpath = ["."]`.

- [ ] **Step 5: Commit** — `git init`, `git add -A`, `git commit -m "feat: package skeleton and summary math"`.

---

### Task 2: The fixed prompt set

**Files:**
- Create: `scripts/make_prompts.py`, `data/prompts.jsonl`, `tests/test_prompts.py`

**Interfaces:**
- Produces: `bench.prompts.load_prompts(path: str = "data/prompts.jsonl") -> list[str]` raising `ValueError` unless there are exactly 200 non-empty strings and the file's sha256 equals the constant `PROMPTS_SHA256` in that module.

- [ ] **Step 1: Write the failing test** — assert `load_prompts()` returns 200 strings, each 150–400 characters, and that a mutated copy raises `ValueError`.
- [ ] **Step 2: Run it, watch it fail.**
- [ ] **Step 3: Implement** `scripts/make_prompts.py` as a one-off generator: read a public-domain text passed on the command line (FAA AC 43.13-1B PDF converted with `pdftotext`), split into paragraphs, keep those of 800–1400 characters, take every *k*-th until 200 with `random.Random(0)`, write `{"id": i, "text": ...}` per line. Run it once; commit `data/prompts.jsonl` and the resulting sha in `bench/prompts.py`.
- [ ] **Step 4: Tests pass** — `python -m pytest tests/test_prompts.py -v`.
- [ ] **Step 5: Commit.**

---

### Task 3: Three HTTP engine adapters

**Files:**
- Create: `bench/engines.py`, `tests/test_engines.py`

**Interfaces:**
- Produces: `Completion(completion_tokens: int, latency_ms: float, text: str)`; `OllamaEngine(base_url, model, max_tokens=MAX_TOKENS)`, `LlamaCppEngine(base_url, max_tokens=MAX_TOKENS)`, `VllmEngine(base_url, model, max_tokens=MAX_TOKENS)`, each with `name: str` and `async def complete(prompt: str) -> Completion`.
- Each adapter takes an injected `httpx.AsyncClient` (`client=` keyword) so tests use `httpx.MockTransport`.

- [ ] **Step 1: Write the failing tests** — one per engine, asserting the request body and the parsed token count:

```python
async def test_ollama_raw_completion():
    seen = {}
    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"response": "ok", "eval_count": 42})
    engine = OllamaEngine("http://x", "qwen3-8b-q4", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = await engine.complete("hello")
    assert out.completion_tokens == 42
    assert seen["raw"] is True and seen["model"] == "qwen3-8b-q4" and seen["stream"] is False
    assert seen["options"]["temperature"] == 0 and seen["options"]["num_predict"] == 256
```

Equivalent tests: llama.cpp POSTs `/completion` with `{"prompt": ..., "n_predict": 256, "temperature": 0, "stream": false}` and reads `timings.predicted_n`; vLLM POSTs `/v1/completions` with `{"model": ..., "prompt": ..., "max_tokens": 256, "temperature": 0}` and reads `usage.completion_tokens`. Also assert a 500 raises `EngineError` carrying the body.

- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement** the three classes with `time.perf_counter()` around the awaited POST; `EngineError(RuntimeError)` with `f"{name}: HTTP {status}: {body[:200]}"`.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 4: Idle-VRAM gate and peak sampler

**Files:**
- Create: `bench/vram.py`, `tests/test_vram.py`

**Interfaces:**
- Produces: `parse_free_mb(nvidia_smi_output: str) -> int`; `assert_idle(min_free_mb: int = 22000) -> None` raising `RuntimeError("GPU busy: N MiB free, need 22000")`; `VramSampler` context manager with `.peak_mb` (polls `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` every 500 ms on a daemon thread).

- [ ] **Step 1: Failing test** — `parse_free_mb("12000\n") == 12000`, and `assert_idle(22000)` raises when the injected reader returns 11338 (pass the reader as an argument so no GPU is needed).
- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 5: The sweep runner

**Files:**
- Create: `bench/runner.py`, `tests/test_runner.py`

**Interfaces:**
- Produces: `async def run_sweep(engine, prompts: list[str], concurrency: int, timeout_s: float = 120.0) -> tuple[list[dict], float]` returning `(records, duration_s)`; each record is `{"prompt_id": int, "latency_ms": float, "completion_tokens": int, "text_chars": int, "error": str | None}`.

- [ ] **Step 1: Failing test** with a fake engine that sleeps 50 ms:

```python
async def test_concurrency_is_respected_and_errors_are_records():
    fake = FakeEngine(delay=0.05, fail_on={3})
    records, duration = await run_sweep(fake, [f"p{i}" for i in range(10)], concurrency=5)
    assert len(records) == 10
    assert fake.max_in_flight <= 5
    assert [r["prompt_id"] for r in records if r["error"]] == [3]
    assert duration < 0.3          # 10 prompts / 5 lanes at 50 ms, not 0.5 s
```

- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement** with `asyncio.Semaphore(concurrency)` and `asyncio.gather`; `except (EngineError, httpx.HTTPError, asyncio.TimeoutError)` → record with `error=type(exc).__name__ + ": " + str(exc)`; duration measured from first dispatch to last completion.
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 6: CLI that writes results

**Files:**
- Create: `bench/__main__.py`

**Interfaces:**
- Consumes: engines, runner, summarize, vram.
- Produces: `python -m bench --engine {ollama,llamacpp,vllm} --concurrency N [--base-url URL] [--out results/<engine>-c<N>.json]`; writes `{"engine":..., "concurrency":..., **summary}` to `--out` and appends every record to `results/<engine>-c<N>.jsonl`.

- [ ] **Step 1: Failing test** — `test_cli_writes_results` calls `main(["--engine","ollama","--concurrency","2","--out",tmp])` with a monkeypatched `build_engine`, asserts the JSON has `tokens_per_s`, `p95_ms`, `errors`.
- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement** (`argparse`, `asyncio.run`, `assert_idle()` first, `VramSampler` around the sweep).
- [ ] **Step 4: Tests pass.**
- [ ] **Step 5: Commit.**

---

### Task 7: Engine install and model fetch, proven in WSL

**Files:**
- Create: `scripts/install_engines.sh`, `scripts/fetch_models.sh`, `scripts/Modelfile.ollama`, `Makefile`

**Interfaces:**
- Produces: `make install` (Ollama via install.sh + `zstd`; llama.cpp b11819 ubuntu-cuda-13.4 `llama-*` + `cudart-*` tarballs into `$HOME/engines/llama.cpp`), `make models` (two pinned downloads with `sha256sum -c`), `make ollama-model` (`ollama create qwen3-8b-q4 -f scripts/Modelfile.ollama`).

- [ ] **Step 1: Write `fetch_models.sh`** with pinned URLs (`huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf`, `huggingface.co/Qwen/Qwen3-8B-AWQ`) and expected sha256 values computed once by hand; `set -euo pipefail`, skip if checksum already matches, write into `models/`.
- [ ] **Step 2: Write `install_engines.sh`** copying the two commands already proven in this session; `Modelfile.ollama` = `FROM <abs path>/Qwen3-8B-Q4_K_M.gguf` + `PARAMETER temperature 0` + `PARAMETER num_predict 256`.
- [ ] **Step 3: Run `make models`, `make ollama-model`** — expect `success` from `ollama create` and `ollama list` showing `qwen3-8b-q4` at ~5 GB.
- [ ] **Step 4: Smoke each server with curl** — Ollama `POST /api/generate` returns `eval_count > 0`; `llama-server --model models/Qwen3-8B-Q4_K_M.gguf -c 8192 --n-gpu-layers 99` returns `timings.predicted_n > 0`; `vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 4096` answers `/v1/completions`. Record the three commands in the Makefile (`serve-ollama`, `serve-llamacpp`, `serve-vllm`).
- [ ] **Step 5: Commit** (scripts + Makefile only; `models/` stays ignored).

---

### Task 8: Chart and README table generator

**Files:**
- Create: `bench/plot.py`, `tests/test_plot.py`, `docs/chart.svg`

**Interfaces:**
- Produces: `load_results(paths: list[str]) -> list[dict]`; `to_svg(rows, path) -> None` (grouped bars: x = concurrency, series = engine, y = tokens/s, value labels); `to_markdown(rows) -> str` (engine, concurrency, tok/s, p50, p95, peak VRAM, $/1M).

- [ ] **Step 1: Failing test** — feed three fixture rows and assert the markdown contains `| vllm | 32 |` and the SVG contains three `<rect>` bars with heights proportional to `tokens_per_s`.
- [ ] **Step 2: Run, watch fail.**
- [ ] **Step 3: Implement** hand-written SVG (no plotting dependency): 640×360 viewBox, 3 groups × 3 bars, y-axis ticks at 0/max/2.
- [ ] **Step 4: Tests pass**, `python -m bench.plot results/*.json` writes `docs/chart.svg`.
- [ ] **Step 5: Commit.**

---

### Task 9: README, real run, deploy

**Files:**
- Create: `README.md`; Modify: `Makefile` (`bench` loops the 9 combinations, `chart` runs plot).

- [ ] **Step 1: `make bench` on an idle GPU** — 9 runs (3 engines × 1/8/32), each preceded by stopping the other engines. Expect c32 vLLM to be the fastest row; if any run fails, record the failure in the README table rather than dropping the row.
- [ ] **Step 2: `make chart` and paste the output table into the README** — chart above the fold, then the table, then:
  ```
  git clone … && cd llm-serving-bench
  make install models
  make bench chart
  ```
  plus the model table (size, licence, link), the cost formula with `0.35 $/h`, and one sentence on the GGUF/AWQ asymmetry.
- [ ] **Step 3: Verify a stranger's path** — the three commands work from a clean WSL checkout; every number in the README appears in `results/`.
- [ ] **Step 4: Commit, push, add the repo link to `portfolio/index.html`** (What I work on → project entry).
