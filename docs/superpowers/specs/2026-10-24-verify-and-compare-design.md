# Reproduce and compare — design

Date: 2026-10-24
Project: `llm-serving-bench`

## Why

The repo currently reports what happened on one laptop. A reader who wants to
know whether *their* number is normal has no way to find out, and a reader who
wants to bench their own model has to edit Python. The stated purpose of the
project is to teach measurement: hold everything constant, isolate one variable,
report the tail, and say what a number cannot support. That purpose is only met
if a reader can produce their own row and compare it with ours.

Success: a stranger with one GPU can (a) run one sweep, (b) see their number
next to ours with a delta, and (c) diagnose a wrong number from a symptom table
instead of guessing.

## Goals

1. `python -m bench.compare <reference> <yours>` — compare two result sets and
   print a delta table. Read-only, no GPU, no network.
2. `--model` on the sweep CLI, so vLLM and Ollama can be pointed at another
   model without editing `DEFAULTS`; llama.cpp stays file-driven.
3. A README teaching section: prerequisites, wall times, "what differs
   legitimately", and a pitfalls table (symptom → cause → fix) built from traps
   this project actually hit.
4. `make verify` — the verify profile of `run_all.sh`: one baseline sweep into
   `results/local/`, then `compare` against the committed row.

## Non-goals

- No web UI, no database of historical runs, no hosted dashboard.
- No model downloader beyond the existing `make models`.
- No GPU regression testing in CI; CI stays the mocked pytest suite.
- No automatic recommendations ("use vLLM if…") — the repo shows numbers, the
  reader decides.

## Anti-goals

- A comparison that hides a real regression behind a wide tolerance.
- A "bring your own model" path that silently changes the prompt set,
  `max_tokens` or the engine shape and then compares incomparable numbers.
- More charts. The nine existing ones are enough; this work adds no chart.

## Constraints

- Python 3.12, dependencies stay `httpx` (plus pytest for tests).
- `bench.compare` must work offline on the files in `results/`.
- Existing behaviour of `python -m bench` must not change when `--model` is
  absent.

## Design

### `meta` becomes the comparability record

`bench/__main__.py` already writes `engine_version`, `gpu`, `date_utc`,
`prompt_sha256`, `max_tokens`. It gains three fields, because without them two
runs can look comparable when they are not:

- `model` — the model id used for the sweep (`DEFAULTS[name][1]` or `--model`).
- `slots` — parallel slots the engine was configured with.
- `ctx` — context length per slot.

`run()` takes `model`, `slots`, `ctx` (defaults `None`, `32`, `1024`), so
`python -m bench` keeps working unchanged. `scripts/run_all.sh` passes the
profile's `SLOTS`/`CTX` and `MODEL` into the CLI; a `MODEL=` run writes into
`results/local/` instead of overwriting the reference rows.

The 12 committed `vllm-fp8`/`vllm-spec` files predate `meta`. They stay as they
are — no fabricated provenance — and `compare` reports them as
`incomparable: no meta`.

### `bench/compare.py`

```
python -m bench.compare A B [--tolerance 0.25] [--strict]
```

- `A` is the **reference** (committed results), `B` is the reader's run. Delta is
  `(B − A) / A`, so a positive throughput delta means faster than the reference.
- Each argument is either a directory or a single JSON file. Directories are
  scanned **one level** (not recursive), so `results/`, `results/short/`,
  `results/wide/`, `results/long/` are compared one at a time. A directory
  nested inside the other (`results/local` inside `results`) is skipped.
- Pairs are keyed by file stem. A stem present on one side only is listed as
  `missing` and, under `--strict`, fails.
- Guard: a pair is `incomparable` when any of `prompt_sha256`, `max_tokens`,
  `model`, `slots`, `ctx` differ, or when either side has no `meta`. The reason
  is printed.
- Gated table: `file | metric | reference | yours | delta % | band` for
  `tokens_per_s`, `per_stream_tps`, `p50_ms`, `p95_ms`. Throughput sign: faster
  is positive. Latency sign: slower is positive. Band `ok` when
  `abs(delta) <= tolerance`, else `out`.
- Second, ungated table: `peak_vram_mb`, `avg_power_w`, `joules_per_1m_tokens`,
  `cost_per_1m_tokens` — machine-specific, a band would be noise.
- Exit 0 by default; `--strict` exits 1 on any `out`, `incomparable` or
  `missing` row.

### `--model`

- `build_engine(name, base_url=None, max_tokens=MAX_TOKENS, model=None)`;
  `model or DEFAULTS[name][1]`.
- `--model` is rejected for `llamacpp` with a clear message (the server loads the
  file), not silently ignored.
- Ollama still needs `ollama create` from a GGUF before a new model id exists;
  documented, not automated.

### `make verify`

`make verify [ENGINE=ollama]` runs `PROFILE=verify bash scripts/run_all.sh`:

- profile: `CONCS="1"`, `SLOTS=32`, `CTX=1024`, output `results/local/`,
  `bench --skip-idle-check` only for the sweeps after the first — the idle gate
  still runs **before** the server starts, exactly as in the other profiles;
- after the sweep, `python -m bench.compare results/<engine>-c1.json
  results/local/<engine>-c1.json`.

Cost on this machine: ≈ 8 minutes (480 s of requests plus model load), because
only the 256-token baseline row is comparable with the main table; the 32-token
short row is a different contract and would need `results/short` as reference.

### README

New section `## Reproduce on your machine`, placed after `## Reproduce`:

1. Prerequisites: one CUDA GPU with ≥ 16 GB free, WSL2 Ubuntu 24.04, engines
   from `make install`, weights from `make models`.
2. Wall times observed here, labelled approximate: baseline ≈ 40 min,
   `SHORT=only` ≈ 10 min, `PROFILE=wide` ≈ 20 min, `PROFILE=long` ≈ 15 min,
   `make verify` ≈ 8 min.
3. Three commands to a comparable row: `make verify`, then read the table.
4. `### What differs legitimately` — GPU class, driver, desktop VRAM, clocks.
5. `### When your number is wrong` — symptom → cause → fix:
   - 10× slow, no CUDA lines in the server log → llama.cpp dropped the CUDA
     backend (runtime libs not beside the binary) → put `libcudart`/`libcublas`
     next to `llama-server`.
   - `attempted to allocate 135168.00 MB` → Ollama multiplies context by parallel
     slots → `OLLAMA_CONTEXT_LENGTH` is per slot.
   - GPU busy before the sweep → another process holds VRAM (an embedding model
     is the usual one) → stop it; the gate wants 14 GB free.
   - `Could not find nvcc` / DeepGEMM assertion → vLLM wants a CUDA toolkit for
     a JIT kernel → `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`.
   - 30–50% off with everything healthy → laptop throttling; rerun cool and
     compare `avg_power_w`.

## Tests

- `compare`: delta signs per metric, band verdict, incomparable on differing
  `model`/`slots`/`ctx`/`prompt_sha256`/`max_tokens`, missing-meta case, missing
  row, nested-directory skip, single-file mode.
- `--model`: `build_engine("ollama", model="x").model == "x"`; default unchanged;
  `--model` with `llamacpp` exits non-zero.
- `meta`: `test_cli` asserts `model`, `slots`, `ctx` are written.
- No GPU, no network: plain JSON written to `tmp_path`.

## Calls I made (check these)

- Tolerance default 25%, advisory unless `--strict`.
- Gated metrics: throughput (total and per-user) and p50/p95 latency. VRAM,
  power, energy, cost are shown, never gated.
- `--model` does not extend to llama.cpp and does not rebuild the Ollama model.
- Verify is the faithful 256-token row (~8 min), not the 1.5-minute short row.
- The 12 meta-less committed files stay unbackfilled and are reported as
  incomparable rather than silently compared.
