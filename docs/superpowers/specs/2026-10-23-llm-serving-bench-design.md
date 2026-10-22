# llm-serving-bench — design

Date: 2026-10-23
Status: approved in conversation (three-engine bench, all engines in WSL2)

## Intent

A public, reproducible answer to one question a hiring manager asks about an
inference engineer: *what does one token cost, and why?* Three serving stacks,
one model, one GPU, one prompt set, one chart.

Portfolio slot: P2 of `portfolio-plan.md` (Dec 2026). Success = the chart is the
README and a stranger reproduces it with `make bench`.

## Goals

- Measure **Ollama**, **llama.cpp** (`llama-server`), **vLLM** on the same GPU.
- Report, per engine and per concurrency (1 / 8 / 32):
  output tokens/s, p50 and p95 end-to-end latency, peak VRAM, \$ per 1M output tokens.
- One fixed 200-prompt set, temperature 0, fixed `max_tokens`, **raw completions**
  (no chat template) so all three engines receive identical prompt tokens.
- Re-runnable: `make bench` writes `results/`, `make chart` rewrites the README chart.
- Model weights are never in git: `scripts/fetch_models.sh` downloads pinned URLs
  with checksums into `models/` (git-ignored), and the README lists size, licence, link.

## Non-goals

- No quality evaluation (P3 owns retrieval/answer quality).
- No multi-GPU, no tensor parallel, no quantisation sweep (FP8/AWQ is a later run).
- No Windows-native engines in this iteration. Windows Ollama is untouched and unused.
- No web UI, no database, no auth.

## Constraints (measured on this machine, 2026-10-23)

- GPU: RTX 5090 Laptop 24 GB, driver 617.42, CUDA UMD 13.4, visible inside WSL2
  via `/dev/dxg` with no NVIDIA driver installed in Linux.
- All engines run in **Ubuntu-24.04 (WSL2)**, user `dev`, so kernel, driver stack
  and VRAM accounting are identical. WSL localhost forwarding is Windows→WSL only,
  so the Windows Ollama cannot be reused; that is why the Linux build is installed.
- Runtimes: Ollama 0.40.8 (systemd service, 127.0.0.1:11434), llama.cpp b11819
  ubuntu-cuda-13.4 prebuilt (no CUDA toolkit, no compile), vLLM 0.32.0 (torch 2.13.0).
- Python: mise-managed 3.12.15 (`mise exec -- python`), venv at `~/engines/vllm-venv`.
- Every measurement starts from an **idle GPU**: a loaded 27B model already
  occupied 11.3 GB during recon. Engines are stopped between runs, and the runner
  refuses to start if free VRAM < 22 GB.
- Sequential runs only. Two engines on one GPU measure contention, not engines.

## Decisions

- **Qwen3-8B**, not 14B: headroom at concurrency 32, and the story is engine
  overhead, not model size.
- **Same GGUF bytes** for Ollama and llama.cpp (`Qwen3-8B-Q4_K_M.gguf`, 5.0 GB),
  so the only variable between those two rows is the serving stack. vLLM gets
  `Qwen/Qwen3-8B-AWQ` (5.5 GB) because vLLM's fast path is AWQ/Marlin on sm_120.
  The README states this asymmetry rather than hiding it.
- **Raw completions**, temperature 0, `max_tokens=256`, seed 0.
- Token counts come from each server's own accounting (`eval_count`,
  `timings.predicted_n`, `usage.completion_tokens`); the README says so.
- **Cost** = measured GPU-hour price ÷ measured tokens/s, with the price as an
  explicit constant in `bench/config.py` (default \$0.35/h, a rented 5090), and the
  formula printed next to the table.
- Prompt set: 200 fixed prompts (~200 tokens each) sampled deterministically from
  a public-domain text, committed as `data/prompts.jsonl` (no large files, no NDA).

## Shape

```
llm-serving-bench/
  bench/           config.py, engines.py, runner.py, summarize.py, plot.py
  data/prompts.jsonl
  scripts/         install_engines.sh, fetch_models.sh
  results/         <engine>-c<N>.jsonl  (committed; small)
  docs/            chart.svg + specs/ + plans/
  tests/           fixtures + pytest
  Makefile         bench, chart, test, install, models
  README.md        chart first, then the table and the three commands
```

## Anti-goals

A bench that reports one number per engine and hides the concurrency curve.
An install script that needs a CUDA toolkit or a compiler.
Results that cannot be regenerated on a machine with the same GPU.

## Resolved risk

vLLM 0.32.0's wheel was checked on this machine: torch 2.13.0+cu130 reports
capability (12, 0) and `torch.cuda.get_arch_list()` includes `sm_120`, so no
cu128 pin and no engine swap is needed. If a future vLLM upgrade drops it, report
the engine as unsupported rather than silently swapping models.
