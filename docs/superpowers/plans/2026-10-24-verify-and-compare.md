# Reproduce and Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reader can run one sweep on their own GPU, compare it against the committed results with a delta table, and diagnose a wrong number from a symptom table.

**Architecture:** Results gain a comparability record (`model`, `slots`, `ctx`) in `meta`; a new pure module `bench/compare.py` matches result files by stem and prints signed deltas with a tolerance band; `scripts/run_all.sh` gains a `verify` profile that reuses its engine lifecycle and ends by calling compare.

**Tech Stack:** Python 3.12, stdlib only for compare (`json`, `argparse`, `pathlib`), pytest, bash.

**Spec:** `docs/superpowers/specs/2026-10-24-verify-and-compare-design.md`

## Global Constraints

- Dependencies stay `httpx` (plus pytest); compare must import nothing else.
- `python -m bench` behaviour is unchanged when `--model` is absent.
- No new chart.
- Every result JSON written by `bench` carries `meta` with `engine_version`, `gpu`, `date_utc`, `prompt_sha256`, `max_tokens`, `model`, `slots`, `ctx`.
- Compare is offline: reads JSON files only, no HTTP, no GPU.
- Tests run without a GPU or network: `cd D:\Projects\dev\llm-serving-bench && .venv/bin/python -m pytest -q` inside WSL (`wsl -d Ubuntu-24.04 -- bash -lc "..."`). Files must be LF (`sed -i 's/\r$//'`).

## Review Focus

- A pair whose `meta` disagrees on prompt set, `max_tokens`, `model`, slots or context must never produce a delta — silently comparing different shapes is the failure this feature exists to prevent.
- Result files with no `meta` at all (12 committed `vllm-fp8`/`vllm-spec` files) must be reported, not crash, not compared.
- Directory matching must not merge `results/ollama-c1.json` with `results/short/ollama-c1.json`, and must not treat `results/local` as a member of `results`.
- A metric that is `None` on either side (all requests failed) must not divide by zero.
- `--model llama.cpp` must fail loudly rather than run and produce a misleading row.

---

### Task 1: Comparability fields in `meta` and `--model`

**Files:**
- Modify: `bench/__main__.py` (`build_engine`, `run`, `main`)
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: existing `DEFAULTS`, `Engine` classes, `summarize`.
- Produces: `build_engine(name: str, base_url: str | None = None, max_tokens: int = MAX_TOKENS, model: str | None = None)`; `run(..., model: str | None = None, slots: int = 32, ctx: int = 1024)`; CLI flags `--model`, `--slots`, `--ctx`; `meta` keys `model`, `slots`, `ctx`.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli.py`

```python
def test_meta_records_model_slots_and_ctx(tmp_path):
    out = tmp_path / "s.json"
    summary = asyncio.run(
        run("ollama", 1, out, "data/prompts.jsonl", "http://x", True,
            model="my-model", slots=8, ctx=512)
    )
    assert summary["meta"]["model"] == "my-model"
    assert summary["meta"]["slots"] == 8
    assert summary["meta"]["ctx"] == 512


def test_meta_defaults_to_the_engine_default_model(tmp_path):
    out = tmp_path / "s.json"
    summary = asyncio.run(
        run("ollama", 1, out, "data/prompts.jsonl", "http://x", True)
    )
    assert summary["meta"]["model"] == "qwen3-8b-q4"
    assert summary["meta"]["slots"] == 32
    assert summary["meta"]["ctx"] == 1024


def test_model_override_reaches_the_engine():
    engine = build_engine("vllm", model="someone/other")
    assert engine.model == "someone/other"


def test_model_flag_is_rejected_for_llamacpp(capsys):
    with pytest.raises(SystemExit):
        main(["--engine", "llamacpp", "--model", "x"])
```

Reuse the existing fake `httpx` transport fixture in that file; keep the existing tests passing unchanged.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_cli.py -q`
Expected: failures on `model`/`slots`/`ctx` (TypeError or KeyError), `build_engine` unexpected keyword, and no `SystemExit`.

- [ ] **Step 3: Implement** in `bench/__main__.py`

- `build_engine(..., model=None)` selects `model or DEFAULTS[name][1]`; for `llamacpp` the model stays `None`.
- `run()` passes `model` to `build_engine`, writes `"model": model or DEFAULTS[engine_name][1]`, `"slots": slots`, `"ctx": ctx` into `meta`.
- `main()` adds `--model`, `--slots` (int, default 32), `--ctx` (int, default 1024). When `--model` is given with `--engine llamacpp`, call `parser.error("--model is not supported for llamacpp: the server loads the GGUF file")`, which exits with status 2.
- `meta["model"]` for `llamacpp` is the string `"gguf-file"`.

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (24 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add bench/__main__.py tests/test_cli.py
git commit -m "feat: record model, slots and context in result meta"
```

---

### Task 2: `bench/compare.py` — load, match, delta table

**Files:**
- Create: `bench/compare.py`
- Create: `tests/test_compare.py`

**Interfaces:**
- Consumes: result JSON files written by `bench`.
- Produces: `collect(target: str | Path) -> dict[str, dict]` (stem → summary), `delta(reference: float, yours: float) -> float` returning a fraction, `compare(reference: dict[str, dict], yours: dict[str, dict], tolerance: float = 0.25) -> list[dict]`, `to_table(rows: list[dict]) -> str`.

Row shape (both tables are rendered from these):

```python
{"stem": "ollama-c1", "status": "ok", "reason": None,
 "metrics": {"tokens_per_s": {"reference": 106.07, "yours": 100.0, "delta": -0.0572}}}
```

- [ ] **Step 1: Write the failing tests** in `tests/test_compare.py`

```python
def _write(path, *, engine="ollama", concurrency=1, tokens=100.0, p50=1000.0,
           per_stream=100.0, p95=1200.0, model="qwen3-8b-q4",
           sha="abc", max_tokens=256, slots=32, ctx=1024, meta=True):
    payload = {"engine": engine, "concurrency": concurrency,
               "tokens_per_s": tokens, "p50_ms": p50, "p95_ms": p95,
               "per_stream_tps": per_stream}
    if meta:
        payload["meta"] = {"prompt_sha256": sha, "max_tokens": max_tokens,
                           "model": model, "slots": slots, "ctx": ctx}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_delta_is_relative_to_the_reference():
    assert compare_module.delta(100.0, 90.0) == pytest.approx(-0.10)
    assert compare_module.delta(100.0, 120.0) == pytest.approx(0.20)


def test_compare_reports_band_verdict(tmp_path):
    ref = {"ollama-c1": json.loads(_write(tmp_path / "a/ollama-c1.json", tokens=100).read_text())}
    yours = {"ollama-c1": json.loads(_write(tmp_path / "b/ollama-c1.json", tokens=90).read_text())}
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "out"
    assert row["metrics"]["tokens_per_s"]["delta"] == pytest.approx(-0.10)
```

Plus, same file:

- `test_compare_marks_ok_inside_tolerance` (tokens 95 → `ok`).
- `test_compare_flags_incomparable_when_meta_differs` — build the "yours" row with `sha="other"`; assert `status == "incomparable"` and `"prompt_sha256" in row["reason"]`; repeat for `model`, `slots`, `ctx`, `max_tokens`.
- `test_compare_flags_rows_without_meta` — `meta=False` → `status == "incomparable"`, `row["reason"] == "no meta"`.
- `test_compare_reports_missing_stems` — one side has `vllm-c1`, the other does not → `status == "missing"`.
- `test_compare_none_metric_is_not_divided` — `p50=None` on both sides → row `status == "incomparable"`, `"p50_ms" in row["reason"]`, no exception.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: `ModuleNotFoundError: No module named 'bench.compare'`.

- [ ] **Step 3: Implement** `bench/compare.py`

- `GATED = ("tokens_per_s", "per_stream_tps", "p50_ms", "p95_ms")`, `UNGATED = ("peak_vram_mb", "avg_power_w", "joules_per_1m_tokens", "cost_per_1m_tokens")`, `KEYS = ("prompt_sha256", "max_tokens", "model", "slots", "ctx")` — all module constants, because the tests and the README reference them.
- `collect(target)`: if `target` is a file, return `{path.stem: json.loads(...)}`; if a directory, glob `*.json` **one level only** (`sorted(dir.glob("*.json"))`), skipping any subdirectory.
- `delta(reference, yours)` returns `(yours - reference) / reference`.
- `compare(...)`: for each stem in `sorted(set(reference) | set(yours))`:
  - missing on one side → `{"stem", "status": "missing", "reason": "only in reference|yours", "metrics": {}}`;
  - either side without `meta` → `incomparable`, reason `"no meta"`;
  - any `KEYS` differing → `incomparable`, reason `"<key> differs: <a> != <b>"` (first key only);
  - a gated metric that is `None` on either side → `incomparable`, reason `"<metric> is null"`;
  - otherwise `status` = `"ok"` if every gated metric's `abs(delta) <= tolerance`, else `"out"`.
- `to_table(rows)`: one fixed-width text table for gated metrics and one for ungated metrics; `—` for missing values; returns a single string.

- [ ] **Step 4: Run to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/compare.py tests/test_compare.py
git commit -m "feat: compare two result sets with signed deltas"
```

---

### Task 3: `compare` CLI — flags, exit codes, nested directory

**Files:**
- Modify: `bench/compare.py`
- Modify: `tests/test_compare.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> int`; CLI `python -m bench.compare A B [--tolerance 0.25] [--strict]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_main_returns_zero_with_out_rows_by_default(tmp_path, capsys):
    a = _write(tmp_path / "a/ollama-c1.json", tokens=100)
    b = _write(tmp_path / "b/ollama-c1.json", tokens=50)
    assert compare_module.main([str(a.parent), str(b.parent)]) == 0


def test_main_strict_returns_one_on_out(tmp_path):
    a = _write(tmp_path / "a/ollama-c1.json", tokens=100)
    b = _write(tmp_path / "b/ollama-c1.json", tokens=50)
    assert compare_module.main([str(a.parent), str(b.parent), "--strict"]) == 1


def test_collect_ignores_nested_directories(tmp_path):
    _write(tmp_path / "r/ollama-c1.json")
    _write(tmp_path / "r/local/ollama-c1.json")
    assert set(compare_module.collect(tmp_path / "r")) == {"ollama-c1"}
```

Also `test_main_accepts_two_single_files` (file paths instead of directories → one row).

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: failures on missing `main`, and `collect` returning `local/ollama-c1` today.

- [ ] **Step 3: Implement**

- `collect` already globs one level; confirm the nested file is not returned.
- `main` parses `reference`, `yours`, `--tolerance` (float, default 0.25), `--strict`; prints the two tables and a one-line summary (`N ok, N out, N incomparable, N missing`); returns `1` when `--strict` and any row is not `ok`, else `0`. Wire with `if __name__ == "__main__": raise SystemExit(main())`.

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_compare.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bench/compare.py tests/test_compare.py
git commit -m "feat: compare CLI with tolerance band and strict exit"
```

---

### Task 4: verify profile in `run_all.sh`, `MODEL` isolation, Makefile target

**Files:**
- Modify: `scripts/run_all.sh`
- Modify: `Makefile`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `bench.compare.main` (Task 3), `bench --model --slots --ctx` (Task 1).
- Produces: `PROFILE=verify` (one c1 sweep, `results/local/`, then compare); `MODEL=<id>` writes to `results/local/`; `make verify [ENGINE=...]`.

- [ ] **Step 1: Add the profile and model plumbing**

In the `PROFILE` case add:

```bash
  verify) CONCS="1"; SLOTS=32; CTX=1024 ;;
```

and set `OUTDIR="results"` for baseline/wide/long, `OUTDIR="results/local"` when `PROFILE=verify` or `MODEL` is non-empty. `mkdir -p` must include `$OUTDIR`. The `bench` invocations pass `--slots "$SLOTS" --ctx "$CTX"` and, when `MODEL` is set, `--model "$MODEL"`. `serve_vllm` uses `${MODEL:-$2}`. For `llamacpp`, a non-empty `MODEL` prints `MODEL is ignored for llamacpp: the server loads models/Qwen3-8B-Q4_K_M.gguf` and unsets it.

- [ ] **Step 2: Call compare at the end of the verify profile**

```bash
if [ "$PROFILE" = verify ]; then
  "$PY" -m bench.compare "results/${ENGINE}-c1.json" "$OUTDIR/${ENGINE}-c1.json"
fi
```

using the single-engine name from `ENGINES`; when several engines are selected, loop over them. Exit non-zero if compare does.

- [ ] **Step 3: Makefile**

```make
verify:
	PROFILE=verify ENGINES=$(or $(ENGINE),ollama) bash scripts/run_all.sh
```

Add `verify` to `.PHONY`.

- [ ] **Step 4: Gitignore**

Add `results/local/` to `.gitignore` so a reader's run never dirties the tree.

- [ ] **Step 5: Verify by inspection and a dry run**

Run: `bash -n scripts/run_all.sh && echo OK` and `make -n verify` inside the repo (WSL).
Expected: `OK`; the recipe expands to `PROFILE=verify ENGINES=ollama bash scripts/run_all.sh`. Do **not** run the full sweep here — Task 6 does that once.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_all.sh Makefile .gitignore
git commit -m "feat: verify profile compares one sweep against the reference"
```

---

### Task 5: README teaching section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `## Reproduce on your machine`** after `## Reproduce`, with the five subsections from the spec: prerequisites, wall times (`baseline ≈ 40 min`, `SHORT=only ≈ 10 min`, `PROFILE=wide ≈ 20 min`, `PROFILE=long ≈ 15 min`, `make verify ≈ 8 min`), three commands to a comparable row, "What differs legitimately", and "When your number is wrong" as a symptom → cause → fix table with the five rows named in the spec.
- [ ] **Step 2: State the comparison contract** in one paragraph: first argument is the reference, delta is `(yours − reference) / reference`, gated metrics are throughput (total and per-user) and p50/p95, band is 25% and advisory unless `--strict`, and the 12 meta-less FP8/spec files are reported as incomparable.
- [ ] **Step 3: Update the existing blocks** — `## Reproduce` gains one line pointing at the new section; Layout mentions `bench/compare.py`; the test count in the Reproduce paragraph rises from 24 to the real number after Task 6.
- [ ] **Step 4: Verify** by reading the rendered section and grepping for the anchors (`## Reproduce on your machine`, `make verify`, `--strict`).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: how to reproduce and compare on your own machine"
```

---

### Task 6: End-to-end verify run and push

**Files:** none.

- [ ] **Step 1: Run the suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass; count recorded for the README.

- [ ] **Step 2: Run `make verify` for real** (GPU free)

Run: `make verify ENGINE=ollama`
Expected: one 200-request sweep into `results/local/ollama-c1.json`, then a table with one row and a band verdict; exit 0. If the machine is hot from earlier runs, note `avg_power_w` in the report.

- [ ] **Step 3: Confirm the guard fires**

Copy `results/local/ollama-c1.json`, edit its `meta.max_tokens` to `1`, run compare against it, and confirm `incomparable: max_tokens differs`. Delete the copy.

- [ ] **Step 4: Confirm the 12 meta-less files are reported**

Run `python -m bench.compare results results` and check the FP8/spec stems appear as `incomparable: no meta`, with no exception.

- [ ] **Step 5: Commit the test-count fix if Task 5 left a placeholder, then push**

```bash
git push origin master
```

---

## Self-Review

- Spec coverage: `meta` fields (Task 1), compare module and guards (Tasks 2–3), verify profile and `MODEL` isolation (Task 4), README teaching section (Task 5), live proof (Task 6). Every spec section maps to a task.
- The nested-directory, missing-meta, missing-stem, null-metric and `--model llamacpp` cases each have a test in the owning task (Review Focus).
- Names used consistently: `collect`, `delta`, `compare`, `to_table`, `main`, `GATED`/`UNGATED`/`KEYS`, `PROFILE=verify`, `OUTDIR`.
