import json

import pytest

from bench import compare as compare_module


def _write(
    path,
    *,
    engine="ollama",
    concurrency=1,
    tokens=100.0,
    p50=1000.0,
    per_stream=100.0,
    p95=1200.0,
    model="qwen3-8b-q4",
    sha="abc",
    max_tokens=256,
    slots=32,
    ctx=1024,
    meta=True,
):
    payload = {
        "engine": engine,
        "concurrency": concurrency,
        "tokens_per_s": tokens,
        "p50_ms": p50,
        "p95_ms": p95,
        "per_stream_tps": per_stream,
    }
    if meta:
        payload["meta"] = {
            "prompt_sha256": sha,
            "max_tokens": max_tokens,
            "model": model,
            "slots": slots,
            "ctx": ctx,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _load(path):
    return json.loads(path.read_text())


def _pair(tmp_path, *, ref_tokens=100.0, your_tokens=90.0, **kwargs):
    ref = {"ollama-c1": _load(_write(tmp_path / "a/ollama-c1.json", tokens=ref_tokens))}
    yours = {
        "ollama-c1": _load(
            _write(tmp_path / "b/ollama-c1.json", tokens=your_tokens, **kwargs)
        )
    }
    return ref, yours


def test_delta_is_relative_to_the_reference():
    assert compare_module.delta(100.0, 90.0) == pytest.approx(-0.10)
    assert compare_module.delta(100.0, 120.0) == pytest.approx(0.20)


def test_compare_reports_band_verdict(tmp_path):
    ref, yours = _pair(tmp_path, your_tokens=50.0)
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "out"
    assert row["metrics"]["tokens_per_s"]["delta"] == pytest.approx(-0.50)


def test_compare_marks_ok_inside_tolerance(tmp_path):
    ref, yours = _pair(tmp_path, your_tokens=95.0)
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "ok"


@pytest.mark.parametrize(
    ("kwargs", "key"),
    [
        ({"sha": "other"}, "prompt_sha256"),
        ({"model": "other"}, "model"),
        ({"slots": 8}, "slots"),
        ({"ctx": 512}, "ctx"),
        ({"max_tokens": 1}, "max_tokens"),
    ],
)
def test_compare_flags_incomparable_when_meta_differs(tmp_path, kwargs, key):
    ref, yours = _pair(tmp_path, **kwargs)
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "incomparable"
    assert key in row["reason"]


def test_compare_flags_rows_without_meta(tmp_path):
    ref, yours = _pair(tmp_path, meta=False)
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "incomparable"
    assert row["reason"] == "no meta"


def test_compare_reports_missing_stems(tmp_path):
    ref = {
        "ollama-c1": _load(_write(tmp_path / "a/ollama-c1.json")),
        "vllm-c1": _load(_write(tmp_path / "a/vllm-c1.json")),
    }
    yours = {"ollama-c1": _load(_write(tmp_path / "b/ollama-c1.json"))}
    rows = {row["stem"]: row for row in compare_module.compare(ref, yours)}
    assert rows["vllm-c1"]["status"] == "missing"
    assert rows["vllm-c1"]["reason"] == "only in reference"


def test_compare_none_metric_is_not_divided(tmp_path):
    ref, yours = _pair(tmp_path, p50=None)
    row = compare_module.compare(ref, yours)[0]
    assert row["status"] == "incomparable"
    assert "p50_ms" in row["reason"]


def test_main_returns_zero_with_out_rows_by_default(tmp_path, capsys):
    a = _write(tmp_path / "a/ollama-c1.json", tokens=100)
    b = _write(tmp_path / "b/ollama-c1.json", tokens=50)
    assert compare_module.main([str(a.parent), str(b.parent)]) == 0


def test_main_strict_returns_one_on_out(tmp_path):
    a = _write(tmp_path / "a/ollama-c1.json", tokens=100)
    b = _write(tmp_path / "b/ollama-c1.json", tokens=50)
    assert compare_module.main([str(a.parent), str(b.parent), "--strict"]) == 1


def test_main_accepts_two_single_files(tmp_path, capsys):
    a = _write(tmp_path / "a/ollama-c1.json", tokens=100)
    b = _write(tmp_path / "b/ollama-c1.json", tokens=95)
    assert compare_module.main([str(a), str(b)]) == 0
    assert "ollama-c1" in capsys.readouterr().out


def test_collect_ignores_nested_directories(tmp_path):
    _write(tmp_path / "r/ollama-c1.json")
    _write(tmp_path / "r/local/ollama-c1.json")
    assert set(compare_module.collect(tmp_path / "r")) == {"ollama-c1"}


def test_to_table_prints_both_tables_and_the_reason(tmp_path):
    ref = {
        "ollama-c1": _load(_write(tmp_path / "a/ollama-c1.json")),
        "vllm-c1": _load(_write(tmp_path / "a/vllm-c1.json")),
    }
    yours = {
        "ollama-c1": _load(_write(tmp_path / "b/ollama-c1.json", sha="other")),
        "vllm-c1": _load(_write(tmp_path / "b/vllm-c1.json")),
    }
    text = compare_module.to_table(compare_module.compare(ref, yours))
    assert "tokens_per_s" in text
    assert "prompt_sha256 differs" in text
