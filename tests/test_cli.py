import asyncio
import json

import pytest

from bench import __main__ as cli
from bench.engines import Completion
from bench.prompts import prompt_sha


class FakeEngine:
    name = "ollama"

    async def complete(self, prompt):
        return Completion(completion_tokens=7, latency_ms=5.0, text="ok")

    async def version(self):
        return "test-1"

    async def aclose(self):
        pass


def test_cli_writes_summary_and_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_engine",
        lambda name, base_url=None, max_tokens=None, model=None: FakeEngine(),
    )
    monkeypatch.setattr(cli, "assert_idle", lambda *a, **k: None)

    out = tmp_path / "ollama-c1.json"
    cli.main(["--engine", "ollama", "--concurrency", "1", "--out", str(out)])

    summary = json.loads(out.read_text())
    assert summary["engine"] == "ollama" and summary["tokens_per_s"] > 0
    assert summary["errors"] == 0 and summary["p95_ms"] == 5.0
    assert summary["meta"]["engine_version"] == "test-1"
    assert summary["meta"]["max_tokens"] == 256
    assert summary["meta"]["prompt_sha256"] == prompt_sha()
    assert len(out.with_suffix(".jsonl").read_text().strip().splitlines()) == 200


def test_cli_rejects_an_unknown_engine():
    with pytest.raises(SystemExit):
        cli.main(["--engine", "tgi", "--concurrency", "1"])


def _fake_engine(monkeypatch):
    monkeypatch.setattr(
        cli,
        "build_engine",
        lambda name, base_url=None, max_tokens=None, model=None: FakeEngine(),
    )


def test_meta_records_model_slots_and_ctx(tmp_path, monkeypatch):
    _fake_engine(monkeypatch)
    out = tmp_path / "s.json"
    summary = asyncio.run(
        cli.run(
            "ollama",
            1,
            out,
            "data/prompts.jsonl",
            "http://x",
            True,
            model="my-model",
            slots=8,
            ctx=512,
        )
    )
    assert summary["meta"]["model"] == "my-model"
    assert summary["meta"]["slots"] == 8
    assert summary["meta"]["ctx"] == 512


def test_meta_defaults_to_the_engine_default_model(tmp_path, monkeypatch):
    _fake_engine(monkeypatch)
    out = tmp_path / "s.json"
    summary = asyncio.run(
        cli.run("ollama", 1, out, "data/prompts.jsonl", "http://x", True)
    )
    assert summary["meta"]["model"] == "qwen3-8b-q4"
    assert summary["meta"]["slots"] == 32
    assert summary["meta"]["ctx"] == 1024


def test_model_override_reaches_the_engine():
    engine = cli.build_engine("vllm", model="someone/other")
    assert engine.model == "someone/other"


def test_model_flag_is_rejected_for_llamacpp(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--engine", "llamacpp", "--model", "x"])
