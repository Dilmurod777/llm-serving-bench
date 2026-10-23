import json

import pytest

from bench import __main__ as cli
from bench.engines import Completion


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
        cli, "build_engine", lambda name, base_url=None, max_tokens=None: FakeEngine()
    )
    monkeypatch.setattr(cli, "assert_idle", lambda *a, **k: None)

    out = tmp_path / "ollama-c1.json"
    cli.main(["--engine", "ollama", "--concurrency", "1", "--out", str(out)])

    summary = json.loads(out.read_text())
    assert summary["engine"] == "ollama" and summary["tokens_per_s"] > 0
    assert summary["errors"] == 0 and summary["p95_ms"] == 5.0
    assert summary["meta"]["engine_version"] == "test-1"
    assert summary["meta"]["max_tokens"] == 256
    assert len(out.with_suffix(".jsonl").read_text().strip().splitlines()) == 200


def test_cli_rejects_an_unknown_engine():
    with pytest.raises(SystemExit):
        cli.main(["--engine", "tgi", "--concurrency", "1"])
