import asyncio
import json

import httpx
import pytest

from bench.engines import EngineError, LlamaCppEngine, OllamaEngine, VllmEngine


def _engine(cls, handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return cls("http://box", client=client, **kw)


def test_ollama_sends_raw_completion():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.url.path == "/api/generate"
        return httpx.Response(200, json={"response": "ok", "eval_count": 42})

    engine = _engine(OllamaEngine, handler, model="qwen3-8b-q4")
    out = asyncio.run(engine.complete("hello"))
    assert out.completion_tokens == 42
    assert (
        seen["raw"] is True
        and seen["model"] == "qwen3-8b-q4"
        and seen["stream"] is False
    )
    assert seen["options"]["temperature"] == 0 and seen["options"]["num_predict"] == 256


def test_llamacpp_reads_predicted_n():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.url.path == "/completion"
        return httpx.Response(
            200, json={"content": "x", "timings": {"predicted_n": 17}}
        )

    out = asyncio.run(_engine(LlamaCppEngine, handler).complete("hello"))
    assert out.completion_tokens == 17
    assert (
        seen["n_predict"] == 256
        and seen["temperature"] == 0
        and seen["stream"] is False
    )


def test_vllm_reads_usage():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        assert req.url.path == "/v1/completions"
        return httpx.Response(
            200,
            json={"choices": [{"text": "y"}], "usage": {"completion_tokens": 9}},
        )

    out = asyncio.run(
        _engine(VllmEngine, handler, model="Qwen/Qwen3-8B-AWQ").complete("hello")
    )
    assert out.completion_tokens == 9
    assert seen["max_tokens"] == 256 and seen["model"] == "Qwen/Qwen3-8B-AWQ"


def test_error_carries_engine_and_body():
    engine = _engine(LlamaCppEngine, lambda req: httpx.Response(500, text="CUDA OOM"))
    with pytest.raises(EngineError, match="llamacpp: HTTP 500: CUDA OOM"):
        asyncio.run(engine.complete("hello"))
