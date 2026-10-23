"""HTTP adapters for the three serving stacks.

Every adapter sends a raw completion (no chat template) so the three engines
receive identical prompt tokens; only the transport differs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import MAX_TOKENS


class EngineError(RuntimeError):
    pass


@dataclass
class Completion:
    completion_tokens: int
    latency_ms: float
    text: str


class _Engine:
    name = "engine"

    def __init__(
        self,
        base_url: str,
        max_tokens: int = MAX_TOKENS,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._client = client

    async def _post(self, path: str, payload: dict) -> tuple[dict, float]:
        own = self._client is None
        client = self._client or httpx.AsyncClient(timeout=180.0)
        start = time.perf_counter()
        try:
            resp = await client.post(f"{self.base_url}{path}", json=payload)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            if own:
                await client.aclose()
        if resp.status_code != 200:
            raise EngineError(
                f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json(), elapsed

    async def _get_json(self, path: str) -> dict:
        own = self._client is None
        client = self._client or httpx.AsyncClient(timeout=10.0)
        try:
            resp = await client.get(f"{self.base_url}{path}")
            return resp.json() if resp.status_code == 200 else {}
        except Exception:
            return {}
        finally:
            if own:
                await client.aclose()

    async def version(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    async def complete(self, prompt: str) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class OllamaEngine(_Engine):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str = "qwen3-8b-q4",
        max_tokens: int = MAX_TOKENS,
        client=None,
    ):
        super().__init__(base_url, max_tokens, client)
        self.model = model

    async def version(self) -> str:
        return str((await self._get_json("/api/version")).get("version", "unknown"))

    async def complete(self, prompt: str) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "raw": True,
            "options": {"temperature": 0, "num_predict": self.max_tokens, "seed": 0},
        }
        data, elapsed = await self._post("/api/generate", payload)
        return Completion(int(data["eval_count"]), elapsed, data.get("response", ""))


class LlamaCppEngine(_Engine):
    name = "llamacpp"

    async def version(self) -> str:
        data = await self._get_json("/props")
        return str(data.get("build_info") or "unknown")

    async def complete(self, prompt: str) -> Completion:
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": False,
        }
        data, elapsed = await self._post("/completion", payload)
        return Completion(
            int(data["timings"]["predicted_n"]), elapsed, data.get("content", "")
        )


class VllmEngine(_Engine):
    name = "vllm"

    def __init__(
        self,
        base_url: str,
        model: str = "Qwen/Qwen3-8B-AWQ",
        max_tokens: int = MAX_TOKENS,
        client=None,
    ):
        super().__init__(base_url, max_tokens, client)
        self.model = model

    async def version(self) -> str:
        return str((await self._get_json("/version")).get("version", "unknown"))

    async def complete(self, prompt: str) -> Completion:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "seed": 0,
        }
        data, elapsed = await self._post("/v1/completions", payload)
        return Completion(
            int(data["usage"]["completion_tokens"]),
            elapsed,
            data["choices"][0].get("text", ""),
        )
