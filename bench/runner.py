import asyncio
import time

from .engines import EngineError


async def run_sweep(
    engine, prompts: list[str], concurrency: int, timeout_s: float = 120.0
) -> tuple[list[dict], float]:
    """Fire every prompt with at most `concurrency` requests in flight."""
    sem = asyncio.Semaphore(concurrency)
    records: list[dict | None] = [None] * len(prompts)

    async def one(i: int, prompt: str) -> None:
        async with sem:
            try:
                out = await asyncio.wait_for(engine.complete(prompt), timeout_s)
                records[i] = {
                    "prompt_id": i,
                    "latency_ms": round(out.latency_ms, 1),
                    "completion_tokens": out.completion_tokens,
                    "text_chars": len(out.text),
                    "error": None,
                }
            except Exception as exc:  # one bad request is a record, not a dead sweep
                records[i] = {
                    "prompt_id": i,
                    "latency_ms": 0,
                    "completion_tokens": 0,
                    "text_chars": 0,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }

    start = time.perf_counter()
    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    return records, time.perf_counter() - start
