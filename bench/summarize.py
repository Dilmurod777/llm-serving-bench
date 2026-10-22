from pathlib import Path

from .config import GPU_PRICE_PER_HOUR


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value at or above q of the sample."""
    ordered = sorted(values)
    idx = max(0, int(-(-q * len(ordered) // 1)) - 1)  # ceil(q*n) - 1
    return ordered[idx]


def summarize(
    records: list[dict],
    concurrency: int,
    duration_s: float,
    peak_vram_mb: int,
    gpu_price_per_hour: float = GPU_PRICE_PER_HOUR,
) -> dict:
    ok = [r for r in records if not r.get("error")]
    tokens = sum(r["completion_tokens"] for r in ok)
    latencies = [r["latency_ms"] for r in ok]
    tokens_per_s = tokens / duration_s if duration_s > 0 else 0.0
    cost = (
        gpu_price_per_hour / (tokens_per_s * 3600) * 1e6 if tokens_per_s > 0 else None
    )
    return {
        "concurrency": concurrency,
        "requests": len(records),
        "errors": len(records) - len(ok),
        "output_tokens": tokens,
        "duration_s": round(duration_s, 2),
        "tokens_per_s": round(tokens_per_s, 2),
        "p50_ms": round(_percentile(latencies, 0.50), 1) if latencies else None,
        "p95_ms": round(_percentile(latencies, 0.95), 1) if latencies else None,
        "peak_vram_mb": peak_vram_mb,
        "cost_per_1m_tokens": round(cost, 2) if cost is not None else None,
    }
