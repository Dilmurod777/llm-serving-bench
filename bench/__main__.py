import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import CONCURRENCIES, MAX_TOKENS
from .engines import LlamaCppEngine, OllamaEngine, VllmEngine
from .prompts import load_prompts, prompt_sha
from .runner import run_sweep
from .summarize import summarize
from .vram import VramSampler, assert_idle, gpu_name

DEFAULTS = {
    "ollama": ("http://127.0.0.1:11434", "qwen3-8b-q4"),
    "llamacpp": ("http://127.0.0.1:8080", None),
    "vllm": ("http://127.0.0.1:8000", "Qwen/Qwen3-8B-AWQ"),
    "vllm-fp8": ("http://127.0.0.1:8000", "Qwen/Qwen3-8B-FP8"),
    "vllm-spec": ("http://127.0.0.1:8000", "Qwen/Qwen3-8B-AWQ"),
}


def build_engine(name: str, base_url: str | None = None, max_tokens: int = MAX_TOKENS):
    url, model = DEFAULTS[name]
    url = base_url or url
    if name == "ollama":
        return OllamaEngine(url, model, max_tokens)
    if name == "llamacpp":
        return LlamaCppEngine(url, max_tokens)
    return VllmEngine(url, model, max_tokens)


async def run(
    engine_name: str,
    concurrency: int,
    out: Path,
    prompts_path: str,
    base_url: str | None,
    skip_idle: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    engine = build_engine(engine_name, base_url, max_tokens)
    prompts = load_prompts(prompts_path)
    # Only the first sweep of an engine needs a free GPU: the engine stays
    # resident for the next concurrency, so run_all.sh skips the gate there.
    if not skip_idle:
        assert_idle()
    with VramSampler() as vram:
        records, duration_s = await run_sweep(engine, prompts, concurrency)
    engine_version = await engine.version()
    await engine.aclose()
    summary = {
        "engine": engine_name,
        "meta": {
            "engine_version": engine_version,
            "gpu": gpu_name(),
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt_sha256": prompt_sha(prompts_path),
            "max_tokens": max_tokens,
        },
        **summarize(
            records,
            concurrency,
            duration_s,
            vram.peak_mb,
            power_w=vram.avg_power_w,
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    out.with_suffix(".jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    print(json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="bench", description="One engine, one concurrency."
    )
    parser.add_argument("--engine", required=True, choices=sorted(DEFAULTS))
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCIES[0], choices=CONCURRENCIES
    )
    parser.add_argument(
        "--base-url", default=None, help="default: the engine's local port"
    )
    parser.add_argument(
        "--out", default=None, help="default: results/<engine>-c<N>.json"
    )
    parser.add_argument("--prompts", default="data/prompts.jsonl")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="1 turns the sweep into a time-to-first-token measurement",
    )
    parser.add_argument(
        "--skip-idle-check",
        action="store_true",
        help="the engine is already loaded from a previous sweep",
    )
    args = parser.parse_args(argv)
    suffix = "-ttft" if args.max_tokens == 1 else ""
    out = Path(args.out or f"results/{args.engine}-c{args.concurrency}{suffix}.json")
    asyncio.run(
        run(
            args.engine,
            args.concurrency,
            out,
            args.prompts,
            args.base_url,
            args.skip_idle_check,
            args.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
