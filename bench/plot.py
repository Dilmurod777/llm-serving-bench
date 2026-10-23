"""Charts and README tables, built from results/*.json only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ENGINES = ("ollama", "llamacpp", "vllm")
COLORS = {
    "ollama": "#b8321a",
    "llamacpp": "#15171a",
    "vllm": "#5d5f5a",
    "vllm-fp8": "#1f5138",
    "vllm-spec": "#3b5b8a",
}
W, H, PAD_L, PAD_B, PAD_T = 640, 360, 48, 56, 24

MAIN_COLUMNS = (
    ("engine", "engine"),
    ("concurrency", "concurrency"),
    ("tokens_per_s", "output tok/s"),
    ("p50_ms", "p50 ms"),
    ("p95_ms", "p95 ms"),
    ("peak_vram_mb", "peak VRAM MiB"),
    ("cost_per_1m_tokens", "$/1M output tokens"),
)
TTFT_COLUMNS = (
    ("engine", "engine"),
    ("concurrency", "concurrency"),
    ("p50_ms", "TTFT p50 ms"),
    ("p95_ms", "TTFT p95 ms"),
)
POWER_COLUMNS = (
    ("engine", "engine"),
    ("concurrency", "concurrency"),
    ("tokens_per_s", "output tok/s"),
    ("avg_power_w", "watts"),
    ("joules_per_1m_tokens", "J/1M output tokens"),
)
PERSTREAM_COLUMNS = (
    ("engine", "engine"),
    ("concurrency", "concurrency"),
    ("per_stream_tps", "tok/s per user"),
    ("tokens_per_s", "output tok/s in total"),
)


SHORT_COLUMNS = (
    ("engine", "engine"),
    ("concurrency", "concurrency"),
    ("requests", "requests"),
    ("tokens_per_s", "output tok/s"),
    ("cost_per_1m_tokens", "$/1M output tokens"),
)


def load_results(paths: list[str]) -> list[dict]:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    return sorted(rows, key=lambda r: (r["concurrency"], _order(r["engine"])))


def _order(engine: str) -> int:
    return ENGINES.index(engine) if engine in ENGINES else len(ENGINES)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return str(value)


def _label(value: float) -> str:
    """Compact enough that bar labels stay narrower than their bars."""
    a = abs(value)
    if a >= 1e6:
        return f"{value / 1e6:.2f}M"
    if a >= 1e4:
        return f"{value / 1e3:.0f}k"
    if a >= 1e3:
        return f"{value:.0f}"
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def to_markdown(rows: list[dict], columns=MAIN_COLUMNS) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |\n"
    head += "|" + "---|" * len(columns) + "\n"
    body = "".join(
        "| " + " | ".join(_fmt(r.get(key)) for key, _ in columns) + " |\n" for r in rows
    )
    return head + body


def to_svg(
    rows: list[dict],
    path: str | Path,
    metric: str = "tokens_per_s",
    ylabel: str = "tok/s",
    engines: tuple[str, ...] | None = None,
) -> None:
    engines = engines or tuple(
        e for e in ENGINES if any(r["engine"] == e for r in rows)
    )
    concurrencies = sorted({r["concurrency"] for r in rows})
    peak = max((r.get(metric) or 0 for r in rows), default=1) or 1
    plot_w, plot_h = W - PAD_L - 16, H - PAD_B - PAD_T
    group_w = plot_w / max(len(concurrencies), 1)
    bar_w = group_w / (len(engines) + 1)
    axis_y = (PAD_T + H - PAD_B) / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="IBM Plex Mono, monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#e4e3dd"/>',
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="#b6b5ad"/>',
        f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - 16}" y2="{H - PAD_B}" stroke="#b6b5ad"/>',
        f'<text x="{PAD_L - 8}" y="{PAD_T + 8}" text-anchor="end" fill="#5d5f5a">{_label(peak)}</text>',
        f'<text x="14" y="{axis_y:.0f}" transform="rotate(-90 14 {axis_y:.0f})" '
        f'text-anchor="middle" fill="#5d5f5a">{ylabel}</text>',
        f'<text x="{PAD_L - 8}" y="{H - PAD_B}" text-anchor="end" fill="#5d5f5a">0</text>',
    ]
    for gi, conc in enumerate(concurrencies):
        gx = PAD_L + gi * group_w
        parts.append(
            f'<text x="{gx + group_w / 2:.0f}" y="{H - PAD_B + 18}" text-anchor="middle" fill="#15171a">c={conc}</text>'
        )
        for ei, engine in enumerate(engines):
            row = next(
                (r for r in rows if r["engine"] == engine and r["concurrency"] == conc),
                None,
            )
            if not row or row.get(metric) is None:
                continue
            h = (row[metric] / peak) * plot_h
            x = gx + bar_w * (ei + 0.5)
            parts.append(
                f'<rect x="{x:.1f}" y="{H - PAD_B - h:.1f}" width="{bar_w * 0.8:.1f}" height="{h:.1f}" fill="{COLORS[engine]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w * 0.4:.1f}" y="{H - PAD_B - h - 4:.1f}" text-anchor="middle" fill="#15171a">{_label(round(row[metric], 1))}</text>'
            )
    for ei, engine in enumerate(engines):
        lx = PAD_L + ei * 95
        parts.append(
            f'<rect x="{lx}" y="{H - 20}" width="10" height="10" fill="{COLORS[engine]}"/>'
        )
        parts.append(f'<text x="{lx + 15}" y="{H - 11}" fill="#15171a">{engine}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def _load_all(paths: list[str]) -> list[tuple[Path, dict]]:
    return [(Path(p), json.loads(Path(p).read_text())) for p in paths]


def main(argv: list[str]) -> None:
    paths = argv or sorted(
        str(p)
        for p in list(Path("results").glob("*.json"))
        + list(Path("results/wide").glob("*.json"))
        + list(Path("results/long").glob("*.json"))
    )
    if not paths:
        raise SystemExit("no results; run make bench first")
    loaded = _load_all(paths)
    Path("docs").mkdir(exist_ok=True)

    def pick(
        engines: tuple[str, ...], ttft: bool | None = None, folder: str = "results"
    ) -> list[dict]:
        return [
            row
            for path, row in loaded
            if row["engine"] in engines
            and path.parent.name == folder
            and (ttft is None or path.stem.endswith("-ttft") == ttft)
        ]

    base = sorted(
        pick(ENGINES, ttft=False), key=lambda r: (r["concurrency"], _order(r["engine"]))
    )
    to_svg(base, "docs/chart.svg")
    print(to_markdown(base))

    per_stream = [r for r in base if r.get("per_stream_tps") is not None]
    if per_stream:
        to_svg(per_stream, "docs/perstream.svg", "per_stream_tps", "tok/s per user")
        print(to_markdown(per_stream, PERSTREAM_COLUMNS))

    ttft_rows = sorted(
        pick(ENGINES, ttft=True), key=lambda r: (r["concurrency"], _order(r["engine"]))
    )
    if ttft_rows:
        to_svg(ttft_rows, "docs/ttft.svg", "p50_ms", "ms")
        print(to_markdown(ttft_rows, TTFT_COLUMNS))

    for engine in ("vllm-fp8", "vllm-spec"):
        extra = sorted(
            pick(("vllm", engine), ttft=False),
            key=lambda r: (r["concurrency"], _order(r["engine"])),
        )
        if extra:
            name = engine.split("-")[1]
            to_svg(extra, f"docs/{name}.svg", engines=("vllm", engine))
            print(to_markdown(sorted(extra, key=lambda r: r["concurrency"])))

    power_rows = [r for r in base if r.get("joules_per_1m_tokens") is not None]
    if power_rows:
        to_svg(power_rows, "docs/energy.svg", "joules_per_1m_tokens", "J/1M")
        print(to_markdown(power_rows, POWER_COLUMNS))

    wide_rows = sorted(
        pick(ENGINES, folder="wide"),
        key=lambda r: (r["concurrency"], _order(r["engine"])),
    )
    if wide_rows:
        to_svg(wide_rows, "docs/wide.svg")
        print(to_markdown(wide_rows))

    long_rows = sorted(
        pick(ENGINES, folder="long"),
        key=lambda r: (r["concurrency"], _order(r["engine"])),
    )
    if long_rows:
        # The finding is the tail: at 8 concurrent 4k prompts the GGUF p95 is 35 s.
        to_svg(long_rows, "docs/ttft-long.svg", "p95_ms", "ms")
        print(to_markdown(long_rows, TTFT_COLUMNS))

    short_paths = sorted(str(p) for p in Path("results/short").glob("*.json"))
    if short_paths:
        short = sorted(
            (json.loads(Path(p).read_text()) for p in short_paths),
            key=lambda r: (r["concurrency"], _order(r["engine"])),
        )
        to_svg(short, "docs/short.svg", "cost_per_1m_tokens", "$/1M")
        print(to_markdown(short, SHORT_COLUMNS))


if __name__ == "__main__":
    main(sys.argv[1:])
