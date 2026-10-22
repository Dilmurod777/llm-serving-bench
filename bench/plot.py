"""Chart and README table, built from results/*.json only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ENGINES = ("ollama", "llamacpp", "vllm")
COLORS = {"ollama": "#b8321a", "llamacpp": "#15171a", "vllm": "#5d5f5a"}
W, H, PAD_L, PAD_B, PAD_T = 640, 360, 48, 44, 24


def load_results(paths: list[str]) -> list[dict]:
    rows = [json.loads(Path(p).read_text()) for p in paths]
    return sorted(rows, key=lambda r: (r["concurrency"], ENGINES.index(r["engine"])))


def to_markdown(rows: list[dict]) -> str:
    head = (
        "| engine | concurrency | output tok/s | p50 ms | p95 ms | peak VRAM MiB | $/1M output tokens |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {r['engine']} | {r['concurrency']} | {r['tokens_per_s']} | {r['p50_ms']} | {r['p95_ms']} "
        f"| {r['peak_vram_mb']} | {r['cost_per_1m_tokens']} |\n"
        for r in rows
    )
    return head + body


def to_svg(rows: list[dict], path: str | Path) -> None:
    concurrencies = sorted({r["concurrency"] for r in rows})
    peak = max((r["tokens_per_s"] for r in rows), default=1) or 1
    plot_w, plot_h = W - PAD_L - 16, H - PAD_B - PAD_T
    group_w = plot_w / max(len(concurrencies), 1)
    bar_w = group_w / (len(ENGINES) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="IBM Plex Mono, monospace" font-size="11">',
        f'<rect width="{W}" height="{H}" fill="#e4e3dd"/>',
        f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}" stroke="#b6b5ad"/>',
        f'<line x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - 16}" y2="{H - PAD_B}" stroke="#b6b5ad"/>',
        f'<text x="{PAD_L - 8}" y="{PAD_T + 8}" text-anchor="end" fill="#5d5f5a">{round(peak)}</text>',
        f'<text x="8" y="{PAD_T + 8}" fill="#5d5f5a">tok/s</text>',
        f'<text x="8" y="{H - PAD_B}" fill="#5d5f5a">0</text>',
    ]
    for gi, conc in enumerate(concurrencies):
        gx = PAD_L + gi * group_w
        parts.append(
            f'<text x="{gx + group_w / 2:.0f}" y="{H - PAD_B + 18}" text-anchor="middle" fill="#15171a">c={conc}</text>'
        )
        for ei, engine in enumerate(ENGINES):
            row = next(
                (r for r in rows if r["engine"] == engine and r["concurrency"] == conc),
                None,
            )
            if not row:
                continue
            h = (row["tokens_per_s"] / peak) * plot_h
            x = gx + bar_w * (ei + 0.5)
            parts.append(
                f'<rect x="{x:.1f}" y="{H - PAD_B - h:.1f}" width="{bar_w * 0.8:.1f}" height="{h:.1f}" fill="{COLORS[engine]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_w * 0.4:.1f}" y="{H - PAD_B - h - 4:.1f}" text-anchor="middle" fill="#15171a">{row["tokens_per_s"]:.0f}</text>'
            )
    for ei, engine in enumerate(ENGINES):
        lx = PAD_L + ei * 90
        parts.append(
            f'<rect x="{lx}" y="{H - 20}" width="10" height="10" fill="{COLORS[engine]}"/>'
        )
        parts.append(f'<text x="{lx + 15}" y="{H - 11}" fill="#15171a">{engine}</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main(argv: list[str]) -> None:
    paths = argv or sorted(str(p) for p in Path("results").glob("*.json"))
    if not paths:
        raise SystemExit("no results; run make bench first")
    rows = load_results(paths)
    Path("docs").mkdir(exist_ok=True)
    to_svg(rows, "docs/chart.svg")
    print(to_markdown(rows))


if __name__ == "__main__":
    main(sys.argv[1:])
