"""Compare two sets of result JSONs offline: no HTTP, no GPU."""

import json
from pathlib import Path

GATED = ("tokens_per_s", "per_stream_tps", "p50_ms", "p95_ms")
UNGATED = ("peak_vram_mb", "avg_power_w", "joules_per_1m_tokens", "cost_per_1m_tokens")
KEYS = ("prompt_sha256", "max_tokens", "model", "slots", "ctx")


def collect(target: str | Path) -> dict[str, dict]:
    path = Path(target)
    if path.is_file():
        return {path.stem: json.loads(path.read_text())}
    return {p.stem: json.loads(p.read_text()) for p in sorted(path.glob("*.json"))}


def delta(reference: float, yours: float) -> float:
    """Signed fraction; positive means yours is larger, so slower for latency."""
    return (yours - reference) / reference


def compare(
    reference: dict[str, dict], yours: dict[str, dict], tolerance: float = 0.25
) -> list[dict]:
    rows = []
    for stem in sorted(set(reference) | set(yours)):
        if stem not in reference or stem not in yours:
            reason = "only in reference" if stem in reference else "only in yours"
            rows.append(
                {"stem": stem, "status": "missing", "reason": reason, "metrics": {}}
            )
            continue
        ref, own = reference[stem], yours[stem]
        if "meta" not in ref or "meta" not in own:
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": "no meta",
                    "metrics": {},
                }
            )
            continue
        differing = next(
            (key for key in KEYS if ref["meta"].get(key) != own["meta"].get(key)), None
        )
        if differing:
            reason = (
                f"{differing} differs: {ref['meta'].get(differing)}"
                f" != {own['meta'].get(differing)}"
            )
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": reason,
                    "metrics": {},
                }
            )
            continue
        null = next(
            (
                metric
                for metric in GATED
                if ref.get(metric) is None or own.get(metric) is None
            ),
            None,
        )
        if null:
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": f"{null} is null",
                    "metrics": {},
                }
            )
            continue
        metrics = {}
        for metric in GATED + UNGATED:
            a, b = ref.get(metric), own.get(metric)
            metrics[metric] = {
                "reference": a,
                "yours": b,
                "delta": None if a is None or b is None else delta(a, b),
            }
        ok = all(abs(metrics[metric]["delta"]) <= tolerance for metric in GATED)
        rows.append(
            {
                "stem": stem,
                "status": "ok" if ok else "out",
                "reason": None,
                "metrics": metrics,
            }
        )
    return rows


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _table(
    title: str, names: tuple[str, ...], rows: list[dict], gated: bool
) -> list[str]:
    lines = [
        title,
        f"{'file':<20} {'metric':<24} {'reference':>12} {'yours':>12} {'delta %':>9}  band",
    ]
    for row in rows:
        for name in names:
            values = row["metrics"].get(name)
            if not values or (values["reference"] is None and values["yours"] is None):
                continue
            change = (
                "—" if values["delta"] is None else f"{values['delta'] * 100:+.1f}%"
            )
            band = row["status"] if gated else "—"
            lines.append(
                f"{row['stem']:<20} {name:<24} {_number(values['reference']):>12}"
                f" {_number(values['yours']):>12} {change:>9}  {band}"
            )
    return lines


def to_table(rows: list[dict]) -> str:
    lines = _table("GATED", GATED, rows, gated=True)
    lines += [""] + _table("UNGATED", UNGATED, rows, gated=False)
    skipped = [row for row in rows if row["status"] not in ("ok", "out")]
    if skipped:
        lines += ["", "NOT COMPARED"]
        lines += [
            f"{row['stem']:<20} {row['status']:<14} {row['reason']}" for row in skipped
        ]
    return "\n".join(lines)
