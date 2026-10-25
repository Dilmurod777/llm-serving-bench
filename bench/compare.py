"""Compare two sets of result JSONs offline: no HTTP, no GPU."""

import argparse
import json
from pathlib import Path

GATED = ("tokens_per_s", "per_stream_tps", "p50_ms", "p95_ms")
UNGATED = ("peak_vram_mb", "avg_power_w", "joules_per_1m_tokens", "cost_per_1m_tokens")
KEYS = ("prompt_sha256", "max_tokens", "model", "slots", "ctx")


def collect(target: str | Path) -> dict[str, dict]:
    path = Path(target)
    if not path.exists():
        raise SystemExit(f"no such path: {path}")
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    out = {}
    for file in files:
        try:
            out[file.stem] = json.loads(file.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{file}: {exc}") from exc
    return out


def delta(reference: float, yours: float) -> float | None:
    """Signed fraction; positive means yours is larger, so slower for latency."""
    if reference == 0:
        return None
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
        ref_meta, own_meta = ref.get("meta"), own.get("meta")
        if not isinstance(ref_meta, dict) or not isinstance(own_meta, dict):
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": "no meta",
                    "metrics": {},
                }
            )
            continue
        missing = next(
            (key for key in KEYS if key not in ref_meta or key not in own_meta), None
        )
        if missing:
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": f"no {missing} in meta",
                    "metrics": {},
                }
            )
            continue
        differing = next((key for key in KEYS if ref_meta[key] != own_meta[key]), None)
        if differing:
            reason = (
                f"{differing} differs: {ref_meta[differing]} != {own_meta[differing]}"
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
        zero = next((metric for metric in GATED if ref.get(metric) == 0), None)
        if zero:
            rows.append(
                {
                    "stem": stem,
                    "status": "incomparable",
                    "reason": f"{zero} has zero reference",
                    "metrics": {},
                }
            )
            continue
        metrics = {}
        for metric in GATED + UNGATED:
            a, b = ref.get(metric), own.get(metric)
            change = None if a is None or b is None else delta(a, b)
            band = "—"
            if metric in GATED and change is not None:
                band = "ok" if abs(change) <= tolerance else "out"
            metrics[metric] = {
                "reference": a,
                "yours": b,
                "delta": change,
                "band": band,
            }
        ok = all(metrics[metric]["band"] == "ok" for metric in GATED)
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


def _table(title: str, names: tuple[str, ...], rows: list[dict]) -> list[str]:
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
            band = values.get("band", "—")
            lines.append(
                f"{row['stem']:<20} {name:<24} {_number(values['reference']):>12}"
                f" {_number(values['yours']):>12} {change:>9}  {band}"
            )
    return lines


def to_table(rows: list[dict]) -> str:
    lines = _table("GATED", GATED, rows)
    lines += [""] + _table("UNGATED", UNGATED, rows)
    skipped = [row for row in rows if row["status"] not in ("ok", "out")]
    if skipped:
        lines += ["", "NOT COMPARED"]
        lines += [
            f"{row['stem']:<20} {row['status']:<14} {row['reason']}" for row in skipped
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare", description="Compare two sets of bench results offline."
    )
    parser.add_argument("reference", help="committed results (file or directory)")
    parser.add_argument("yours", help="your results (file or directory)")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="fraction, default 0.25 (=25%%)",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    if args.tolerance <= 0:
        parser.error("--tolerance must be > 0")

    rows = compare(collect(args.reference), collect(args.yours), args.tolerance)
    print(to_table(rows))
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("ok", "out", "incomparable", "missing")
    }
    print(
        f"{counts['ok']} ok, {counts['out']} out, "
        f"{counts['incomparable']} incomparable, {counts['missing']} missing"
    )
    return 1 if args.strict and any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
