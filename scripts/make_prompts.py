"""One-off generator for data/prompts.jsonl.

Usage: python scripts/make_prompts.py path/to/manual.txt

The manual must be a public-domain document; the checked-in lines are all a
stranger needs, so this script exists only to prove where they came from.
"""

import json
import random
import re
import sys
from pathlib import Path

TARGET = 200
SKIP = re.compile(r"TABLE|FIGURE|CONTINUED|Advisory Circular|AC No\.|^Page |^AC 43")


def passages(text: str) -> list[str]:
    blocks = [" ".join(b.split()) for b in text.split("\n\n")]
    blocks = [b for b in blocks if b and not SKIP.search(b)]
    out: list[str] = []
    buf = ""
    for block in blocks:
        buf = f"{buf} {block}".strip()
        if len(buf) < 150:
            continue
        if len(buf) > 400:
            cut = buf.rfind(". ", 150, 400)
            buf = buf[: cut + 1] if cut != -1 else buf[:390].rsplit(" ", 1)[0] + "."
        if 150 <= len(buf) <= 400:
            out.append(buf)
        buf = ""
    seen, unique = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main(source: str) -> None:
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    pool = passages(text)
    if len(pool) < TARGET:
        raise SystemExit(f"only {len(pool)} usable passages in {source}")
    picked = sorted(random.Random(0).sample(range(len(pool)), TARGET))
    out = Path("data/prompts.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, idx in enumerate(picked):
            fh.write(json.dumps({"id": i, "text": pool[idx]}, ensure_ascii=True) + "\n")
    print(f"wrote {TARGET} prompts from {len(pool)} candidates to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
