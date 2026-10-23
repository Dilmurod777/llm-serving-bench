"""One-off generator for data/prompts_long.jsonl (the 4k-token TTFT set).

Usage: python scripts/make_long_prompts.py path/to/manual.txt

Same public-domain source as data/prompts.jsonl, but the passages are joined into
~16,000-character prompts instead of being cut at 400. Sequential, not sampled, so
the result is reproducible from the same text.
"""

import json
import re
import sys
from pathlib import Path

CHARS = 16_000
TARGET = 50
SKIP = re.compile(r"TABLE|FIGURE|CONTINUED|Advisory Circular|AC No\.|^Page |^AC 43")


def long_prompts(text: str, target: int = TARGET) -> list[str]:
    blocks = [" ".join(b.split()) for b in text.split("\n\n")]
    blocks = [b for b in blocks if len(b) >= 80 and not SKIP.search(b)]
    out: list[str] = []
    buf = ""
    for block in blocks:
        if len(buf) + len(block) + 1 > CHARS and buf:
            cut = buf.rfind(". ", CHARS - 800, CHARS)
            out.append(buf[: cut + 1] if cut != -1 else buf)
            buf = ""
            if len(out) == target:
                break
        buf = f"{buf} {block}".strip()
    return out


def main(source: str) -> None:
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    prompts = long_prompts(text)
    if len(prompts) < TARGET:
        raise SystemExit(f"only {len(prompts)} long prompts in {source}, need {TARGET}")
    out = Path("data/prompts_long.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for i, p in enumerate(prompts):
            fh.write(json.dumps({"id": i, "text": p}, ensure_ascii=True) + "\n")
    lengths = [len(p) for p in prompts]
    print(
        f"wrote {len(prompts)} prompts to {out}: "
        f"{min(lengths)}-{max(lengths)} chars, mean {sum(lengths) // len(lengths)}"
    )


if __name__ == "__main__":
    main(sys.argv[1])
