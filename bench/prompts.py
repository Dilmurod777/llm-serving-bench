"""The fixed prompt set: 200 paragraphs of a public-domain US government manual."""

import hashlib
import json
from pathlib import Path

PROMPTS_PATH = Path("data/prompts.jsonl")
PROMPTS_SHA256 = "132cbe407888c957db48807b152ca9b1f95e1904abd7f3a81b415025f0aa0d9c"
EXPECTED_COUNT = 200


def load_prompts(path: Path | str = PROMPTS_PATH) -> list[str]:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROMPTS_SHA256:
        raise ValueError(f"{path} changed (sha256 {digest}); regenerate the pinned set")
    prompts = [
        json.loads(line)["text"] for line in raw.decode().splitlines() if line.strip()
    ]
    if len(prompts) != EXPECTED_COUNT:
        raise ValueError(f"expected {EXPECTED_COUNT} prompts, found {len(prompts)}")
    if not all(isinstance(p, str) and 150 <= len(p) <= 400 for p in prompts):
        raise ValueError("every prompt must be 150-400 characters")
    return prompts
