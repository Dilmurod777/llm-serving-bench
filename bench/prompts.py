"""The fixed prompt sets: 200 short paragraphs and 50 ~4k-token prompts.

Both come from a public-domain US government manual, sha256-pinned so a change to
the inputs is a loud failure rather than a silent shift in the numbers.
"""

import hashlib
import json
from pathlib import Path

PROMPTS_PATH = Path("data/prompts.jsonl")
PROMPTS_SHA256 = "132cbe407888c957db48807b152ca9b1f95e1904abd7f3a81b415025f0aa0d9c"
EXPECTED_COUNT = 200

LONG_PATH = Path("data/prompts_long.jsonl")
LONG_SHA256 = "b9df55a88732874702285224d339def6c34cf0cb00866a3bad08100130cc00aa"
LONG_COUNT = 50


def prompt_sha(path: Path | str = PROMPTS_PATH) -> str:
    """Hash of the prompt file actually used, for the result's provenance block."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prompts(path: Path | str = PROMPTS_PATH) -> list[str]:
    path = Path(path)
    if path.name == LONG_PATH.name:
        return load_long_prompts(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PROMPTS_SHA256:
        raise ValueError(f"{path} changed (sha256 {digest}); regenerate the pinned set")
    prompts = _read(raw)
    if len(prompts) != EXPECTED_COUNT:
        raise ValueError(f"expected {EXPECTED_COUNT} prompts, found {len(prompts)}")
    if not all(isinstance(p, str) and 150 <= len(p) <= 400 for p in prompts):
        raise ValueError("every prompt must be 150-400 characters")
    return prompts


def load_long_prompts(path: Path | str = LONG_PATH) -> list[str]:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != LONG_SHA256:
        raise ValueError(f"{path} changed (sha256 {digest}); regenerate the pinned set")
    prompts = _read(raw)
    if len(prompts) != LONG_COUNT:
        raise ValueError(f"expected {LONG_COUNT} prompts, found {len(prompts)}")
    if not all(isinstance(p, str) and 10_000 <= len(p) <= 17_000 for p in prompts):
        raise ValueError("every long prompt must be 10,000-17,000 characters")
    return prompts


def _read(raw: bytes) -> list[str]:
    return [
        json.loads(line)["text"] for line in raw.decode().splitlines() if line.strip()
    ]
