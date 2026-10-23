import subprocess
import threading
import time


def parse_free_mb(output: str) -> int:
    return int(output.strip().splitlines()[0].strip())


def read_free_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return parse_free_mb(out)


def read_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return parse_free_mb(out)


def assert_idle(min_free_mb: int = 14000, reader=read_free_mb) -> None:
    """Refuse to measure while another model is loaded: two engines share the GPU.

    The Windows desktop holds 7-11 GB of the card depending on what is open, so
    14 GB free means no model of ours is resident. Below that, something else
    (an Ollama model, a half-dead server) is still holding memory.
    """
    free = reader()
    if free < min_free_mb:
        raise RuntimeError(f"GPU busy: {free} MiB free, need {min_free_mb}")


class VramSampler:
    """Absolute peak VRAM while the sweep runs.

    Measured absolutely, not above a baseline: the engine stays loaded between
    sweeps at the same concurrency, so a baseline taken after warmup would
    report ~0 for every sweep after the first. The README states that the
    Windows desktop holds several GB of the same card, so the figure is the
    engine's footprint plus the desktop's.
    """

    def __init__(self, interval_s: float = 0.5, reader=read_used_mb):
        self.interval_s = interval_s
        self.reader = reader
        self._peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_mb(self) -> int:
        return self._peak

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self.reader())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._peak = self.reader()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self._peak = max(self._peak, self.reader())
        except Exception:
            pass
