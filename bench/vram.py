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


def assert_idle(min_free_mb: int = 22000, reader=read_free_mb) -> None:
    """Refuse to measure while another model is loaded: two engines share the GPU."""
    free = reader()
    if free < min_free_mb:
        raise RuntimeError(f"GPU busy: {free} MiB free, need {min_free_mb}")


class VramSampler:
    def __init__(self, interval_s: float = 0.5, reader=read_used_mb):
        self.interval_s = interval_s
        self.reader = reader
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_mb = max(self.peak_mb, self.reader())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self.peak_mb = self.reader()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            self.peak_mb = max(self.peak_mb, self.reader())
        except Exception:
            pass
