import asyncio

from bench.engines import Completion
from bench.runner import run_sweep


class FakeEngine:
    name = "fake"

    def __init__(self, delay=0.05, fail_on=frozenset()):
        self.delay = delay
        self.fail_on = set(fail_on)
        self.in_flight = 0
        self.max_in_flight = 0

    async def complete(self, prompt):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            if int(prompt[1:]) in self.fail_on:
                raise RuntimeError("boom")
            return Completion(
                completion_tokens=10, latency_ms=self.delay * 1000, text="x"
            )
        finally:
            self.in_flight -= 1


def test_concurrency_is_respected_and_errors_are_records():
    fake = FakeEngine(delay=0.05, fail_on={3})
    records, duration = asyncio.run(
        run_sweep(fake, [f"p{i}" for i in range(10)], concurrency=5)
    )
    assert len(records) == 10
    assert fake.max_in_flight <= 5
    assert [r["prompt_id"] for r in records if r["error"]] == [3]
    assert duration < 0.3  # 10 prompts over 5 lanes at 50 ms, not 500 ms
    assert records[0]["completion_tokens"] == 10


def test_single_lane_is_serial():
    fake = FakeEngine(delay=0.01)
    records, duration = asyncio.run(run_sweep(fake, ["p0", "p1", "p2"], concurrency=1))
    assert fake.max_in_flight == 1 and duration >= 0.03
