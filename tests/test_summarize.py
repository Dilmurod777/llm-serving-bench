from bench.summarize import summarize


def test_summarize_known_numbers():
    records = [
        {"latency_ms": ms, "completion_tokens": 100, "error": None}
        for ms in (1000, 2000, 3000, 4000)
    ]
    s = summarize(records, concurrency=1, duration_s=10.0, peak_vram_mb=20000)
    assert s["requests"] == 4 and s["output_tokens"] == 400
    assert s["tokens_per_s"] == 40.0
    assert s["p50_ms"] == 2000 and s["p95_ms"] == 4000
    assert round(s["cost_per_1m_tokens"], 2) == 2.43  # 0.35 / (40*3600) * 1e6


def test_errors_are_excluded_from_latency_but_counted():
    records = [
        {"latency_ms": 1000, "completion_tokens": 100, "error": None},
        {"latency_ms": 0, "completion_tokens": 0, "error": "timeout"},
    ]
    s = summarize(records, 1, 1.0, 1)
    assert s["requests"] == 2 and s["errors"] == 1
    assert s["p95_ms"] == 1000
    assert s["output_tokens"] == 100


def test_all_failed_run_has_no_cost():
    s = summarize(
        [{"latency_ms": 0, "completion_tokens": 0, "error": "boom"}], 1, 1.0, 0
    )
    assert (
        s["tokens_per_s"] == 0
        and s["cost_per_1m_tokens"] is None
        and s["p50_ms"] is None
    )


def test_energy_per_million_tokens():
    records = [
        {"latency_ms": 1000, "completion_tokens": 100, "error": None},
        {"latency_ms": 1000, "completion_tokens": 100, "error": None},
    ]
    s = summarize(records, 1, 10.0, 1000, power_w=300.0)
    # 200 tokens / 10 s = 20 tok/s; 300 W / 20 = 15 J/token
    assert s["avg_power_w"] == 300.0
    assert s["joules_per_1m_tokens"] == 15_000_000
