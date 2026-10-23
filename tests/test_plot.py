from bench.plot import load_results, to_markdown, to_svg


def _rows():
    return [
        {
            "engine": "ollama",
            "concurrency": 1,
            "tokens_per_s": 40.0,
            "p50_ms": 6400,
            "p95_ms": 6500,
            "peak_vram_mb": 9000,
            "cost_per_1m_tokens": 2.43,
        },
        {
            "engine": "llamacpp",
            "concurrency": 1,
            "tokens_per_s": 55.0,
            "p50_ms": 4600,
            "p95_ms": 4700,
            "peak_vram_mb": 8800,
            "cost_per_1m_tokens": 1.77,
        },
        {
            "engine": "vllm",
            "concurrency": 32,
            "tokens_per_s": 900.0,
            "p50_ms": 9000,
            "p95_ms": 12000,
            "peak_vram_mb": 21000,
            "cost_per_1m_tokens": 0.11,
        },
    ]


def test_markdown_has_one_row_per_measurement():
    md = to_markdown(_rows())
    assert "| vllm | 32 | 900.0 |" in md
    assert md.count("\n| ") == 3


def test_svg_bars_scale_with_throughput(tmp_path):
    out = tmp_path / "chart.svg"
    to_svg(_rows(), out)
    svg = out.read_text()
    assert svg.count("<rect") >= 3 + 3  # bars + legend swatches
    heights = [float(v) for v in __import__("re").findall(r'height="([\d.]+)"', svg)]
    assert max(heights) > min(heights) * 5  # 900 tok/s towers over 40


def test_load_results_sorts_by_concurrency(tmp_path):
    a = tmp_path / "a.json"
    a.write_text('{"engine":"vllm","concurrency":32}')
    b = tmp_path / "b.json"
    b.write_text('{"engine":"ollama","concurrency":1}')
    rows = load_results([str(a), str(b)])
    assert [r["concurrency"] for r in rows] == [1, 32]


def test_svg_can_plot_a_latency_metric(tmp_path):
    out = tmp_path / "ttft.svg"
    to_svg(_rows(), out, "p50_ms", "ms", engines=("ollama", "llamacpp"))
    svg = out.read_text()
    assert ">ms<" in svg and "p50" not in svg
    assert (
        svg.count("<rect") == 1 + 2 + 2
    )  # background + two bars + two legend swatches
