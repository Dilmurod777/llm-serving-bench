from bench.vram import assert_idle, parse_free_mb


def test_parse_free_mb():
    assert parse_free_mb("12000\n") == 12000
    assert parse_free_mb(" 11338 ") == 11338


def test_assert_idle_raises_when_a_model_is_loaded():
    try:
        assert_idle(22000, reader=lambda: 11338)
    except RuntimeError as exc:
        assert "GPU busy: 11338 MiB free, need 22000" in str(exc)
    else:
        raise AssertionError("busy GPU was accepted")


def test_assert_idle_passes_on_empty_gpu():
    assert_idle(22000, reader=lambda: 23500)
