from bench.prompts import load_long_prompts, load_prompts


def test_long_prompt_set_is_fixed():
    prompts = load_long_prompts()
    assert len(prompts) == 50
    assert all(10_000 <= len(p) <= 17_000 for p in prompts)
    assert load_prompts("data/prompts_long.jsonl") == prompts


def test_prompt_set_is_fixed():
    prompts = load_prompts()
    assert len(prompts) == 200
    assert all(150 <= len(p) <= 400 for p in prompts)
    assert len(set(prompts)) == 200


def test_a_changed_file_is_rejected(tmp_path):
    import json

    prompts = load_prompts()
    rows = [json.loads(line) for line in open("data/prompts.jsonl", encoding="utf-8")]
    assert [r["text"] for r in rows] == prompts
    rows[0]["text"] = rows[0]["text"][:-1] + "Q"
    tampered = tmp_path / "prompts.jsonl"
    tampered.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    try:
        load_prompts(tampered)
    except ValueError as exc:
        assert "changed" in str(exc)
    else:
        raise AssertionError("tampered prompt set was accepted")
