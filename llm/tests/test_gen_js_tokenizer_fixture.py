from spiderman_llm.scripts.gen_js_tokenizer_fixture import SAMPLE_TEXTS, SPECIAL_TOKENS, generate_fixture


def test_generate_fixture_is_well_formed():
    fixture = generate_fixture()

    assert fixture["specialTokens"] == SPECIAL_TOKENS
    assert [case["text"] for case in fixture["cases"]] == SAMPLE_TEXTS

    for case in fixture["cases"]:
        assert isinstance(case["ids"], list)
        assert all(isinstance(i, int) for i in case["ids"])
        # decode(encode(text)) should round-trip for every sample text.
        assert case["decoded"] == case["text"]


def test_generate_fixture_is_deterministic():
    first = generate_fixture()
    second = generate_fixture()
    assert first == second
