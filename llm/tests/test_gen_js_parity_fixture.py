from spiderman_llm.scripts.gen_js_parity_fixture import CONFIG, TOKEN_IDS, generate_fixture


def test_generate_fixture_is_well_formed():
    fixture = generate_fixture()

    assert fixture["config"] == CONFIG
    assert fixture["tokenIds"] == TOKEN_IDS

    batch, seq = len(TOKEN_IDS), len(TOKEN_IDS[0])
    assert fixture["logits"]["shape"] == [batch, seq, CONFIG["vocab_size"]]
    assert len(fixture["logits"]["data"]) == batch * seq * CONFIG["vocab_size"]

    # Every state_dict entry's flattened data length must match its declared shape.
    for name, entry in fixture["stateDict"].items():
        expected_size = 1
        for dim in entry["shape"]:
            expected_size *= dim
        assert len(entry["data"]) == expected_size, name


def test_generate_fixture_is_deterministic():
    first = generate_fixture()
    second = generate_fixture()
    assert first["logits"]["data"] == second["logits"]["data"]
