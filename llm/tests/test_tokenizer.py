from spiderman_llm.tokenizer import Tokenizer, train_bpe


def test_train_bpe_merges_frequent_pairs(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("low low low lower lower widest widest widest newest newest newest newest")

    vocab, merges = train_bpe(corpus, vocab_size=256 + 6, special_tokens=[])

    assert len(vocab) == 256 + 6
    assert len(merges) == 6
    # The 256 raw bytes should still all be present as single-byte tokens.
    assert all(bytes([i]) in vocab.values() for i in range(256))


def test_train_bpe_is_deterministic(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the quick brown fox jumps over the lazy dog " * 20)

    vocab1, merges1 = train_bpe(corpus, vocab_size=300, special_tokens=[])
    vocab2, merges2 = train_bpe(corpus, vocab_size=300, special_tokens=[])

    assert merges1 == merges2
    assert vocab1 == vocab2


def test_encode_decode_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world! hello there, world of tokenizers.")

    vocab, merges = train_bpe(corpus, vocab_size=290, special_tokens=["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    text = "hello world! <|endoftext|> hello there"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_special_tokens_are_single_ids(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("some ordinary training text for the tokenizer to chew on.")

    vocab, merges = train_bpe(corpus, vocab_size=280, special_tokens=["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    ids = tokenizer.encode("<|endoftext|>")
    assert ids == [tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]]


def test_encode_iterable_matches_encode(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line one here\nline two here\nline three here\n")

    vocab, merges = train_bpe(corpus, vocab_size=280, special_tokens=[])
    tokenizer = Tokenizer(vocab, merges, special_tokens=[])

    lines = ["line one here\n", "line two here\n", "line three here\n"]
    streamed = list(tokenizer.encode_iterable(lines))
    whole = tokenizer.encode("".join(lines))
    assert streamed == whole


def test_save_and_load_roundtrip(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("saving and loading a tokenizer should be lossless.")

    vocab, merges = train_bpe(corpus, vocab_size=280, special_tokens=["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    tokenizer.save(vocab_path, merges_path)

    reloaded = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    text = "saving and loading <|endoftext|> a tokenizer"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == text
