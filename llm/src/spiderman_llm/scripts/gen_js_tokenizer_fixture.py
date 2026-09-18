"""Regenerates the JS <-> Python tokenizer parity fixture.

``web/nn/tests/tokenizer.test.ts`` checks that the TS port of the BPE
tokenizer (encode/decode only, in ``web/nn/src/tokenizer.ts``) produces the
same token ids as this (real) Python ``Tokenizer``, using the actual trained
vocab/merges already exported to ``web/weights/``. Regenerate with:

    uv run python -m spiderman_llm.scripts.gen_js_tokenizer_fixture

Re-running with no changes should reproduce the committed fixture
byte-for-byte (deterministic: same vocab/merges in, same encodings out).
"""

from __future__ import annotations

import json
from pathlib import Path

from spiderman_llm.tokenizer import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
WEIGHTS_DIR = REPO_ROOT / "web" / "weights"
DEFAULT_OUTPUT = REPO_ROOT / "web" / "nn" / "tests" / "fixtures" / "tokenizer_parity.json"

SPECIAL_TOKENS = ["<|endoftext|>"]

# Deliberately varied: plain words, contractions, punctuation, multiple
# whitespace runs, a special token, and an empty string edge case.
SAMPLE_TEXTS = [
    "Once upon a time, there was a little girl named Sue.",
    "I don't think that's right -- it can't be!",
    "  leading and trailing whitespace   ",
    "line one\nline two\n\nline three",
    "The quick, brown fox jumps over 42 lazy dogs!!!",
    "<|endoftext|>Once upon a time<|endoftext|>",
    "",
]


def generate_fixture() -> dict:
    tokenizer = Tokenizer.from_files(
        WEIGHTS_DIR / "vocab.json", WEIGHTS_DIR / "merges.json", SPECIAL_TOKENS
    )
    cases = []
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    return {"specialTokens": SPECIAL_TOKENS, "cases": cases}


def main() -> None:
    fixture = generate_fixture()
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, ensure_ascii=False)
    print(f"Wrote fixture to {DEFAULT_OUTPUT} ({len(fixture['cases'])} cases)")


if __name__ == "__main__":
    main()
