"""Tokenize a plain-text corpus to a ``.npy`` of int64 token ids.

``train.py``'s in-memory ``list(tokenizer.encode_iterable(f))`` materializes
every token id as a Python int before converting to numpy — fine for small
corpora, but a 200MB corpus becomes ~55M Python ints (~2GB) and can OOM a
small box. This instead encodes in streaming text chunks, accumulates the
ids as numpy arrays, and writes one ``.npy`` file; ``train.py`` then loads it
with ``mmap_mode="r"`` via ``--tokenized-input`` without ever holding the
full id list in RAM.

Usage::

    uv run python -m spiderman_llm.scripts.tokenize_to_npy \\
        --input data/fineweb-edu-corpus.txt \\
        --vocab-path checkpoints-fineweb/vocab.json \\
        --merges-path checkpoints-fineweb/merges.json \\
        --output data/fineweb-edu-tokens.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spiderman_llm.tokenizer import Tokenizer

# Must match prepare_fineweb_corpus.py's document separator: chunk boundaries
# are snapped to it so a special token never straddles two chunks.
SEPARATOR = "\n<|endoftext|>\n"


def chunked_texts(path: Path, chunk_chars: int):
    """Yield text chunks, split only at document separators."""
    with open(path, encoding="utf-8") as f:
        carry = ""
        while True:
            text = f.read(chunk_chars)
            if not text:
                if carry:
                    yield carry
                return
            text = carry + text
            idx = text.rfind(SEPARATOR)
            if idx == -1:
                carry = text  # no separator in this window; accumulate
                continue
            cut = idx + len(SEPARATOR)
            yield text[:cut]
            carry = text[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--chunk-chars", type=int, default=1_000_000)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)

    chunks: list[np.ndarray] = []
    total = 0
    for text in chunked_texts(args.input, args.chunk_chars):
        ids = np.fromiter(tokenizer.encode_iterable([text]), dtype=np.int64, count=-1)
        chunks.append(ids)
        total += len(ids)
        print(f"  ... {total / 1e6:.1f}M tokens", end="\r", flush=True)
    print()

    data = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, data)
    print(f"Wrote {len(data)} tokens ({len(data) / 1e6:.1f}M) to {args.output}")


if __name__ == "__main__":
    main()
