"""Byte-level BPE tokenizer, migrated from cs336.

Two pieces:

- ``train_bpe`` learns a byte-level BPE vocabulary and merge list from a text
  file, the same algorithm GPT-2 uses: start from the 256 individual bytes,
  then repeatedly merge the most frequent adjacent pair of tokens.
- ``Tokenizer`` applies a trained vocabulary/merge list to encode text to
  token ids and decode ids back to text, with support for user-defined
  special tokens (e.g. ``<|endoftext|>``).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

# GPT-2's pretokenization pattern: keeps contractions, runs of letters, runs
# of digits, runs of punctuation, and whitespace as separate chunks so BPE
# merges never cross e.g. a word/space boundary.
PRETOKENIZE_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge_pretoken(
    pretoken: tuple[bytes, ...], pair: tuple[bytes, bytes], merged: bytes
) -> tuple[bytes, ...]:
    result = []
    i = 0
    while i < len(pretoken):
        if i < len(pretoken) - 1 and pretoken[i] == pair[0] and pretoken[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(pretoken[i])
            i += 1
    return tuple(result)


def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    if special_tokens:
        split_pattern = "|".join(regex.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True))
        chunks = regex.split(f"({split_pattern})", text)
    else:
        chunks = [text]

    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in chunks:
        if chunk in special_tokens:
            continue
        for match in PRETOKENIZE_PATTERN.finditer(chunk):
            token_bytes = match.group().encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on the text file at ``input_path``.

    Returns ``(vocab, merges)`` where ``vocab`` maps token id -> raw bytes,
    and ``merges`` is the ordered list of byte-pair merges that produced the
    vocabulary beyond the initial 256 byte tokens and special tokens.
    """
    special_tokens = special_tokens or []
    text = Path(input_path).read_text(encoding="utf-8")

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    pretokens = list(_pretoken_counts(text, special_tokens).items())

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_indices: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for idx, (pretoken, count) in enumerate(pretokens):
        for pair in zip(pretoken, pretoken[1:]):
            pair_counts[pair] += count
            pair_to_indices[pair].add(idx)

    merges: list[tuple[bytes, bytes]] = []
    num_merges = max(0, vocab_size - len(vocab))
    for _ in range(num_merges):
        if not pair_counts:
            break
        # Ties broken by the lexicographically greater pair, so merges are
        # deterministic regardless of dict/hash iteration order.
        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        merges.append(best_pair)
        next_id += 1

        for idx in list(pair_to_indices.get(best_pair, ())):
            pretoken, count = pretokens[idx]
            for pair in zip(pretoken, pretoken[1:]):
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_indices[pair].discard(idx)

            new_pretoken = _merge_pretoken(pretoken, best_pair, merged)
            pretokens[idx] = (new_pretoken, count)

            for pair in zip(new_pretoken, new_pretoken[1:]):
                pair_counts[pair] += count
                pair_to_indices[pair].add(idx)

    return vocab, merges


class Tokenizer:
    """Encodes text to token ids / decodes ids to text using a trained BPE."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_tokens = list(special_tokens or [])

        next_id = (max(self.vocab) + 1) if self.vocab else 0
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[next_id] = token_bytes
                next_id += 1

        self.token_to_id = {token_bytes: idx for idx, token_bytes in self.vocab.items()}

        self._special_pattern = None
        if self.special_tokens:
            pattern = "|".join(regex.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True))
            self._special_pattern = regex.compile(f"({pattern})")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(idx): bytes(byte_values) for idx, byte_values in raw_vocab.items()}

        with open(merges_filepath, encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = [(bytes(a), bytes(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    def save(self, vocab_filepath: str | Path, merges_filepath: str | Path) -> None:
        raw_vocab = {str(idx): list(token_bytes) for idx, token_bytes in self.vocab.items()}
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_vocab, f)

        raw_merges = [[list(a), list(b)] for a, b in self.merges]
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump(raw_merges, f)

    def _split_on_specials(self, text: str) -> Iterator[tuple[str, bool]]:
        if self._special_pattern is None:
            yield text, False
            return
        for chunk in self._special_pattern.split(text):
            if not chunk:
                continue
            yield chunk, chunk in self.special_tokens

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        parts: list[bytes] = [bytes([b]) for b in pretoken.encode("utf-8")]
        while len(parts) > 1:
            pairs = list(zip(parts, parts[1:]))
            ranked = [(self.merge_ranks[pair], i) for i, pair in enumerate(pairs) if pair in self.merge_ranks]
            if not ranked:
                break
            _, i = min(ranked)
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2 :]
        return [self.token_to_id[part] for part in parts]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk, is_special in self._split_on_specials(text):
            if is_special:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue
            for match in PRETOKENIZE_PATTERN.finditer(chunk):
                ids.extend(self._encode_pretoken(match.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode e.g. the lines of a file without materializing it all."""
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        return token_bytes.decode("utf-8", errors="replace")
