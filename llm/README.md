# spiderman-llm

The Python half of the project: a from-scratch LLM implementation, migrated
piece by piece from a cs336-style assignment repo.

## Layout

- `src/spiderman_llm/nn.py` — model modules (RoPE, attention, transformer
  blocks, RMSNorm, SwiGLU, embeddings, cross-entropy, ...).
- `src/spiderman_llm/tokenizer.py` — byte-level BPE tokenizer: `train_bpe`
  learns a vocab/merge list from a text file, `Tokenizer` encodes/decodes
  (with special-token support and a memory-efficient `encode_iterable`).
- `src/spiderman_llm/scripts/train.py` — CLI training loop for
  `TransformerLM`: trains (or reuses) a tokenizer, tokenizes the corpus,
  and runs AdamW with a cosine LR schedule, grad clipping, periodic eval,
  and checkpointing.
- `tests/` — tests to confirm module health as pieces land.

## Usage

```sh
uv sync            # install deps into .venv
uv run pytest       # run the test suite

# train a small model, from a plain-text corpus
uv run python -m spiderman_llm.scripts.train \
  --input path/to/corpus.txt \
  --checkpoint-dir checkpoints \
  --vocab-size 10000
```
