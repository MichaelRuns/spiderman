# spiderman-llm

The Python half of the project: a from-scratch LLM implementation, migrated
piece by piece from a cs336-style assignment repo.

## Layout

- `src/spiderman_llm/nn.py` — model modules (RoPE, attention, transformer
  blocks, RMSNorm, SwiGLU, the BPE tokenizer, the optimizer, ...). Placeholder
  until the cs336 code is copied over.
- `src/spiderman_llm/scripts/` — training / data-processing entry points
  (e.g. `train.py`). Placeholder until the cs336 training loop is copied over.
- `tests/` — tests to confirm module health as pieces land.

## Usage

```sh
uv sync            # install deps into .venv
uv run pytest       # run the test suite
uv run python -m spiderman_llm.scripts.train   # run training (once implemented)
```
