# spiderman

A web-based LLM visualizer and module explorer: train a small LLM from
scratch, then serve it — and visualize it — entirely in the browser.

## Structure

The project has two halves:

```
spiderman/
├── llm/     # Python: from-scratch LLM implementation (migrated from cs336)
└── web/     # JS: browser LLM runtime, visualizations, and module playground
```

### `llm/` — model & training (Python, managed with [uv](https://docs.astral.sh/uv/))

The model code — RoPE, attention, transformer blocks, RMSNorm, SwiGLU, the
BPE tokenizer, the training loop — plus the scripts used to train it and
export its weights.

- `llm/src/spiderman_llm/nn.py` — model modules
- `llm/src/spiderman_llm/tokenizer.py` — BPE tokenizer (train + encode/decode)
- `llm/src/spiderman_llm/scripts/` — training (`train.py`), weight export
  (`export_weights.py`), and JS-parity fixture generators
- `llm/tests/` — tests to confirm module health (`uv run pytest`)

### `web/` — browser runtime & visualizations (JS, see `web/README.md`)

A pnpm workspace, served via GitHub Pages (auto-deployed on push to `main`).

- `web/nn/` — a from-scratch TS port of `nn.py`'s forward pass (inference
  only — no autograd), plus a real KV cache, tokenizer encode/decode, and
  sampling, so the browser can run genuine autoregressive generation.
- `web/visualizer/` — a React + Three Fiber app: prompt in, real generation
  runs in-browser, every layer's activations/attention/KV-cache are
  visualized in 3D from actual captured forward-pass state, with live perf
  stats and a playback-speed control to slow inference to a human-readable
  pace.
- `web/playground/` — the "module playground": a graphical, explained walk
  through how each model module works, independent of running the full model.
  Not yet implemented.
- `web/weights/` — the trained model's exported weights, tracked in git.

## Status

`llm/` and `web/nn/` and `web/visualizer/` are implemented and tested (see
each package's tests: `uv run pytest` in `llm/`, `pnpm test` in `web/nn/`
and `web/visualizer/`). A small model is trained and its weights exported
to `web/weights/`. `web/playground/` is not yet implemented.
