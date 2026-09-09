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

The model code — RoPE, attention, transformer blocks, the BPE tokenizer,
the optimizer, etc. — plus the scripts used to train it. This will be
populated by migrating over an existing cs336-based implementation.

- `llm/src/spiderman_llm/nn.py` — model modules
- `llm/src/spiderman_llm/scripts/` — training scripts
- `llm/tests/` — tests to confirm module health (`uv run pytest`)

### `web/` — browser runtime & visualizations (JS)

Served via GitHub Pages once it exists. Two sections:

- `web/visualizer/` — a JS implementation of the trained model that loads
  and runs its weights in-browser, with visualizations of activations and
  layer-by-layer behavior, plus a "time" slider to slow inference down to a
  human-readable pace.
- `web/playground/` — the "module playground": a graphical, explained walk
  through how each model module works, independent of running the full model.

## Status

Initial skeleton only — directory structure and `uv` project setup for
`llm/`. No model or web implementation yet.
