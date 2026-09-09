# web

The JS half of the project, eventually served via GitHub Pages. Two
sections:

- `visualizer/` — a from-scratch LLM inference implementation in JS that
  loads the trained weights (from `weights/`) and runs them, wired up with
  visualizations of what's happening at each layer (activations, attention,
  etc.). Includes a "time" slider to slow inference down to a human-readable
  pace.
- `playground/` — the "module playground": a standalone, graphical
  explainer for how each model module works (RoPE, attention, BPE, ...),
  independent of running the full model.
- `weights/` — exported weights from `llm/` training runs, served to the
  visualizer at runtime.

Currently an empty skeleton — no implementation yet.
