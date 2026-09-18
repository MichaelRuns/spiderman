"""Exports a training checkpoint to the binary format ``web/`` consumes.

JSON-encoding float32 weights as text bloats their size roughly 2-3x (every
number becomes ~8-10 ASCII digits instead of 4 raw bytes) — a real cost for
something served over GitHub Pages. This instead writes:

- ``weights.bin`` — every parameter tensor's raw float32 bytes, concatenated
  in a fixed order (no padding needed: every tensor's byte length is itself
  a multiple of 4, so offsets stay 4-byte aligned throughout).
- ``manifest.json`` — the model config, the tokenizer's special tokens, and
  for each tensor its shape and ``(byteOffset, byteLength)`` slice into
  ``weights.bin``.

The JS side (``@spiderman/nn``'s ``stateDictFromBinary``) fetches both and
slices ``Float32Array`` views directly out of the downloaded ArrayBuffer —
no parsing, no copy beyond the fetch itself.

Usage::

    uv run python -m spiderman_llm.scripts.export_weights \\
        --checkpoint checkpoints/ckpt_final.pt \\
        --vocab-path checkpoints/vocab.json \\
        --merges-path checkpoints/merges.json \\
        --special-tokens "<|endoftext|>" \\
        --output-dir ../web/weights
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def export_weights(
    checkpoint_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    output_dir: Path,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(
            f"{checkpoint_path} has no saved config (checkpoints from before this export "
            "script existed don't carry one) — retrain, or pass --config-json."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, dict] = {}
    chunks: list[bytes] = []
    offset = 0
    # Buffers (RoPE's cos/sin tables) are deterministic functions of config
    # (dK, context_length, theta) — the JS side recomputes them at
    # construction, so skip shipping them.
    for name, tensor in state_dict.items():
        if "rope.cos_table" in name or "rope.sin_table" in name:
            continue
        data = tensor.detach().to(torch.float32).contiguous()
        raw = data.numpy().tobytes()
        tensors[name] = {"shape": list(data.shape), "byteOffset": offset, "byteLength": len(raw)}
        chunks.append(raw)
        offset += len(raw)

    weights_path = output_dir / "weights.bin"
    with open(weights_path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)

    with open(vocab_path, encoding="utf-8") as f:
        vocab_size = len(json.load(f))

    manifest = {
        "config": {
            "vocabSize": vocab_size,
            "contextLength": config["context_length"],
            "dModel": config["d_model"],
            "numLayers": config["num_layers"],
            "numHeads": config["num_heads"],
            "dFF": config["d_ff"],
            "ropeTheta": config["theta"],
        },
        "specialTokens": special_tokens,
        "tensors": tensors,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    # The tokenizer's vocab/merges are small (JSON text of a few thousand
    # short byte-arrays), so no need for a binary format there too.
    shutil.copy(vocab_path, output_dir / "vocab.json")
    shutil.copy(merges_path, output_dir / "merges.json")

    total_bytes = weights_path.stat().st_size
    print(
        f"Wrote {weights_path} ({total_bytes / 1e6:.1f}MB), {manifest_path}, "
        f"vocab.json, merges.json to {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_weights(args.checkpoint, args.vocab_path, args.merges_path, args.special_tokens, args.output_dir)


if __name__ == "__main__":
    main()
