"""Sample text from a trained checkpoint with a fixed seed (for PR comparisons).

Usage:
    uv run python -m spiderman_llm.scripts.sample --checkpoint <ckpt.pt> \
        --vocab-path <vocab.json> --merges-path <merges.json> \
        --prompt "The scientist discovered that" --max-new-tokens 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spiderman_llm.nn import TransformerLM
from spiderman_llm.tokenizer import Tokenizer

PROMPTS = [
    "The scientist discovered that",
    "Once upon a time",
    "In order to build a better",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, default=None, help="Training checkpoint (.pt).")
    src.add_argument("--weights-dir", type=Path, default=None, help="Exported web/weights dir.")
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--merges-path", type=Path, default=None)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_from_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    return config, state


def load_from_weights_dir(path: Path) -> tuple[dict, dict]:
    manifest = json.loads((path / "manifest.json").read_text())
    cfg = manifest["config"]
    config = {
        "vocab_size": cfg["vocabSize"],
        "context_length": cfg["contextLength"],
        "d_model": cfg["dModel"],
        "num_layers": cfg["numLayers"],
        "num_heads": cfg["numHeads"],
        "d_ff": cfg["dFF"],
        "theta": cfg["ropeTheta"],
    }
    raw = (path / "weights.bin").read_bytes()
    state: dict[str, torch.Tensor] = {}
    for name, meta in manifest["tensors"].items():
        arr = np.frombuffer(raw, dtype=np.float32, count=int(np.prod(meta["shape"])), offset=meta["byteOffset"])
        state[name] = torch.from_numpy(arr.copy()).reshape(meta["shape"])
    return config, state


@torch.no_grad()
def generate(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.rope.max_seq_len :]
        logits = model(x_cond)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1).item())
        x = torch.cat([x, torch.tensor([[next_id]])], dim=1)
        if tokenizer.decode([next_id]) == "<|endoftext|>":
            break
    return tokenizer.decode(x[0].tolist())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.checkpoint is not None:
        config, state = load_from_checkpoint(args.checkpoint)
        vocab_path = args.vocab_path
        merges_path = args.merges_path
    else:
        assert args.weights_dir is not None
        config, state = load_from_weights_dir(args.weights_dir)
        vocab_path = args.vocab_path or args.weights_dir / "vocab.json"
        merges_path = args.merges_path or args.weights_dir / "merges.json"
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config.get("theta", 10000.0),
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, args.special_tokens)

    prompts = [args.prompt] if args.prompt else PROMPTS
    for prompt in prompts:
        text = generate(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
        print(f">>> {prompt!r}\n{text}\n")


if __name__ == "__main__":
    main()
