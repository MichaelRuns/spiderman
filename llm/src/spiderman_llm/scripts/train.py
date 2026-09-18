"""Training entry point.

Trains a ``TransformerLM`` (see ``spiderman_llm.nn``) on a plain-text
corpus with next-token prediction. If no tokenizer is supplied, one is
trained from scratch on the input file and saved alongside the checkpoints.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from spiderman_llm.nn import TransformerLM, cross_entropy
from spiderman_llm.tokenizer import Tokenizer, train_bpe


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_or_train_tokenizer(args: argparse.Namespace) -> Tokenizer:
    if args.vocab_path and args.merges_path:
        return Tokenizer.from_files(args.vocab_path, args.merges_path, args.special_tokens)

    print(f"Training a new BPE tokenizer (vocab_size={args.vocab_size}) on {args.input}...")
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens)
    tokenizer = Tokenizer(vocab, merges, args.special_tokens)

    vocab_path = args.checkpoint_dir / "vocab.json"
    merges_path = args.checkpoint_dir / "merges.json"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(vocab_path, merges_path)
    print(f"Saved tokenizer to {vocab_path} and {merges_path}")
    return tokenizer


def tokenize_corpus(tokenizer: Tokenizer, input_path: Path) -> np.ndarray:
    with open(input_path, encoding="utf-8") as f:
        ids = list(tokenizer.encode_iterable(f))
    return np.array(ids, dtype=np.int64)


def get_batch(
    data: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    x = np.stack([data[s : s + context_length] for s in starts])
    y = np.stack([data[s + 1 : s + 1 + context_length] for s in starts])
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


def lr_cosine_schedule(
    it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def save_checkpoint(model, optimizer, iteration: int, config: dict, path: Path) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer) -> int:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


@torch.no_grad()
def estimate_val_loss(model, val_data: np.ndarray, args: argparse.Namespace, device: str) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        x, y = get_batch(val_data, args.batch_size, args.context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits, y).item())
    model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to a plain-text training corpus.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume-from", type=Path, default=None)

    tokenizer_group = parser.add_argument_group("tokenizer")
    tokenizer_group.add_argument("--vocab-path", type=Path, default=None, help="Reuse an existing tokenizer vocab.")
    tokenizer_group.add_argument("--merges-path", type=Path, default=None, help="Reuse an existing tokenizer merges.")
    tokenizer_group.add_argument("--vocab-size", type=int, default=10000)
    tokenizer_group.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--context-length", type=int, default=256)
    model_group.add_argument("--d-model", type=int, default=512)
    model_group.add_argument("--num-layers", type=int, default=4)
    model_group.add_argument("--num-heads", type=int, default=8)
    model_group.add_argument("--d-ff", type=int, default=1344)
    model_group.add_argument("--rope-theta", type=float, default=10000.0)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--batch-size", type=int, default=32)
    train_group.add_argument("--max-iters", type=int, default=5000)
    train_group.add_argument("--max-lr", type=float, default=3e-4)
    train_group.add_argument("--min-lr", type=float, default=3e-5)
    train_group.add_argument("--warmup-iters", type=int, default=200)
    train_group.add_argument("--cosine-cycle-iters", type=int, default=5000)
    train_group.add_argument("--weight-decay", type=float, default=0.01)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--val-fraction", type=float, default=0.05)
    train_group.add_argument("--eval-every", type=int, default=200)
    train_group.add_argument("--eval-iters", type=int, default=20)
    train_group.add_argument("--log-every", type=int, default=10)
    train_group.add_argument("--checkpoint-every", type=int, default=500)
    train_group.add_argument("--device", type=str, default=None)
    train_group.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or default_device()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_or_train_tokenizer(args)
    data = tokenize_corpus(tokenizer, args.input)
    if len(data) < args.context_length + 1:
        raise ValueError(
            f"Corpus has only {len(data)} tokens, need at least {args.context_length + 1}."
        )

    split = max(1, int(len(data) * (1 - args.val_fraction)))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) <= args.context_length:
        train_data, val_data = data, data

    vocab_size = max(tokenizer.vocab) + 1
    model_config = {
        "vocab_size": vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "theta": args.rope_theta,
    }
    model = TransformerLM(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer) + 1
        print(f"Resumed from {args.resume_from} at iteration {start_iter}")

    print(f"Training on device={device}, {len(train_data)} train tokens, {len(val_data)} val tokens")
    model.train()
    start_time = time.time()
    for it in range(start_iter, args.max_iters):
        lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_every == 0:
            elapsed = time.time() - start_time
            print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s")

        if it % args.eval_every == 0 and it > start_iter:
            val_loss = estimate_val_loss(model, val_data, args, device)
            print(f"iter {it:6d} | val_loss {val_loss:.4f}")

        if it % args.checkpoint_every == 0 and it > start_iter:
            save_checkpoint(model, optimizer, it, model_config, args.checkpoint_dir / f"ckpt_{it}.pt")

    save_checkpoint(model, optimizer, args.max_iters - 1, model_config, args.checkpoint_dir / "ckpt_final.pt")
    print(f"Saved final checkpoint to {args.checkpoint_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()
