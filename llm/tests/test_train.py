import math

import numpy as np
import torch

from spiderman_llm.scripts.train import get_batch, lr_cosine_schedule, save_checkpoint, load_checkpoint
from spiderman_llm.nn import TransformerLM


def test_lr_cosine_schedule_warmup_and_bounds():
    max_lr, min_lr, warmup, cycle = 1e-3, 1e-4, 10, 100

    assert lr_cosine_schedule(0, max_lr, min_lr, warmup, cycle) == max_lr * (1 / warmup)
    assert math.isclose(lr_cosine_schedule(warmup, max_lr, min_lr, warmup, cycle), max_lr, rel_tol=1e-6)
    assert lr_cosine_schedule(cycle + 1, max_lr, min_lr, warmup, cycle) == min_lr


def test_get_batch_shapes_and_next_token_alignment():
    data = np.arange(1000, dtype=np.int64)
    x, y = get_batch(data, batch_size=4, context_length=8, device="cpu")

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_checkpoint_roundtrip(tmp_path):
    model = TransformerLM(
        vocab_size=50, context_length=16, d_model=32, num_layers=1, num_heads=2, d_ff=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    path = tmp_path / "ckpt.pt"
    save_checkpoint(model, optimizer, iteration=7, path=path)

    model2 = TransformerLM(
        vocab_size=50, context_length=16, d_model=32, num_layers=1, num_heads=2, d_ff=64
    )
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    loaded_iter = load_checkpoint(path, model2, optimizer2)

    assert loaded_iter == 7
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)


def test_training_step_reduces_synthetic_loss():
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=20, context_length=16, d_model=32, num_layers=1, num_heads=2, d_ff=64
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    data = np.tile(np.arange(20, dtype=np.int64), 50)

    from spiderman_llm.nn import cross_entropy

    losses = []
    for _ in range(20):
        x, y = get_batch(data, batch_size=8, context_length=16, device="cpu")
        logits = model(x)
        loss = cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
