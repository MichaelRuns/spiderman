"""Regenerates the JS <-> Python parity fixture.

``web/nn/tests/parity.test.ts`` checks that the TS port of ``nn.py``
produces the same logits as this (real) Python ``TransformerLM``, given the
same weights and input. That fixture — ``web/nn/tests/fixtures/
transformer_lm.json`` — is generated here rather than by hand, so it can be
reproduced (or regenerated after a model config change) with:

    uv run python -m spiderman_llm.scripts.gen_js_parity_fixture

The seed, config, and token ids below are fixed on purpose: re-running this
script with no changes should reproduce the committed fixture byte-for-byte.
If you change ``TransformerLM``'s architecture, bump these as needed and
re-run to refresh the fixture (then re-run the JS parity test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from spiderman_llm.nn import TransformerLM

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "web" / "nn" / "tests" / "fixtures" / "transformer_lm.json"

SEED = 1234
CONFIG = dict(
    vocab_size=13,
    context_length=6,
    d_model=8,
    num_layers=2,
    num_heads=2,
    d_ff=16,
    theta=10000.0,
)
TOKEN_IDS = [[1, 4, 2, 7, 0]]  # [batch=1, seq=5]


def generate_fixture() -> dict:
    torch.manual_seed(SEED)
    model = TransformerLM(**CONFIG)
    model.eval()

    token_ids = torch.tensor(TOKEN_IDS, dtype=torch.long)
    with torch.no_grad():
        logits = model(token_ids)

    state_dict = {
        name: {"shape": list(tensor.shape), "data": tensor.detach().flatten().tolist()}
        for name, tensor in model.state_dict().items()
    }

    return {
        "config": CONFIG,
        "tokenIds": TOKEN_IDS,
        "stateDict": state_dict,
        "logits": {"shape": list(logits.shape), "data": logits.detach().flatten().tolist()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fixture = generate_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fixture, f)

    print(f"Wrote fixture to {args.output} (logits shape {fixture['logits']['shape']})")


if __name__ == "__main__":
    main()
