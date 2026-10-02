"""Build a plain-text training corpus from a FineWeb-Edu parquet shard.

Downloads one shard of the FineWeb-Edu 10B sample
(``HuggingFaceFW/fineweb-edu``, ``sample/10BT``), extracts the ``text``
column, joins documents with the ``<|endoftext|>`` separator, and writes a
single ``corpus.txt`` that ``train.py`` consumes.

Needs ``pyarrow``, which is intentionally not a project dependency (only
needed for this one-off prep step)::

    uv run --with pyarrow python -m spiderman_llm.scripts.prepare_fineweb_corpus \\
        --output data/fineweb-edu-corpus.txt

``--max-chars`` caps the output size; the default (~200MB, ~50M tokens under
the project's BPE tokenizer) is plenty for the repo's toy model configs.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SHARD_URL = (
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/"
    "sample/10BT/000_00000.parquet"
)
SEPARATOR = "<|endoftext|>"


def download_shard(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Reusing existing shard at {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/fineweb-edu-corpus.txt"))
    parser.add_argument("--shard-url", default=SHARD_URL)
    parser.add_argument("--shard-path", type=Path, default=Path("data/fwedu_000.parquet"))
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    args = parser.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "pyarrow is required for corpus prep (not a project dependency). "
            "Re-run with: uv run --with pyarrow python -m spiderman_llm.scripts.prepare_fineweb_corpus"
        ) from e

    args.shard_path.parent.mkdir(parents=True, exist_ok=True)
    download_shard(args.shard_url, args.shard_path)

    print("Reading text column (streaming row groups)...")
    chars = 0
    docs = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for batch in pq.ParquetFile(args.shard_path).iter_batches(columns=["text"], batch_size=4096):
            done = False
            for text in batch.column("text").to_pylist():
                if not text or not text.strip():
                    continue
                chunk = text.strip() + "\n" + SEPARATOR + "\n"
                if chars + len(chunk) > args.max_chars:
                    done = True
                    break
                f.write(chunk)
                chars += len(chunk)
                docs += 1
            if done:
                break
    print(f"Wrote {docs} docs, {chars / 1e6:.1f}M chars to {args.output}")


if __name__ == "__main__":
    main()
