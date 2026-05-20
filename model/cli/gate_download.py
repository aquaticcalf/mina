"""
CLI entry point: download and organise the fish gate dataset.

Usage:
    uv run gate-download [--target-dir PATH] [--seed N]

Downloads two Kaggle datasets and builds the ImageFolder-compatible
gate_data/ directory structure.

Requires Kaggle credentials:
  - Either ~/.kaggle/kaggle.json
  - Or env vars KAGGLE_USERNAME and KAGGLE_KEY
"""

import argparse
import shutil
from pathlib import Path

from mina.core.constants import GATE_DATA_DIR
from mina.gate_dataset import build_gate_dataset, download_kaggle_fish, download_kaggle_no_fish


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and organise the fish gate classifier dataset from Kaggle"
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=GATE_DATA_DIR,
        help=f"Output directory for gate_data/ structure (default: {GATE_DATA_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible image sampling (default: 42)",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the raw downloaded archives. By default they are deleted after build.",
    )
    args = parser.parse_args()

    raw_dir = args.target_dir.parent / "gate_data_raw"

    fish_src = download_kaggle_fish(raw_dir / "fish")
    no_fish_src = download_kaggle_no_fish(raw_dir / "no_fish")

    build_gate_dataset(
        fish_source=fish_src,
        no_fish_source=no_fish_src,
        target_dir=args.target_dir,
        seed=args.seed,
    )

    if not args.keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"Cleaned up raw download directory: {raw_dir}")


if __name__ == "__main__":
    main()
