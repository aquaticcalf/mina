"""
CLI for hyperparameter tuning using Ray Tune.

Usage:
    uv run mina-tune [--data PATH] [--epochs N] [--iterations N] [--optimizer NAME] [--device DEVICE]
"""

import argparse

from mina.tune import tune_hyperparameters
from mina.core.constants import DEFAULT_TUNE_EPOCHS, DEFAULT_TUNE_ITERATIONS


def main():
    parser = argparse.ArgumentParser(
        description="Tune YOLOv8 hyperparameters using Ray Tune"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (default: auto-detect)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TUNE_EPOCHS,
        help=f"Training epochs per iteration (default: {DEFAULT_TUNE_EPOCHS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_TUNE_ITERATIONS,
        help=f"Total tuning iterations (default: {DEFAULT_TUNE_ITERATIONS})",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam"],
        help="Optimizer to use (default: AdamW)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: 0)",
    )

    args = parser.parse_args()

    tune_hyperparameters(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer=args.optimizer,
        device=args.device,
    )


if __name__ == "__main__":
    main()
