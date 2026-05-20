"""
CLI entry point: train the fish gate classifier.

Usage:
    uv run gate-train [--data-dir PATH] [--output-dir PATH]
                      [--phase1-epochs N] [--phase2-epochs N]
                      [--batch N] [--device DEVICE]

Runs two-phase transfer learning on MobileNetV3-Small.
Best weights are saved to runs/gate/fish_gate_best.pt.
"""

import argparse
from pathlib import Path

from mina.core.constants import GATE_DATA_DIR, GATE_RUNS_DIR
from mina.gate_train import train_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the MobileNetV3-Small fish gate classifier"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GATE_DATA_DIR,
        help=f"Root of gate_data/ (default: {GATE_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GATE_RUNS_DIR,
        help=f"Directory for saved checkpoints (default: {GATE_RUNS_DIR})",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=5,
        help="Epochs for head-only training, Phase 1 (default: 5)",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=15,
        help="Epochs for backbone fine-tuning, Phase 2 (default: 15)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on: 'cuda', 'cpu', or None for auto-detect (default: auto)",
    )
    args = parser.parse_args()

    train_gate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
