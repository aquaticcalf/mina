"""
CLI entry point: evaluate the exported gate ONNX model on the validation set.

Usage:
    uv run gate-evaluate [--onnx PATH] [--data-dir PATH] [--threshold F]

Prints accuracy, precision, recall, and F1 for the "fish" class, a confusion
matrix, and an acceptance criteria pass/fail check against the Phase 1 targets.
"""

import argparse
from pathlib import Path

from mina.core.constants import GATE_DATA_DIR, GATE_RUNS_DIR, GATE_THRESHOLD
from mina.gate_evaluate import evaluate_gate_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fish gate ONNX model on the validation split"
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=GATE_RUNS_DIR / "fish_gate.onnx",
        help="Path to fish_gate.onnx (default: runs/gate/fish_gate.onnx)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GATE_DATA_DIR,
        help=f"Root of gate_data/ (default: {GATE_DATA_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=GATE_THRESHOLD,
        help=f"Sigmoid probability threshold for 'fish' prediction (default: {GATE_THRESHOLD})",
    )
    args = parser.parse_args()

    evaluate_gate_onnx(
        onnx_path=args.onnx,
        data_dir=args.data_dir,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
