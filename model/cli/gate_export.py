"""
CLI entry point: export the trained gate model to ONNX and verify it.

Usage:
    uv run gate-export [--weights PATH] [--output PATH] [--opset N]

Exports fish_gate_best.pt → fish_gate.onnx and runs a CPU sanity check
using onnxruntime to confirm the graph loads and produces shape (1, 1).
"""

import argparse
from pathlib import Path

from mina.core.constants import GATE_RUNS_DIR
from mina.gate_export import export_gate_onnx, verify_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the fish gate classifier to ONNX format"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=GATE_RUNS_DIR / "fish_gate_best.pt",
        help="Path to trained .pt weights (default: runs/gate/fish_gate_best.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the .onnx file (default: same dir as weights)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, compatible with ORT-Web >= 1.17)",
    )
    args = parser.parse_args()

    onnx_path = export_gate_onnx(
        weights_path=args.weights,
        output_path=args.output,
        opset=args.opset,
    )
    verify_onnx(onnx_path)

    print(f"\nNext step: copy {onnx_path} to web/public/model/fish_gate.onnx")


if __name__ == "__main__":
    main()
