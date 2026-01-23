"""
CLI for training the fish disease detection model.

Usage:
    uv run mina-train [--epochs N] [--batch N] [--imgsz N] [--name NAME] [--device DEVICE] [--hyp PATH]
"""

import argparse

from mina.train import train
from mina.core.constants import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_IMAGE_SIZE


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n model on fish disease dataset"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Input image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="fish_disease",
        help="Training run name (default: fish_disease)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: auto-detect)",
    )
    parser.add_argument(
        "--hyp",
        type=str,
        default=None,
        help="Path to hyperparameters YAML file",
    )

    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        device=args.device,
        hyp=args.hyp,
    )


if __name__ == "__main__":
    main()
