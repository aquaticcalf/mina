"""
Hyperparameter tuning using YOLOv8's built-in tune() method.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import (
    DEFAULT_TUNE_EPOCHS,
    DEFAULT_TUNE_ITERATIONS,
)
from mina.train import get_data_yaml_path


def tune_hyperparameters(
    data: str | None = None,
    epochs: int = DEFAULT_TUNE_EPOCHS,
    iterations: int = DEFAULT_TUNE_ITERATIONS,
    optimizer: str = "AdamW",
    device: str = "0",
) -> Path:
    """
    Tune YOLOv8 hyperparameters using Ray Tune.

    Args:
        data: Path to data.yaml (auto-detected if None)
        epochs: Training epochs per iteration
        iterations: Total tuning iterations
        optimizer: Optimizer type
        device: Device to use

    Returns:
        Path to best hyperparameters file
    """
    if data is None:
        data_yaml = get_data_yaml_path()
    else:
        data_yaml = Path(data)
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    print(f"Using dataset config: {data_yaml}")
    print("Starting hyperparameter tuning...")
    print(f"  Epochs per iteration: {epochs}")
    print(f"  Total iterations: {iterations}")
    print(f"  Optimizer: {optimizer}")

    model = YOLO("yolov8n.pt")

    model.tune(
        data=str(data_yaml),
        epochs=epochs,
        iterations=iterations,
        optimizer=optimizer,
        device=device,
        plots=True,
        save=True,
        val=True,
    )

    print("\nTuning complete!")
    print("Results saved to: runs/tune")

    return Path("runs/tune")
