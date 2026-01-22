"""
Training logic for fish disease detection model.
"""

from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
)


def get_data_yaml_path(data_dir: Path | None = None) -> Path:
    """
    Get the path to the data.yaml configuration file.

    Args:
        data_dir: Optional data directory. If None, searches common locations.

    Returns:
        Path to data.yaml

    Raises:
        FileNotFoundError: If data.yaml not found
    """
    if data_dir is not None:
        yaml_path = data_dir / "data.yaml"
        if yaml_path.exists():
            return yaml_path
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

    # Look for dataset in common locations
    model_dir = Path(__file__).parent.parent
    possible_paths = [
        DATA_DIR / "data.yaml",
        model_dir / "data" / "data.yaml",
        model_dir / "Mina-2" / "data.yaml",
        model_dir / "mina-2" / "data.yaml",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"data.yaml not found. Looked in: {[str(p) for p in possible_paths]}\n"
        "Please run the dataset download script first: uv run mina-download"
    )


def get_device() -> str:
    """Get the best available device for training."""
    import torch

    if torch.cuda.is_available():
        return "0"  # Use first GPU
    return "cpu"


def train(
    epochs: int = DEFAULT_EPOCHS,
    batch: int = DEFAULT_BATCH_SIZE,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    name: str = "fish_disease",
    pretrained: str = "yolov8n.pt",
    data_dir: Path | None = None,
    patience: int = DEFAULT_PATIENCE,
    device: str | None = None,
    hyp: str | None = None,
) -> Path:
    """
    Train YOLOv8n model on fish disease dataset.

    Args:
        epochs: Number of training epochs
        batch: Batch size for training
        imgsz: Input image size (square)
        name: Name for the training run
        pretrained: Pretrained model to start from
        data_dir: Optional data directory
        patience: Early stopping patience
        device: Device to train on ('0' for GPU, 'cpu' for CPU, None for auto-detect)
        hyp: Path to hyperparameters YAML file (e.g., from tuning). If provided,
             uses these hyperparameters via Ultralytics cfg= parameter.
             If None, uses hardcoded defaults.

    Returns:
        Path to the best model weights
    """
    data_yaml = get_data_yaml_path(data_dir)
    print(f"Using dataset config: {data_yaml}")

    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    # Load YOLOv8 nano model (optimized for mobile)
    model = YOLO(pretrained)

    # Base training arguments (always used)
    base_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "name": name,
        "save": True,
        "save_period": -1,  # Only save best and last
        "patience": patience,
        "workers": 4,
        "device": device,
    }

    # Train with tuned hyperparameters or hardcoded defaults
    if hyp is not None:
        hyp_path = Path(hyp)
        if not hyp_path.exists():
            raise FileNotFoundError(
                f"Hyperparameters file not found: {hyp_path}\n"
                "Run tuning first: uv run mina-tune"
            )
        print(f"Using tuned hyperparameters from: {hyp_path}")
        # Use Ultralytics cfg= parameter for safe YAML loading with validation
        results = model.train(cfg=str(hyp_path), **base_args)
    else:
        print("Using hardcoded default hyperparameters")
        # Augmentation settings for better generalization (fallback defaults)
        results = model.train(
            **base_args,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )

    # Get path to best weights
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print("\nTraining complete!")
    print(f"Best weights saved to: {best_weights}")
    print(f"Validation mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(
        f"Validation mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}"
    )

    return best_weights
