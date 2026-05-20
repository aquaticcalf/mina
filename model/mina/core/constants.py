"""
Shared constants for the fish disease detection model.
"""

from pathlib import Path

# Disease classes from the Roboflow dataset (order matters - matches class indices)
DISEASE_CLASSES: list[str] = [
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "parasite",
    "white_tail",
]

# Number of classes
NUM_CLASSES: int = len(DISEASE_CLASSES)

# Default thresholds and sizes
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3
DEFAULT_IMAGE_SIZE: int = 640
DEFAULT_IOU_THRESHOLD: float = 0.6

# Default training parameters
DEFAULT_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_PATIENCE: int = 20

# Tuning parameters
DEFAULT_TUNE_EPOCHS: int = 30
DEFAULT_TUNE_ITERATIONS: int = 300

# Model paths
MODEL_DIR: Path = Path(__file__).parent.parent.parent
RUNS_DIR: Path = MODEL_DIR / "runs" / "detect"
DATA_DIR: Path = MODEL_DIR / "data"
TEST_DATA_DIR: Path = MODEL_DIR / "test_data"

# Supported image extensions
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Fish gate classifier constants ──────────────────────────────────────────
GATE_IMAGE_SIZE: int = 224          # MobileNetV3-Small expected input resolution
GATE_THRESHOLD: float = 0.6         # sigmoid output > threshold → classify as fish
GATE_CLASSES: list[str] = ["fish", "no_fish"]  # alphabetical = ImageFolder order

GATE_RUNS_DIR: Path = MODEL_DIR / "runs" / "gate"
GATE_DATA_DIR: Path = MODEL_DIR / "gate_data"

# ImageNet normalisation applied during MobileNetV3-Small pre-training.
# These MUST match the values hard-coded in the JS gate worker (gate-worker.ts).
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

