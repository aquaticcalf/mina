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
