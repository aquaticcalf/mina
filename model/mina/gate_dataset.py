"""
Dataset download and organisation for the fish gate classifier.

Two Kaggle datasets are used:

  Fish class  : crowww/a-large-scale-fish-dataset  (~9 000 images, 9 species)
  No-fish class: puneet6060/intel-image-classification  (natural scenes, no fish)

1 000 images per class are randomly sampled and split into train (800) / val (200).
The resulting structure under gate_data/ is what torchvision.datasets.ImageFolder
expects:

    gate_data/
      train/
        fish/
        no_fish/
      val/
        fish/
        no_fish/
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

from mina.core.constants import GATE_DATA_DIR, IMAGE_EXTENSIONS

# ─── Config ───────────────────────────────────────────────────────────────────

FISH_PER_SPLIT: dict[str, int] = {"train": 800, "val": 200}
NO_FISH_PER_SPLIT: dict[str, int] = {"train": 800, "val": 200}

_KAGGLE_FISH_SLUG = "crowww/a-large-scale-fish-dataset"
_KAGGLE_NO_FISH_SLUG = "puneet6060/intel-image-classification"


# ─── Kaggle download helpers ──────────────────────────────────────────────────


def _kaggle_download(slug: str, output_dir: Path) -> Path:
    """
    Download and unzip a Kaggle dataset into output_dir.
    Requires KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
    Returns output_dir (unzipped contents placed inside).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(output_dir),
            "--unzip",
        ],
        check=True,
    )
    return output_dir


def download_kaggle_fish(output_dir: Path) -> Path:
    """
    Download the large-scale fish dataset from Kaggle.
    Returns the directory containing downloaded (unzipped) contents.
    """
    print(f"Downloading fish dataset ({_KAGGLE_FISH_SLUG}) → {output_dir}")
    return _kaggle_download(_KAGGLE_FISH_SLUG, output_dir)


def download_kaggle_no_fish(output_dir: Path) -> Path:
    """
    Download a natural-scenes (no-fish) dataset from Kaggle.
    Returns the directory containing downloaded (unzipped) contents.
    """
    print(f"Downloading no-fish dataset ({_KAGGLE_NO_FISH_SLUG}) → {output_dir}")
    return _kaggle_download(_KAGGLE_NO_FISH_SLUG, output_dir)


# ─── Image collection ─────────────────────────────────────────────────────────


def collect_images(source_dir: Path) -> list[Path]:
    """Recursively collect all image paths from source_dir."""
    return [p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]


# ─── Dataset builder ─────────────────────────────────────────────────────────


def build_gate_dataset(
    fish_source: Path,
    no_fish_source: Path,
    target_dir: Path = GATE_DATA_DIR,
    seed: int = 42,
) -> None:
    """
    Sample images from source directories and write the ImageFolder structure
    into target_dir (gate_data/).

    Args:
        fish_source:    Root dir of the downloaded fish dataset.
        no_fish_source: Root dir of the downloaded no-fish dataset.
        target_dir:     Where to build the final train/val directory tree.
        seed:           Random seed for reproducible sampling.
    """
    random.seed(seed)

    fish_images = collect_images(fish_source)
    no_fish_images = collect_images(no_fish_source)

    total_fish = FISH_PER_SPLIT["train"] + FISH_PER_SPLIT["val"]
    total_no_fish = NO_FISH_PER_SPLIT["train"] + NO_FISH_PER_SPLIT["val"]

    if len(fish_images) < total_fish:
        raise RuntimeError(
            f"Not enough fish images: found {len(fish_images)}, need {total_fish}.\n"
            "Check that the Kaggle download completed successfully."
        )
    if len(no_fish_images) < total_no_fish:
        raise RuntimeError(
            f"Not enough no-fish images: found {len(no_fish_images)}, need {total_no_fish}.\n"
            "Check that the Kaggle download completed successfully."
        )

    fish_sample = random.sample(fish_images, total_fish)
    no_fish_sample = random.sample(no_fish_images, total_no_fish)

    splits: dict[str, dict[str, list[Path]]] = {
        "train": {
            "fish": fish_sample[: FISH_PER_SPLIT["train"]],
            "no_fish": no_fish_sample[: NO_FISH_PER_SPLIT["train"]],
        },
        "val": {
            "fish": fish_sample[FISH_PER_SPLIT["train"] :],
            "no_fish": no_fish_sample[NO_FISH_PER_SPLIT["train"] :],
        },
    }

    print(f"\nBuilding gate dataset at: {target_dir}")
    for split, classes in splits.items():
        for cls, paths in classes.items():
            dest = target_dir / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for src in paths:
                # Prefix with parent folder name to avoid collisions between
                # species that share the same sequential filenames (e.g. 00001.png)
                unique_name = f"{src.parent.name}_{src.name}"
                shutil.copy2(src, dest / unique_name)

    # Summary
    print("\nDataset ready:")
    for split in ("train", "val"):
        for cls in ("fish", "no_fish"):
            count = len(list((target_dir / split / cls).glob("*")))
            print(f"  {split}/{cls}: {count} images")
