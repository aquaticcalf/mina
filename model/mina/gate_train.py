"""
Transfer learning trainer for the binary fish gate classifier.

Architecture: MobileNetV3-Small fine-tuned for binary "fish vs. no_fish" classification.
Runtime: onnxruntime-web (WASM) in the browser — this training script produces the
         .pt weights that gate_export.py converts to ONNX.

Training strategy — two phases:
  Phase 1 (head-only, 5 epochs, lr=1e-3):
    Freeze all feature layers. Train only the new Linear(in_features, 1) head.
    Converges quickly to a reasonable baseline.

  Phase 2 (fine-tune, 15 epochs, lr=5e-5):
    Unfreeze last 2 backbone blocks + head. Lower learning rate adapts the
    backbone to fish images without catastrophic forgetting of ImageNet features.

Label convention:
  torchvision.datasets.ImageFolder sorts classes alphabetically:
    class_to_idx = {'fish': 0, 'no_fish': 1}
  We train with BCEWithLogitsLoss and FLIP the original label so that:
    fish   → target = 1.0   (high sigmoid output = "it IS a fish")
    no_fish→ target = 0.0
  In the JS gate worker, sigmoid(logit) > 0.6 therefore means "fish detected".
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from mina.core.constants import (
    GATE_DATA_DIR,
    GATE_IMAGE_SIZE,
    GATE_RUNS_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ─── Transforms ───────────────────────────────────────────────────────────────


def get_transforms(split: str) -> transforms.Compose:
    """Return image transforms for the given split (train or val)."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(GATE_IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    # val / test — fully deterministic
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(GATE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ─── Model factory ────────────────────────────────────────────────────────────


def build_gate_model() -> nn.Module:
    """
    MobileNetV3-Small pretrained on ImageNet1K with the final classifier
    replaced by a single-output linear layer for binary classification.
    """
    base = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features: int = base.classifier[3].in_features  # type: ignore[union-attr]
    base.classifier[3] = nn.Linear(in_features, 1)
    return base


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all feature extraction layers (used during Phase 1)."""
    for param in model.features.parameters():  # type: ignore[attr-defined]
        param.requires_grad = False


def unfreeze_last_n_blocks(model: nn.Module, n: int = 2) -> None:
    """Unfreeze the last *n* children of model.features (used at start of Phase 2)."""
    blocks = list(model.features.children())  # type: ignore[attr-defined]
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True


# ─── Single epoch ─────────────────────────────────────────────────────────────


def run_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training or validation epoch.

    Args:
        model:     The gate model.
        loader:    DataLoader for this split.
        criterion: Loss function (BCEWithLogitsLoss).
        optimizer: If None, run in eval mode (no gradient updates).
        device:    Torch device.

    Returns:
        (average_loss, accuracy) over the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            # ImageFolder: fish=0, no_fish=1 → flip so fish=1 (target for sigmoid)
            labels = (1 - labels).float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ─── Main training function ───────────────────────────────────────────────────


def train_gate(
    data_dir: Path = GATE_DATA_DIR,
    output_dir: Path = GATE_RUNS_DIR,
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
    batch_size: int = 32,
    device_str: str | None = None,
) -> Path:
    """
    Train the fish gate classifier using two-phase transfer learning.

    Args:
        data_dir:       Root of gate_data/ directory (must contain train/ and val/).
        output_dir:     Where checkpoints and the best .pt file are saved.
        phase1_epochs:  Epochs for head-only training (Phase 1).
        phase2_epochs:  Epochs for backbone fine-tuning (Phase 2).
        batch_size:     Batch size. 32 is safe for Colab T4 GPU at 224×224.
        device_str:     'cuda', 'cpu', or None to auto-detect.

    Returns:
        Path to the best saved weights file (fish_gate_best.pt).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # ── Data loaders ─────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(str(data_dir / "train"), get_transforms("train"))
    val_ds = datasets.ImageFolder(str(data_dir / "val"), get_transforms("val"))

    print(f"Class mapping (ImageFolder): {train_ds.class_to_idx}")
    print(f"  → 'fish' maps to index {train_ds.class_to_idx['fish']}, "
          f"target after flip = 1.0 (sigmoid > 0.6 = fish)")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_gate_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    best_weights_path = output_dir / "fish_gate_best.pt"

    def _save_if_best(val_acc: float) -> None:
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  ✓ New best val accuracy: {val_acc:.4f} — saved to {best_weights_path}")

    # ── Phase 1: Train classification head only ───────────────────────────────
    print(f"\n=== Phase 1: Head-only training ({phase1_epochs} epochs) ===")
    freeze_backbone(model)

    optimizer_p1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=phase1_epochs
    )

    for epoch in range(1, phase1_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p1, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler_p1.step()
        print(
            f"  Epoch {epoch:02d}/{phase1_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        _save_if_best(val_acc)

    # ── Phase 2: Fine-tune last backbone blocks ───────────────────────────────
    print(f"\n=== Phase 2: Backbone fine-tuning ({phase2_epochs} epochs) ===")
    unfreeze_last_n_blocks(model, n=2)

    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=phase2_epochs
    )

    for epoch in range(1, phase2_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_p2, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler_p2.step()
        print(
            f"  Epoch {epoch:02d}/{phase2_epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )
        _save_if_best(val_acc)

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best weights:      {best_weights_path}")
    return best_weights_path
