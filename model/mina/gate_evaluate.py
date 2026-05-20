"""
Evaluation of the exported gate ONNX model on the validation split.

Runs the ONNX model (via onnxruntime CPU) over every image in gate_data/val/
and reports:
  - Accuracy
  - Precision, Recall, F1 for the "fish" class  (the positive class)

Acceptance criteria before proceeding to Phase 2 (app integration):
  - Val accuracy  ≥ 90 %
  - Precision     ≥ 88 %
  - Recall        ≥ 88 %
  - ONNX size     ≤ 4 MB
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from mina.core.constants import (
    GATE_DATA_DIR,
    GATE_IMAGE_SIZE,
    GATE_RUNS_DIR,
    GATE_THRESHOLD,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

_VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(GATE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def _to_numpy(img_path: Path) -> np.ndarray:
    """Load, transform, and return image as a (1, 3, H, W) float32 array."""
    img = Image.open(img_path).convert("RGB")
    tensor = _VAL_TRANSFORM(img).unsqueeze(0)
    return tensor.numpy()


def evaluate_gate_onnx(
    onnx_path: Path = GATE_RUNS_DIR / "fish_gate.onnx",
    data_dir: Path = GATE_DATA_DIR,
    threshold: float = GATE_THRESHOLD,
) -> dict[str, float]:
    """
    Evaluate the gate ONNX model on the validation split.

    Labels used here:
      fish   → positive class (ground-truth label = 1)
      no_fish→ negative class (ground-truth label = 0)

    Args:
        onnx_path: Path to fish_gate.onnx.
        data_dir:  Root of gate_data/ (must contain val/fish and val/no_fish).
        threshold: Sigmoid probability threshold for "fish" prediction.

    Returns:
        Dict with keys: accuracy, precision, recall, f1.
    """
    import onnxruntime as ort

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    val_dir = data_dir / "val"

    y_true: list[int] = []
    y_pred: list[int] = []

    # ground-truth label: fish=1, no_fish=0
    for cls, label in (("fish", 1), ("no_fish", 0)):
        cls_dir = val_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Expected val directory not found: {cls_dir}\n"
                "Run gate-download first."
            )
        images = sorted(p for p in cls_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        for img_path in images:
            arr = _to_numpy(img_path)
            logit = float(session.run(None, {"images": arr})[0][0][0])
            prob = 1 / (1 + np.exp(-logit))
            y_true.append(label)
            y_pred.append(1 if prob > threshold else 0)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = float((y_true_arr == y_pred_arr).mean())
    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    print(f"\n=== Gate Model Evaluation  (threshold = {threshold}) ===")
    print(f"  Model size : {size_mb:.2f} MB")
    print(f"  Samples    : {len(y_true)} ({tp + fn} fish, {tn + fp} no_fish)")
    print()
    print(f"  Accuracy   : {accuracy:.4f}  (≥ 0.90 required)")
    print(f"  Precision  : {precision:.4f}  (≥ 0.88 required)")
    print(f"  Recall     : {recall:.4f}  (≥ 0.88 required)")
    print(f"  F1 Score   : {f1:.4f}")
    print()
    print(f"  Confusion matrix:")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")
    print()

    # Acceptance criteria
    passed = accuracy >= 0.90 and precision >= 0.88 and recall >= 0.88 and size_mb <= 4.0
    if passed:
        print("  ✓ All acceptance criteria met — ready for Phase 2.")
    else:
        print("  ✗ Some criteria NOT met — consider:")
        if accuracy < 0.90:
            print("    • Increase phase2_epochs to 20–25")
        if precision < 0.88 or recall < 0.88:
            print("    • Adjust threshold (try 0.5 to improve recall, 0.7 to improve precision)")
        if size_mb > 4.0:
            print("    • Check that you exported the correct model file")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
