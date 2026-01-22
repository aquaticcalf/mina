# Hyperparameter Tuning Integration Plan

## Executive Summary

This document outlines the plan to properly integrate hyperparameter tuning with training in the mina model pipeline, followed by evaluation, and GitHub release of improved models.

---

## 1. Greptile Suggestion Assessment

### What Greptile Suggested
> "Add a --no-tune flag to the CLI and modify train.py to auto-load tuned hyperparameters from runs/detect/tune/best_hyperparameters.yaml"

### Assessment: **Partially Correct, Needs Refinement**

| Aspect | Greptile Claim | Reality | Verdict |
|--------|---------------|---------|---------|
| Tuning output location | `runs/tune/best_hyperparameters.yaml` | Depends on task: `runs/{task}/tune/` | ⚠️ Fragile |
| Auto-loading by YOLO | Implied automatic | **NOT automatic** - requires explicit code | ❌ Incorrect |
| `--no-tune` flag concept | Skip loading tuned params | **Better approach**: explicit `--hyp` flag | ⚠️ Rethink |
| Using `cfg` parameter | Pass YAML to train() | Valid - YOLO accepts `cfg=` parameter natively | ✅ Correct |
| Manual dict merging | `train_args.update(tuned)` | **Risky!** Use `cfg=` for validation | ❌ Avoid |

### Key Findings

1. **YOLO's `train()` method does NOT automatically detect or load `best_hyperparameters.yaml`**

2. **The output path is NOT reliably predictable**:
   - Ultralytics uses `get_save_dir()` which follows: `runs/{task}/{name}`
   - For detection: `runs/detect/tune/best_hyperparameters.yaml`
   - But this depends on task type and whether `name` is overridden
   - **Hardcoding this path is fragile and error-prone**

3. **Better approach**: Instead of hardcoding or auto-detection, use an **explicit `--hyp` flag** that lets the user specify the path directly. This is:
   - More explicit and transparent
   - Works regardless of where tuning saved the file
   - Follows the principle of least surprise
   - Similar to how YOLOv5 does it: `python train.py --hyp runs/evolve/hyp_evolved.yaml`

4. **Use `cfg=` parameter, NOT manual dict merging**:
   - ❌ `train_args.update(yaml.safe_load(...))` - No validation, risky
   - ✅ `model.train(cfg="path/to/yaml", ...)` - Ultralytics validates keys via `check_cfg()`

---

## 2. Current State Analysis

### Current `mina/train.py` (Lines 53-139)
```python
def train(...):
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        name=name,
        device=device,
        # Hardcoded augmentation settings:
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.2,
    )
```

**Problem**: Hardcoded hyperparameters ignore tuning results.

### Current `cli/train.py`
```python
@click.option("--epochs", default=100, ...)
@click.option("--batch", default=16, ...)
@click.option("--imgsz", default=640, ...)
@click.option("--name", default="fish_disease", ...)
@click.option("--device", default="0", ...)
```

**Missing**: No way to load tuned hyperparameters.

### Current `mina/tune.py`
```python
# Returns Path("runs/tune") but actual output is runs/detect/tune/
# The print statement is misleading
```

**Problem**: Misleading return value and print statement.

### Current `mina.ipynb` (Notebook)
```
Tuning Cell → Training Cell → Evaluate → Export → Infer
```

**Problem**: Training cell claims YOLO "automatically loads" tuned hyperparameters - this is incorrect.

---

## 3. Implementation Plan

### Phase 1: Code Changes

#### 3.1 Modify `model/mina/tune.py`

**Return the actual output path from tuning:**

```python
def tune_hyperparameters(...) -> Path:
    """..."""
    # ... existing code ...
    
    model.tune(...)
    
    # Return the ACTUAL output path (not hardcoded)
    # Ultralytics saves to runs/{task}/tune/ by default
    tune_dir = Path("runs/detect/tune")
    best_hyp_path = tune_dir / "best_hyperparameters.yaml"
    
    print("\nTuning complete!")
    print(f"Best hyperparameters saved to: {best_hyp_path}")
    
    return best_hyp_path  # Return actual file path, not directory
```

#### 3.2 Modify `model/mina/train.py`

**Key Insight: Use Ultralytics' built-in `cfg=` parameter**

Ultralytics' `model.train()` method natively supports a `cfg` parameter that accepts a path to a YAML file. This is **much safer** than manually loading and merging dict keys because:

1. Ultralytics validates all keys via `check_cfg()` (type checking, range checking)
2. It handles key deprecations automatically
3. It prevents unknown/conflicting keys from causing silent issues
4. It's the officially supported way to pass configuration files

**From Ultralytics source (model.py line ~756):**
```python
overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
```

**❌ WRONG approach (manual dict merging - risky!):**
```python
hyperparams = yaml.safe_load(open(hyp_path))
train_args.update(hyperparams)  # Dangerous! No validation!
model.train(**train_args)
```

**✅ CORRECT approach (use cfg= parameter):**
```python
model.train(cfg=hyp_path, data=..., epochs=...)  # Ultralytics handles it safely
```

---

**Simplified implementation for `train()` function:**

```python
def train(
    weights: str = "yolov8n.pt",
    data: str | None = None,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    name: str = "fish_disease",
    device: str = "0",
    hyp: str | None = None,  # Path to hyperparameters YAML (from mina-tune)
) -> Results:
```

**Modify training logic to use `cfg=` parameter:**

```python
def train(..., hyp: str | None = None):
    # ... existing model/data setup ...
    
    # Base training arguments (always applied)
    base_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "name": name,
        "device": device,
    }
    
    if hyp:
        # Use Ultralytics' cfg= parameter to safely load hyperparameters
        # This lets Ultralytics handle validation, type checking, etc.
        hyp_path = Path(hyp)
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hyperparameters file not found: {hyp_path}")
        print(f"Using hyperparameters from: {hyp_path}")
        results = model.train(cfg=str(hyp_path), **base_args)
    else:
        # Use default augmentation hyperparameters (hardcoded fallback)
        print("Using default hyperparameters")
        results = model.train(
            **base_args,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.3,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.8,
            mixup=0.2,
        )
    
    return results
```

**Why this is better:**
- No manual YAML parsing needed
- Ultralytics validates keys (CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, etc.)
- Unknown keys will raise errors instead of being silently ignored
- Handles deprecated key names automatically
- Matches how Ultralytics CLI works: `yolo train cfg=best_hyperparameters.yaml`

#### 3.3 Modify `model/cli/train.py`

**Add `--hyp` option:**

```python
@click.command()
@click.option("--epochs", default=100, help="Number of training epochs")
@click.option("--batch", default=16, help="Batch size")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--name", default="fish_disease", help="Run name")
@click.option("--device", default="0", help="Device (cuda:0 or cpu)")
@click.option("--hyp", default=None, type=click.Path(exists=True),
              help="Path to hyperparameters YAML (e.g., from mina-tune)")  # NEW
def main(epochs, batch, imgsz, name, device, hyp):
    train(
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        name=name,
        device=device,
        hyp=hyp,  # NEW
    )
```

#### 3.4 Update `model/mina.ipynb`

**Fix the training cell to explicitly pass hyperparameters:**

```python
# Train with tuned hyperparameters
# Pass the path to best_hyperparameters.yaml from the tuning step

!uv run mina-train --epochs 100 --batch 16 --hyp runs/detect/tune/best_hyperparameters.yaml
```

---

### Phase 2: Google Colab Workflow

#### 2.1 Complete Workflow Order

```
1. Setup (clone, install)
2. Download dataset (mina-download)
3. Tune hyperparameters (mina-tune) → outputs runs/detect/tune/best_hyperparameters.yaml
4. Train with tuned params (mina-train --hyp <path>) → uses the YAML from step 3
5. Evaluate (mina-evaluate) → Get metrics
6. Export (mina-export) → Create TFLite
7. Test inference (mina-infer) → Verify it works
```

#### 2.2 Tuning Configuration

```python
# Tuning: 300 iterations × 30 epochs each = ~300 GPU hours
!uv run mina-tune --iterations 300 --epochs 30 --optimizer AdamW

# Output: runs/detect/tune/best_hyperparameters.yaml
```

#### 2.3 Training Configuration

```python
# Training: Explicitly pass the hyperparameters file
!uv run mina-train --epochs 100 --batch 16 --hyp runs/detect/tune/best_hyperparameters.yaml

# Or without tuned hyperparameters (uses defaults):
!uv run mina-train --epochs 100 --batch 16
```

#### 2.4 Evaluation Metrics to Compare

| Metric | Description | Target |
|--------|-------------|--------|
| mAP50 | Mean Average Precision @ IoU 0.5 | Higher is better |
| mAP50-95 | Mean Average Precision @ IoU 0.5-0.95 | Higher is better |
| Precision | True positives / (True + False positives) | Higher is better |
| Recall | True positives / (True + False negatives) | Higher is better |

```python
# Evaluate and capture metrics
!uv run mina-evaluate --weights runs/detect/fish_disease/weights/best.pt
```

---

### Phase 3: GitHub Release Process

#### 3.1 Decision Criteria

Compare new model metrics against current production model:

```
IF (new_mAP50 > current_mAP50) OR (new_mAP50-95 > current_mAP50-95):
    → Release new model
ELSE:
    → Keep current model
```

#### 3.2 Release Artifacts

Files to include in release:

| File | Description |
|------|-------------|
| `best.pt` | PyTorch weights (full precision) |
| `best_saved_model/best_float32.tflite` | TFLite model (float32) |
| `best_hyperparameters.yaml` | Tuned hyperparameters used |
| `results.csv` | Training metrics |
| `confusion_matrix.png` | Evaluation visualization |

#### 3.3 Release Commands

```bash
# 1. Create release tag
git tag -a v1.x.x -m "Model release with tuned hyperparameters"

# 2. Push tag
git push origin v1.x.x

# 3. Create GitHub release with assets
gh release create v1.x.x \
  runs/detect/fish_disease/weights/best.pt \
  runs/detect/fish_disease/weights/best_saved_model/best_float32.tflite \
  runs/detect/tune/best_hyperparameters.yaml \
  runs/detect/fish_disease/results.csv \
  --title "Fish Disease Detection Model v1.x.x" \
  --notes "Trained with tuned hyperparameters. mAP50: X.XX, mAP50-95: X.XX"
```

---

## 4. File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `model/mina/train.py` | Modify | Add `load_hyperparameters()`, add `hyp` parameter to `train()` |
| `model/cli/train.py` | Modify | Add `--hyp` option |
| `model/mina/tune.py` | Modify | Return actual path to best_hyperparameters.yaml |
| `model/mina.ipynb` | Modify | Fix training cell to use `--hyp` flag |

---

## 5. Testing Plan

### Unit Tests
- [ ] `load_hyperparameters()` returns dict when file exists
- [ ] `load_hyperparameters()` raises FileNotFoundError when file doesn't exist
- [ ] `train()` with `hyp=<path>` loads from YAML
- [ ] `train()` with `hyp=None` uses defaults

### Integration Tests
- [ ] Full pipeline: tune → train --hyp → evaluate → export
- [ ] Verify exported TFLite works in mobile app

---

## 6. Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Code changes | 1-2 hours | Implementation |
| Tuning | ~300 GPU hours | Can take several days on Colab |
| Training | ~5-10 hours | 100 epochs |
| Evaluation | ~30 min | Quick |
| Export + Test | ~30 min | Quick |
| Release | ~30 min | GitHub release creation |

---

## 7. Advantages of `--hyp` Approach

| Aspect | `--no-tune` (Greptile) | `--hyp <path>` (Updated) |
|--------|------------------------|--------------------------|
| Explicit | ❌ Magic auto-detection | ✅ User specifies exact file |
| Flexible | ❌ Hardcoded path | ✅ Any YAML file location |
| Debuggable | ❌ Silent failures if path wrong | ✅ Clear error if file missing |
| Portable | ❌ Breaks if tune output changes | ✅ Works with any tuning setup |
| Similar to | - | YOLOv5 `--hyp` flag |

---

## 8. Next Steps

1. **Approve this plan** - Confirm the approach
2. **Implement code changes** - Modify train.py, cli/train.py, tune.py
3. **Update notebook** - Fix training cell to use `--hyp`
4. **Run in Colab** - Execute full pipeline
5. **Compare metrics** - Evaluate model quality
6. **Release** - If metrics improve, create GitHub release

---

*Plan updated: Using explicit `--hyp` flag instead of auto-detection for reliability*
