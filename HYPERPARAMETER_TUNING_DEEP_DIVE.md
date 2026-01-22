# 📘 Hyperparameter Tuning in MINA – Deep Explanation

> **PR #11 Analysis: Understanding Automated Hyperparameter Optimization**
> 
> This document explains what hyperparameter tuning is, why we need it, and how PR #11 implements it in our fish disease detection pipeline.

---

## 1. What Problem This PR Solves

### What Are Hyperparameters?

**Simple explanation:** Settings you configure BEFORE training starts that control HOW the model learns.

**In traditional software terms:**
```python
# Regular parameters: Learned from data
model_weights = [0.5, 0.3, 0.2, ...]  # These change during training

# Hyperparameters: Set by you before training
learning_rate = 0.001   # How big are the learning steps?
batch_size = 16         # How many images to process at once?
optimizer = "AdamW"     # Which algorithm to use for learning?
```

**Analogy:** 
- **Parameters** = Skills a student learns (learned from studying)
- **Hyperparameters** = Study plan settings (study 2 hours/day? Use flashcards or practice tests?)

### Examples from Our Codebase

**In `train.py`, we hardcoded these:**
```python
results = model.train(
    data=str(data_yaml),
    epochs=100,              # Hyperparameter
    batch=16,                # Hyperparameter
    imgsz=640,               # Hyperparameter
    patience=20,             # Hyperparameter
    # Augmentation hyperparameters:
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    # ... many more
)
```

**The problem:**
- Who decided `batch=16` is optimal? Maybe `batch=32` is better?
- Who decided `hsv_h=0.015`? Maybe `0.02` gives better accuracy?
- We just guessed these values based on "common practice"

**⚠️ The consequence:**
Our model might be working at 85% of its potential because we didn't optimize these settings.

### Why Tuning Is Needed

**Without tuning (what we had before):**
```
You → Pick hyperparameters based on gut feel
    → Train model (100 epochs)
    → Get mAP = 0.87
    → Shrug and ship it
```

**With tuning (what PR #11 adds):**
```
You → Define a search space (ranges for each hyperparameter)
    → Automated system tries 300 different combinations
    → Each combination trains for 30 epochs
    → System picks the best combination
    → You train final model with optimal hyperparameters
    → Get mAP = 0.91 (4% improvement!)
```

**Think of it like:**
- **Without tuning** = Cooking a new recipe by guessing the amounts
- **With tuning** = Testing 300 variations and finding the perfect recipe

### Why This Matters for OUR Model Specifically

**1. Mobile deployment is resource-constrained**
- Every percentage point of accuracy matters
- Can't just throw more compute at the problem post-deployment
- Getting the best possible model BEFORE export is critical

**2. Limited training data**
- We don't have millions of fish disease images
- Optimal hyperparameters can squeeze more performance from limited data
- The difference between 87% and 91% mAP could mean catching diseases we'd otherwise miss

**3. Cost savings**
- Better hyperparameters = Fewer training runs needed
- Instead of manually trying 50 different configurations, automate it
- Your time is expensive, computer time is cheap

**4. Scientific rigor**
- Instead of "I think batch=16 is good", you have data: "300 trials show batch=24 is optimal"
- Makes the model reproducible and defensible

---

## 2. What Is AdamW

### What "Optimizer" Means in Training

**Simple explanation:** The algorithm that updates the model's weights during training.

**The training loop (simplified):**
```python
for epoch in range(100):
    for batch in training_data:
        # 1. Forward pass: Make predictions
        predictions = model(batch.images)
        
        # 2. Calculate loss: How wrong were we?
        loss = calculate_loss(predictions, batch.labels)
        
        # 3. Backward pass: Calculate gradients
        gradients = loss.backward()  # "Which direction to adjust weights?"
        
        # 4. Optimizer: Update weights
        optimizer.step(gradients)  # ← THIS IS WHERE THE OPTIMIZER WORKS
```

**The optimizer's job:**
> Given the gradients (directions to improve), decide HOW MUCH to adjust each weight.

**Analogy:**
- **Gradients** = Compass showing which way is downhill
- **Optimizer** = Your legs deciding how big of a step to take

### How Adam Works

**Adam** = "Adaptive Moment Estimation"

**What it does:**
1. **Adaptive learning rates**: Different weights get different step sizes
2. **Momentum**: Remembers previous directions to avoid zigzagging
3. **Second-order momentum**: Remembers how steep the gradient was

**In code (conceptually):**
```python
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.beta1 = beta1  # First moment (momentum)
        self.beta2 = beta2  # Second moment (variance)
        self.m = 0  # First moment estimate
        self.v = 0  # Second moment estimate
    
    def step(self, gradient):
        # Update momentum
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        
        # Update variance
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        
        # Compute update
        update = self.lr * self.m / (sqrt(self.v) + epsilon)
        
        # Apply update to weights
        weights -= update
```

**Why Adam is popular:**
- Works well on most problems
- Requires minimal tuning
- Handles sparse gradients well

### What AdamW Changes vs Adam

**The problem with Adam:**
- It applies **weight decay** (regularization) incorrectly
- Weight decay gets tangled with the adaptive learning rate
- This reduces its effectiveness

**AdamW fixes this:**
```python
class AdamW(Adam):
    def step(self, gradient, weight_decay=0.01):
        # 1. Do the normal Adam update
        update = self.compute_adam_update(gradient)
        
        # 2. Apply weight decay SEPARATELY (decoupled)
        weights = weights - weight_decay * weights  # ← The "W" in AdamW
        
        # 3. Then apply the Adam update
        weights = weights - update
```

**Key difference:**
```
Adam:    weights -= lr * gradient / sqrt(v) - weight_decay * weights
         └─────── These are mixed together ──────┘

AdamW:   weights -= weight_decay * weights
         weights -= lr * gradient / sqrt(v)
         └─────── These are decoupled ──────┘
```

**Why this matters:**
- **Better generalization**: The model generalizes better to new data
- **More stable training**: Less likely to overfit
- **Better performance**: Empirically outperforms Adam in many tasks

**In plain English:**
Adam ties weight decay to learning rate. AdamW separates them, which gives better results.

### Why AdamW Is Chosen Here

**From `cli/tune.py`:**
```python
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",  # ← Our choice
    choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam"],
    help="Optimizer to use (default: AdamW)",
)
```

**Reasons:**

1. **State-of-the-art performance**
   - AdamW is the current best practice (2017 research, proven over time)
   - Used in most modern vision models (ViT, BERT, GPT, etc.)

2. **Better than Adam for our use case**
   - Fish disease detection = limited data (better generalization needed)
   - AdamW's decoupled weight decay helps prevent overfitting

3. **YOLOv8 support**
   - Ultralytics library supports AdamW natively
   - Well-tested and optimized for YOLO

4. **Hyperparameter tuning friendly**
   - AdamW has fewer failure modes than SGD
   - More robust across different hyperparameter settings

**⚠️ Design Decision:**
> We default to AdamW but allow other optimizers for experimentation. SGD might work better in some cases, but AdamW is the safest starting point.

---

## 3. What Is Hyperparameter Tuning (In Practice)

### What Parameters Are Being Tuned

**In our implementation (`mina/tune.py`):**
```python
model.tune(
    data=str(data_yaml),
    epochs=epochs,        # Not tuned (fixed per trial)
    iterations=iterations, # Not tuned (controls how many trials)
    optimizer=optimizer,   # Not tuned (we pick AdamW)
    device=device,
    plots=True,
    save=True,
    val=True,
)
```

**What YOLO's `.tune()` actually tunes internally:**

The YOLOv8 library has a **predefined search space**. It tunes:

1. **Learning rate (lr0)**: Initial learning rate (e.g., 0.001 to 0.1)
2. **Final learning rate (lrf)**: Learning rate at end of training
3. **Momentum**: SGD momentum or Adam beta1
4. **Weight decay**: Regularization strength
5. **Warmup epochs**: How long to gradually increase learning rate
6. **Batch size**: How many images per batch
7. **Augmentation hyperparameters**:
   - `hsv_h`: Hue variation
   - `hsv_s`: Saturation variation
   - `hsv_v`: Value (brightness) variation
   - `degrees`: Rotation range
   - `translate`: Translation range
   - `scale`: Scaling range
   - `mosaic`: Mosaic augmentation probability
   - `mixup`: Mixup augmentation probability

**⚠️ Important:**
We don't define the search space ourselves. YOLOv8 has sensible defaults built-in.

### What "Iterations" and "Epochs Per Iteration" Mean

**From `constants.py`:**
```python
DEFAULT_TUNE_EPOCHS: int = 30       # Epochs per trial
DEFAULT_TUNE_ITERATIONS: int = 300  # Number of trials
```

**Breakdown:**

```
Total tuning process:
├─ Iteration 1 (Trial 1)
│  ├─ Try hyperparameters: {lr=0.01, batch=16, hsv_h=0.02, ...}
│  └─ Train for 30 epochs → mAP = 0.83
│
├─ Iteration 2 (Trial 2)
│  ├─ Try hyperparameters: {lr=0.005, batch=24, hsv_h=0.015, ...}
│  └─ Train for 30 epochs → mAP = 0.85
│
├─ Iteration 3 (Trial 3)
│  ├─ Try hyperparameters: {lr=0.008, batch=32, hsv_h=0.018, ...}
│  └─ Train for 30 epochs → mAP = 0.87  ← Best so far!
│
... (repeat 297 more times)
│
└─ Iteration 300 (Trial 300)
   ├─ Try hyperparameters: {lr=0.007, batch=28, hsv_h=0.017, ...}
   └─ Train for 30 epochs → mAP = 0.84

Final result:
Best hyperparameters: Iteration 3's config
Best mAP: 0.87
```

**Key insight:**
- **Iteration** = One complete trial with a specific hyperparameter configuration
- **Epochs** = How long we train each trial (30 is enough to evaluate if settings are good)
- **Total training** = 300 iterations × 30 epochs = 9,000 epochs worth of training!

**Why 30 epochs per iteration?**
- Full training is 100 epochs
- We don't need 100 epochs to know if hyperparameters are good/bad
- 30 epochs is enough to see trends
- Saves time: 30 vs 100 = 3.3x faster

**Why 300 iterations?**
- Ray Tune's default (proven to work well)
- Enough to explore the search space
- Not so many that it takes forever

### How Ray Tune Is Involved

**Ray Tune** = A library for distributed hyperparameter tuning.

**What it does:**
1. **Defines search space**: Ranges for each hyperparameter
2. **Sampling strategy**: How to pick the next set of hyperparameters to try
3. **Early stopping**: Kill bad trials early to save time
4. **Result tracking**: Logs all trials and their performance

**In our code:**
```python
model.tune(
    data=str(data_yaml),
    epochs=30,
    iterations=300,
    optimizer="AdamW",
)
```

**Behind the scenes (inside YOLOv8):**
```python
# Pseudocode of what YOLO does internally
import ray
from ray import tune

def train_trial(config):
    # Extract hyperparameters from config
    lr = config['lr']
    batch = config['batch']
    hsv_h = config['hsv_h']
    # ... etc
    
    # Train model for 30 epochs with these hyperparameters
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=30,
        lr0=lr,
        batch=batch,
        hsv_h=hsv_h,
        # ... etc
    )
    
    # Return metric we're trying to maximize
    return {"mAP": results.mAP50}

# Define search space
search_space = {
    "lr": tune.loguniform(0.0001, 0.1),
    "batch": tune.choice([8, 16, 24, 32]),
    "hsv_h": tune.uniform(0.0, 0.03),
    # ... etc
}

# Run tuning
analysis = tune.run(
    train_trial,
    config=search_space,
    num_samples=300,  # iterations
    # ... other settings
)

# Get best config
best_config = analysis.get_best_config(metric="mAP", mode="max")
```

**Ray Tune's intelligence:**
- **Random sampling** (early iterations): Try random combinations
- **Bayesian optimization** (later iterations): Focus on promising regions
- **Early stopping**: If a trial is clearly bad after 10 epochs, kill it

**Think of it like:**
- Ray Tune = Manager coordinating 300 experiments
- Each trial = One experiment with specific settings
- YOLO = Worker running each experiment

### What YOLO's `.tune()` Actually Does Internally

**Step-by-step:**

1. **Initialize Ray Tune**
   ```python
   ray.init()
   ```

2. **Define search space** (built into YOLO)
   ```python
   search_space = {
       "lr0": tune.uniform(1e-5, 1e-1),
       "lrf": tune.uniform(0.01, 1.0),
       "momentum": tune.uniform(0.6, 0.98),
       "weight_decay": tune.uniform(0.0, 0.001),
       "warmup_epochs": tune.uniform(0.0, 5.0),
       "batch": tune.choice([8, 16, 32, 64]),
       "hsv_h": tune.uniform(0.0, 0.1),
       "hsv_s": tune.uniform(0.0, 0.9),
       "hsv_v": tune.uniform(0.0, 0.9),
       # ... many more
   }
   ```

3. **Run trials**
   ```python
   for iteration in range(300):
       # Sample hyperparameters from search space
       config = sample_from_search_space()
       
       # Train for 30 epochs
       results = train_with_config(config, epochs=30)
       
       # Log results
       log_trial(iteration, config, results.mAP)
   ```

4. **Select best configuration**
   ```python
   best_config = get_best_trial(metric="mAP50")
   ```

5. **Save results**
   ```python
   save_to_yaml("runs/tune/best_hyperparameters.yaml", best_config)
   ```

**What gets saved:**
```yaml
# runs/tune/best_hyperparameters.yaml
lr0: 0.00853
lrf: 0.123
momentum: 0.843
weight_decay: 0.00031
warmup_epochs: 2.3
batch: 24
hsv_h: 0.0172
hsv_s: 0.671
hsv_v: 0.389
# ... etc
```

**What you do next:**
```bash
# Use the tuned hyperparameters for final training
uv run mina-train --config runs/tune/best_hyperparameters.yaml --epochs 100
```

---

## 4. Walkthrough of Our Implementation

### `cli/tune.py` – Line-by-Line

```python
"""
CLI for hyperparameter tuning using Ray Tune.

Usage:
    uv run mina-tune [--data PATH] [--epochs N] [--iterations N] [--optimizer NAME] [--device DEVICE]
"""
```
**Purpose:** Entry point for the tuning command. Thin wrapper around `mina.tune.tune_hyperparameters()`.

```python
import argparse

from mina.tune import tune_hyperparameters
from mina.core.constants import DEFAULT_TUNE_EPOCHS, DEFAULT_TUNE_ITERATIONS
```
**Imports:**
- `argparse`: Parse command-line arguments
- `tune_hyperparameters`: The actual tuning logic
- Constants: Default values for epochs and iterations

```python
def main():
    parser = argparse.ArgumentParser(
        description="Tune YOLOv8 hyperparameters using Ray Tune"
    )
```
**Setup:** Create argument parser with description.

```python
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (default: auto-detect)",
    )
```
**`--data` argument:**
- Optional path to `data.yaml`
- If not provided, auto-detects using `get_data_yaml_path()` (same as `mina-train`)
- **Use case:** "Use the same dataset as training" (most common) or "Try a different dataset"

```python
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TUNE_EPOCHS,
        help=f"Training epochs per iteration (default: {DEFAULT_TUNE_EPOCHS})",
    )
```
**`--epochs` argument:**
- How long to train each trial
- Default: 30 (from constants)
- **Lower value** = Faster tuning, less accurate evaluation
- **Higher value** = Slower tuning, more accurate evaluation
- **⚠️ Warning:** Setting this to 100 would make tuning take 3.3x longer!

```python
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_TUNE_ITERATIONS,
        help=f"Total tuning iterations (default: {DEFAULT_TUNE_ITERATIONS})",
    )
```
**`--iterations` argument:**
- How many different hyperparameter combinations to try
- Default: 300 (Ray Tune's sweet spot)
- **Lower value** = Faster, might miss optimal settings
- **Higher value** = Slower, more thorough search
- **⚠️ Warning:** 300 iterations × 30 epochs = ~100 hours on CPU!

```python
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam"],
        help="Optimizer to use (default: AdamW)",
    )
```
**`--optimizer` argument:**
- Which optimizer to use for all trials
- Default: AdamW (best practice)
- **Choices:**
  - `SGD`: Classic, requires careful learning rate tuning
  - `Adam`: Popular, but AdamW is better
  - `AdamW`: Recommended (decoupled weight decay)
  - `NAdam`: Adam + Nesterov momentum
  - `RAdam`: Rectified Adam (more stable early training)
- **Design decision:** We pick ONE optimizer and tune hyperparameters for it, rather than tuning optimizer choice itself

```python
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use: '0' for GPU, 'cpu' for CPU (default: 0)",
    )
```
**`--device` argument:**
- GPU (fast) vs CPU (slow)
- Default: "0" (first GPU)
- **⚠️ Critical:** Tuning on CPU is PAINFULLY slow (days instead of hours)

```python
    args = parser.parse_args()

    tune_hyperparameters(
        data=args.data,
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer=args.optimizer,
        device=args.device,
    )
```
**Execution:**
1. Parse arguments
2. Call the tuning function with parsed values
3. That's it! CLI is just a thin wrapper

```python
if __name__ == "__main__":
    main()
```
**Entry point:** Standard Python pattern for scripts.

### `mina/tune.py` – Line-by-Line

```python
"""
Hyperparameter tuning using YOLOv8's built-in tune() method.
"""
```
**Module purpose:** Wrapper around YOLOv8's tuning functionality.

```python
from pathlib import Path

from ultralytics import YOLO

from mina.core.constants import (
    DEFAULT_TUNE_EPOCHS,
    DEFAULT_TUNE_ITERATIONS,
)
from mina.train import get_data_yaml_path
```
**Imports:**
- `Path`: File path handling
- `YOLO`: Ultralytics model class
- Constants: Default tuning parameters
- `get_data_yaml_path`: Reuses dataset discovery logic from training

**⚠️ Design Decision:** Reuses `get_data_yaml_path()` from `train.py` to avoid duplication. Same dataset discovery logic for both training and tuning.

```python
def tune_hyperparameters(
    data: str | None = None,
    epochs: int = DEFAULT_TUNE_EPOCHS,
    iterations: int = DEFAULT_TUNE_ITERATIONS,
    optimizer: str = "AdamW",
    device: str = "0",
) -> Path:
```
**Function signature:**
- **Returns:** Path to tuning results (useful for automation/pipelines)
- **Defaults:** All parameters have sensible defaults

```python
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
```
**Docstring:** Clear documentation of what this function does.

```python
    if data is None:
        data_yaml = get_data_yaml_path()
    else:
        data_yaml = Path(data)
        if not data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
```
**Dataset handling:**
- If no path provided → Auto-detect (same logic as `mina-train`)
- If path provided → Validate it exists
- **Fail fast:** Raises error immediately if dataset not found

```python
    print(f"Using dataset config: {data_yaml}")
    print("Starting hyperparameter tuning...")
    print(f"  Epochs per iteration: {epochs}")
    print(f"  Total iterations: {iterations}")
    print(f"  Optimizer: {optimizer}")
```
**User feedback:**
- Print configuration so user knows what's happening
- Important for long-running processes (tuning can take hours/days)
- **UX consideration:** Users need to know the process started successfully

```python
    model = YOLO("yolov8n.pt")
```
**Model initialization:**
- Starts from YOLOv8n pretrained weights (same as training)
- **Each trial** will start from these weights and train with different hyperparameters

```python
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
```
**The actual tuning call:**
- `data`: Dataset configuration
- `epochs`: How long to train each trial (30)
- `iterations`: How many trials (300)
- `optimizer`: Which optimizer to use (AdamW)
- `device`: GPU or CPU
- `plots=True`: Generate visualization plots
- `save=True`: Save results
- `val=True`: Run validation during training (needed to compute mAP)

**⚠️ This is where the magic happens:** YOLOv8's `.tune()` method handles all the Ray Tune complexity internally.

```python
    print("\nTuning complete!")
    print("Results saved to: runs/tune")

    return Path("runs/tune")
```
**Completion:**
- Notify user
- Return path to results (for scripting)
- Results include: best hyperparameters, plots, logs

### Constants Added to `constants.py`

```python
# Tuning parameters
DEFAULT_TUNE_EPOCHS: int = 30
DEFAULT_TUNE_ITERATIONS: int = 300
```

**Why these specific values?**

**`DEFAULT_TUNE_EPOCHS = 30`**
- **Rationale:** 30% of full training (100 epochs)
- **Assumption:** Trends visible by epoch 30
- **Trade-off:** Could be 20 (faster) or 50 (more accurate), but 30 is the sweet spot
- **Math:** If a trial is doing poorly by epoch 30, it won't magically become good at epoch 100

**`DEFAULT_TUNE_ITERATIONS = 300`**
- **Rationale:** Ray Tune's recommended default
- **Search space size:** With ~20 hyperparameters, 300 samples give decent coverage
- **Practical limit:** More than 300 takes too long for most use cases
- **Less than 300:** Might miss optimal configuration

**⚠️ Design Decision:**
> These are CONSERVATIVE values. If you have a GPU cluster, you could do `epochs=50, iterations=1000`. For a laptop, you might do `epochs=10, iterations=50` just to test if tuning helps at all.

### How This Reuses the Existing Training Pipeline

```
┌────────────────────────────────────────────────────────┐
│         SHARED INFRASTRUCTURE                          │
├────────────────────────────────────────────────────────┤
│                                                         │
│  mina/train.py                mina/tune.py             │
│  ↓                            ↓                        │
│  get_data_yaml_path()  ←──────┘                       │
│  (shared function)                                     │
│                                                         │
│  Both use:                                             │
│  - Same dataset (data.yaml)                            │
│  - Same YOLO model (yolov8n.pt)                        │
│  - Same validation logic                               │
│  - Same device handling                                │
│                                                         │
│  Differences:                                          │
│  train.py:             tune.py:                        │
│  - Fixed hyperparams   - Search over hyperparams       │
│  - 100 epochs          - 30 epochs per trial           │
│  - One training run    - 300 training runs             │
│  - Saves best.pt       - Saves best_hyperparams.yaml   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

**Key reuse:**
1. **Dataset discovery:** Both use `get_data_yaml_path()`
2. **Model initialization:** Both start from `yolov8n.pt`
3. **Validation logic:** Both compute mAP the same way
4. **Constants:** Both respect `DISEASE_CLASSES`, `DEFAULT_IMAGE_SIZE`, etc.

**Why this matters:**
- **Consistency:** Tuning and training use the same pipeline
- **No duplication:** Don't rewrite dataset loading logic
- **Trust:** If training works, tuning will work (same code paths)

---

## 5. How This Fits Into Our Existing Training Flow

### Where This Sits Relative to `mina-train`

```
┌─────────────────────────────────────────────────────────────┐
│              RECOMMENDED WORKFLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Download Dataset                                         │
│     └─ uv run mina-download                                  │
│        ↓                                                      │
│     └─ data/images/train/, data/images/val/                 │
│                                                               │
│  2. Quick Baseline (Optional)                                │
│     └─ uv run mina-train --epochs 10                        │
│        Purpose: Verify dataset works, get rough baseline    │
│        ↓                                                      │
│     └─ runs/detect/fish_disease/weights/best.pt             │
│        (Discard this, just for testing)                      │
│                                                               │
│  3. Hyperparameter Tuning (NEW!)                            │
│     └─ uv run mina-tune --epochs 30 --iterations 300        │
│        Purpose: Find optimal hyperparameters                 │
│        Time: ~100 GPU hours (or days on CPU)                │
│        ↓                                                      │
│     └─ runs/tune/best_hyperparameters.yaml                  │
│        {lr0: 0.00853, batch: 24, hsv_h: 0.0172, ...}        │
│                                                               │
│  4. Final Training (With Tuned Hyperparameters)             │
│     └─ uv run mina-train --config runs/tune/best.yaml      │
│        --epochs 100 --patience 20                            │
│        Purpose: Train final model with optimal settings      │
│        Time: ~10 GPU hours                                   │
│        ↓                                                      │
│     └─ runs/detect/fish_disease_final/weights/best.pt      │
│        (This is your production model!)                      │
│                                                               │
│  5. Export to TFLite                                        │
│     └─ uv run mina-export --weights best.pt                 │
│        ↓                                                      │
│     └─ best_full_integer_quant.tflite                       │
│                                                               │
│  6. Deploy to App                                            │
│     └─ Upload to GitHub releases                             │
│        ↓                                                      │
│     └─ App downloads and uses model                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### What Changes and What Stays the Same

**Changes:**
- **Before PR #11:** Manual hyperparameter selection
  ```bash
  uv run mina-train --epochs 100 --batch 16
  # Hope for the best!
  ```

- **After PR #11:** Data-driven hyperparameter selection
  ```bash
  uv run mina-tune --iterations 300  # Find optimal settings
  uv run mina-train --config runs/tune/best.yaml --epochs 100
  # Use proven optimal settings!
  ```

**Stays the same:**
- Dataset format
- Model architecture (YOLOv8n)
- Export process
- App integration
- Disease classes

**⚠️ Important:**
> Tuning doesn't replace training. It's a **preprocessing step** that finds the best hyperparameters. You still run `mina-train` for the final model.

### What Artifacts It Produces

**Directory structure after tuning:**
```
runs/tune/
├── best_hyperparameters.yaml    # ← The important output!
├── tune_results.csv              # All trial results
├── plots/
│   ├── hyperparameter_evolution.png
│   ├── mAP_distribution.png
│   └── parallel_coordinates.png
└── trials/
    ├── trial_001/
    │   ├── weights/
    │   └── results.png
    ├── trial_002/
    │   ├── weights/
    │   └── results.png
    ... (300 trials)
```

**Key files:**

1. **`best_hyperparameters.yaml`** (Most important)
   ```yaml
   lr0: 0.00853
   lrf: 0.123
   momentum: 0.843
   weight_decay: 0.00031
   warmup_epochs: 2.3
   batch: 24
   hsv_h: 0.0172
   hsv_s: 0.671
   hsv_v: 0.389
   degrees: 8.3
   translate: 0.087
   scale: 0.423
   mosaic: 0.91
   mixup: 0.078
   ```
   **Use this for final training!**

2. **`tune_results.csv`**
   ```csv
   trial,lr0,batch,hsv_h,...,mAP50,mAP50-95
   1,0.01,16,0.02,...,0.832,0.671
   2,0.005,24,0.015,...,0.851,0.689
   3,0.008,32,0.018,...,0.873,0.702  ← Best
   ...
   ```
   **Useful for analysis and visualization**

3. **Plots**
   - Visualize which hyperparameters matter most
   - See distribution of results
   - Identify trends

**⚠️ Storage Warning:**
300 trials × ~100MB per trial = ~30GB of data! Make sure you have disk space.

---

## 6. What The Tests Are Actually Proving

### `test_tuning.py` Explained in Simple Terms

**Purpose:** Ensure the tuning configuration is sane and doesn't break.

### Test Class 1: `TestTuningConfiguration`

#### Test 1: `test_default_tune_epochs`
```python
def test_default_tune_epochs(self):
    """Verify default tuning epochs is 30."""
    assert DEFAULT_TUNE_EPOCHS == 30
```

**What it proves:**
- The constant hasn't been accidentally changed
- **Why it matters:** If someone changes this to 100, tuning becomes 3.3x slower

**Invariant protected:**
> `DEFAULT_TUNE_EPOCHS` must always be 30 (unless intentionally changed with understanding of cost)

#### Test 2: `test_default_tune_iterations`
```python
def test_default_tune_iterations(self):
    """Verify default tuning iterations is 300."""
    assert DEFAULT_TUNE_ITERATIONS == 300
```

**What it proves:**
- The constant matches the design spec
- **Why it matters:** This is the agreed-upon sweet spot

**Invariant protected:**
> `DEFAULT_TUNE_ITERATIONS` must be 300 (Ray Tune's recommended value)

#### Test 3: `test_tuning_params_valid` (Property-based)
```python
@given(
    epochs=st.integers(min_value=5, max_value=100),
    iterations=st.integers(min_value=50, max_value=1000),
)
@settings(max_examples=50)
def test_tuning_params_valid(self, epochs: int, iterations: int):
    """
    For any valid combination of tuning parameters,
    they should be within acceptable ranges.
    """
    assert epochs > 0
    assert iterations > 0
```

**What it proves:**
- **For ANY combination** of epochs (5-100) and iterations (50-1000), both are positive
- This is a sanity check, not a deep test

**Why property-based testing here?**
- Tests the CONCEPT: "epochs and iterations must be positive"
- Automatically tests 50 random combinations
- Catches edge cases (e.g., someone accidentally allows negative values)

**Invariant protected:**
> Tuning parameters must always be positive integers

#### Test 4: `test_tune_function_signature`
```python
def test_tune_function_signature(self):
    """Verify tune_hyperparameters has expected parameters."""
    sig = inspect.signature(tune_hyperparameters)
    params = list(sig.parameters.keys())

    assert "data" in params
    assert "epochs" in params
    assert "iterations" in params
    assert "optimizer" in params
    assert "device" in params
```

**What it proves:**
- The function signature matches the design
- **Why it matters:** If someone removes a parameter, this catches it

**Invariant protected:**
> `tune_hyperparameters()` must accept: data, epochs, iterations, optimizer, device

**Think of it like:**
This is a "contract test". The function promises to accept these parameters. If the contract breaks, tests fail.

#### Test 5: `test_tune_function_has_docstring`
```python
def test_tune_function_has_docstring(self):
    """Verify tune_hyperparameters has a docstring."""
    assert tune_hyperparameters.__doc__ is not None
    assert len(tune_hyperparameters.__doc__) > 0
```

**What it proves:**
- The function is documented
- **Why it matters:** Future developers need to understand what it does

**Invariant protected:**
> All public functions must have docstrings

### Test Class 2: `TestTuningConstants`

#### Test 6: `test_tune_epochs_reasonable`
```python
def test_tune_epochs_reasonable(self):
    """
    Verify tuning epochs is reasonable for hyperparameter search.

    Should be shorter than full training (100 epochs) since we're
    searching for optimal hyperparameters.
    """
    assert DEFAULT_TUNE_EPOCHS < 100
    assert DEFAULT_TUNE_EPOCHS >= 10
```

**What it proves:**
- Tuning epochs (30) is less than full training (100) ✅
- Tuning epochs is at least 10 (not too short) ✅

**Why this matters:**
- **Too high** (e.g., 100): Tuning takes forever, defeats the purpose
- **Too low** (e.g., 5): Can't tell if hyperparameters are good

**Invariant protected:**
> `10 <= DEFAULT_TUNE_EPOCHS < 100`

**Design decision being tested:**
> "30 epochs is enough to evaluate hyperparameters without wasting time"

#### Test 7: `test_tune_iterations_sufficient`
```python
def test_tune_iterations_sufficient(self):
    """
    Verify tuning iterations provides sufficient search space.

    300 iterations is the Ray Tune default and provides good
    coverage for hyperparameter search.
    """
    assert DEFAULT_TUNE_ITERATIONS >= 100
    assert DEFAULT_TUNE_ITERATIONS <= 1000
```

**What it proves:**
- Iterations (300) is at least 100 (sufficient coverage) ✅
- Iterations is at most 1000 (not excessively slow) ✅

**Why this matters:**
- **Too low** (e.g., 10): Won't find optimal hyperparameters
- **Too high** (e.g., 5000): Wastes compute, diminishing returns

**Invariant protected:**
> `100 <= DEFAULT_TUNE_ITERATIONS <= 1000`

### What Invariants We Are Protecting

**Summary of all invariants:**

1. ✅ `DEFAULT_TUNE_EPOCHS == 30` (exact value)
2. ✅ `DEFAULT_TUNE_ITERATIONS == 300` (exact value)
3. ✅ Tuning params > 0 (positive integers)
4. ✅ Function signature includes all required parameters
5. ✅ Function has documentation
6. ✅ `10 <= TUNE_EPOCHS < 100` (reasonable range)
7. ✅ `100 <= TUNE_ITERATIONS <= 1000` (reasonable range)

**Why these specific invariants?**
- **Protect against accidental changes:** Someone might change 30 → 100 without realizing cost
- **Enforce design decisions:** These values were chosen deliberately
- **Catch bugs early:** If signature changes, tests fail immediately
- **Documentation as code:** Tests serve as executable spec

### Why Property-Based Testing Is Used Here

**Property-based test:**
```python
@given(
    epochs=st.integers(min_value=5, max_value=100),
    iterations=st.integers(min_value=50, max_value=1000),
)
def test_tuning_params_valid(self, epochs, iterations):
    assert epochs > 0
    assert iterations > 0
```

**Why use property-based testing for such a simple check?**

1. **Consistency with codebase**
   - This project uses property-based testing throughout
   - Keeps testing style uniform

2. **Future-proofing**
   - Right now the check is simple (`> 0`)
   - Later, we might add: `assert epochs < iterations` or other constraints
   - Property tests make it easy to add more complex invariants

3. **Demonstrates the pattern**
   - Shows how to use `@given` and `st.integers`
   - Educational for maintainers

4. **Automatic edge case testing**
   - Tests min values (5, 50)
   - Tests max values (100, 1000)
   - Tests random values in between
   - All without writing separate test cases

**⚠️ Pedagogical Note:**
> This test is arguably overkill for such a simple check. A regular `assert` would work. But it demonstrates the property-based testing approach used throughout the codebase.

---

## 7. What Is Computationally Expensive Here

### What Parts Are Slow

**Breakdown of computational cost:**

```
Total tuning time = iterations × epochs × time_per_epoch

Example:
- iterations = 300
- epochs = 30  
- time_per_epoch = ~2 minutes (on GPU)

Total = 300 × 30 × 2 min = 18,000 minutes = 300 hours = 12.5 days

On GPU: ~12-15 days
On CPU: ~60-90 days (4-6x slower)
```

**Where the time goes:**

1. **Forward passes** (50% of time)
   - Running images through the neural network
   - 640×640 images through YOLOv8n
   - Batch of 16-32 images at a time

2. **Backward passes** (30% of time)
   - Computing gradients
   - Backpropagation through all layers
   - Updating millions of weights

3. **Data loading** (10% of time)
   - Reading images from disk
   - Applying augmentations
   - Converting to tensors

4. **Validation** (10% of time)
   - Running validation set after each epoch
   - Computing mAP metrics
   - Logging results

**⚠️ Critical:** The expensive part is **training**, not the hyperparameter search logic itself. Ray Tune's overhead is negligible (<1%).

### Why Tuning Is Dangerous to Run Blindly

**Danger 1: Cost explosion**
```python
# Innocent-looking change:
uv run mina-tune --epochs 100 --iterations 1000

# Actual cost:
100 × 1000 = 100,000 epochs of training
On GPU: ~140 days
On CPU: ~2 years
Electricity cost: $$$$$
```

**Danger 2: Disk space**
```
Each trial saves:
- Model weights: ~12 MB
- Training logs: ~1 MB
- Plots: ~5 MB
- Validation images: ~50 MB

Total per trial: ~68 MB
300 trials: ~20 GB
1000 trials: ~68 GB
```

**Danger 3: Forgetting it's running**
```bash
# Start tuning, close laptop
uv run mina-tune

# Come back 3 days later
# Your GPU has been melting for 72 hours
# Battery dead, fans screaming
# Still only 20% done
```

**Danger 4: Opportunity cost**
```
Time spent tuning = Time NOT spent on:
- Collecting more training data (often more valuable)
- Improving data quality
- Building app features
- Testing the model in production
```

**⚠️ Real-world advice:**
> Start with a small test run: `--epochs 10 --iterations 10` (~2 hours). Verify it works before committing to the full 300-iteration run.

### Why These Constants Exist (30 epochs, 300 iterations)

**`DEFAULT_TUNE_EPOCHS = 30`**

**Why 30, not 100?**
- **Diminishing returns:** If a model is doing poorly at epoch 30, it rarely becomes good by epoch 100
- **Speed:** 30 epochs is 3.3x faster than 100
- **Coverage:** Would you rather try 300 configs for 30 epochs each, or 90 configs for 100 epochs each?

**Why 30, not 10?**
- **Stability:** Results at epoch 10 are noisy, hard to compare
- **Convergence:** Some trials need 20-30 epochs to show their true potential
- **Validation:** Need enough epochs for validation metrics to stabilize

**Math check:**
```
Option A: 300 iterations × 30 epochs = 9,000 total epochs
Option B: 90 iterations × 100 epochs = 9,000 total epochs

Same compute budget, but Option A explores 3.3x more configs!
```

**`DEFAULT_TUNE_ITERATIONS = 300`**

**Why 300, not 3000?**
- **Diminishing returns:** After 300 trials, you've likely found a good configuration
- **Bayesian optimization:** Ray Tune gets smarter over time; later trials are better informed
- **Practical limit:** Most people won't wait longer than ~2 weeks for tuning

**Why 300, not 30?**
- **Search space size:** With ~20 hyperparameters, 30 samples barely scratch the surface
- **Variance:** Need multiple trials to distinguish luck from genuinely good configs
- **Ray Tune's sweet spot:** 300 is their recommended default for a reason

**Comparison:**
```
30 iterations:   High chance of missing optimal config
100 iterations:  Decent, but might miss important regions
300 iterations:  ✅ Good balance (recommended)
1000 iterations: Marginal improvement, 3.3x slower
```

**⚠️ Design Decision:**
> These constants are **conservative defaults**. If you have unlimited compute, go bigger. If you're testing, go smaller. But 30/300 is the sweet spot for most users.

---

## 8. Glossary (Only From This PR)

### AdamW
**Full name:** Adam with Decoupled Weight Decay  
**What it is:** An optimizer (algorithm for updating model weights during training)  
**From our code:** Default optimizer in `cli/tune.py`
```python
parser.add_argument("--optimizer", default="AdamW")
```
**Why it matters:** Better generalization than regular Adam due to decoupled weight decay  
**Think of it as:** An improved version of the popular Adam optimizer

### Optimizer
**What it is:** Algorithm that decides HOW to update model weights based on gradients  
**From our code:** Parameter passed to `model.tune(optimizer="AdamW")`  
**Choices available:** SGD, Adam, AdamW, NAdam, RAdam  
**Why it matters:** Different optimizers can give vastly different results  
**Think of it as:** The "learning strategy" the model uses (like different study methods)

### Hyperparameter
**What it is:** A setting configured BEFORE training that controls HOW the model learns  
**From our code:** Learning rate, batch size, augmentation values, etc.  
**Examples in tuning:**
```python
lr0: 0.00853      # Learning rate
batch: 24         # Batch size
hsv_h: 0.0172     # Hue augmentation
warmup_epochs: 2.3  # Warmup duration
```
**Contrast with parameters:** Parameters are learned (weights), hyperparameters are set (configuration)  
**Think of it as:** Knobs you turn to configure the training process

### Ray Tune
**What it is:** A library for distributed hyperparameter tuning  
**From our code:** Used internally by YOLOv8's `.tune()` method  
**What it does:**
- Manages 300 trials
- Samples hyperparameters intelligently
- Tracks results
- Finds best configuration
**We don't call it directly:** YOLOv8 wraps it for us  
**Think of it as:** A smart assistant that runs hundreds of experiments and tells you which settings worked best

### Search Space
**What it is:** The range of values each hyperparameter can take  
**From our code:** Defined inside YOLOv8's `.tune()` (not visible to us)  
**Example:**
```python
lr0: uniform(0.0001, 0.1)      # Any value between 0.0001 and 0.1
batch: choice([8, 16, 24, 32]) # One of these four values
hsv_h: uniform(0.0, 0.03)      # Any value between 0.0 and 0.03
```
**Why it matters:** Defines what configurations are possible  
**Think of it as:** The menu of options the tuning process can choose from

### Trial
**What it is:** A single training run with a specific set of hyperparameters  
**From our code:** We run 300 trials (`iterations=300`)  
**What happens in one trial:**
1. Sample hyperparameters from search space
2. Train model for 30 epochs with those hyperparameters
3. Measure validation mAP
4. Log results
**Storage:** Each trial gets a directory: `runs/tune/trials/trial_001/`  
**Think of it as:** One experiment in a series of 300 experiments

### Iteration
**What it is:** Synonym for "trial" in the context of tuning  
**From our code:** `--iterations 300` means 300 trials  
**Why confusing:** In regular training, "iteration" = one batch. Here, "iteration" = one complete trial.  
**Context matters:**
- `mina-train`: iteration = one batch update
- `mina-tune`: iteration = one complete trial
**Our usage:** We use "iteration" to match Ray Tune's terminology

### Objective
**What it is:** The metric we're trying to maximize (or minimize)  
**From our code:** mAP@50 (mean Average Precision at 50% IoU threshold)  
**Why this metric:** It's the standard for object detection performance  
**In tuning:** Ray Tune tries to find hyperparameters that maximize mAP@50  
**Formula:**
```python
objective = maximize(mAP@50)
```
**Think of it as:** The "score" we're trying to get as high as possible

### Epochs (in tuning context)
**What it is:** Number of complete passes through the training data PER TRIAL  
**From our code:** `DEFAULT_TUNE_EPOCHS = 30`  
**Contrast with regular training:** Full training uses 100 epochs  
**Why less:** 30 epochs is enough to evaluate if hyperparameters are good  
**Total training:** 300 trials × 30 epochs = 9,000 epochs worth of work  
**Think of it as:** How long we "test drive" each set of hyperparameters

### Iterations (in tuning context)
**What it is:** Number of different hyperparameter combinations to try  
**From our code:** `DEFAULT_TUNE_ITERATIONS = 300`  
**What happens:** Train 300 different models with 300 different hyperparameter sets  
**Why 300:** Ray Tune's recommended default for thorough search  
**Trade-off:** More iterations = better results but slower  
**Think of it as:** How many different "recipes" we test before picking the best one

---

## 🚨 Critical Warnings & Best Practices

### DO NOT Run Tuning Without Understanding The Cost

```bash
# ⛔ DANGER: This will take WEEKS
uv run mina-tune --epochs 100 --iterations 1000

# ⛔ DANGER: This will fill your disk
uv run mina-tune --iterations 5000

# ⛔ DANGER: This will melt your laptop
uv run mina-tune --device cpu  # (for days)
```

**Safe approach:**
```bash
# ✅ Test run first (2-3 hours)
uv run mina-tune --epochs 10 --iterations 10

# ✅ Verify it works, then commit
uv run mina-tune --epochs 30 --iterations 300

# ✅ Use GPU if available
uv run mina-tune --device 0
```

### DO Start Small

**Recommended progression:**
1. **Test run** (2 hours): `--epochs 10 --iterations 10`
2. **Verify results make sense**
3. **Small run** (1 day): `--epochs 20 --iterations 50`
4. **Check if improvements are happening**
5. **Full run** (12-15 days): `--epochs 30 --iterations 300`

### DO Monitor Disk Space

```bash
# Check space before starting
df -h

# Clean up old runs
rm -rf runs/tune/trials  # (keep best_hyperparameters.yaml)
```

### DO Use The Best Hyperparameters For Final Training

**After tuning completes:**
```bash
# ⛔ WRONG: Ignore tuning results
uv run mina-train --epochs 100

# ✅ CORRECT: Use tuned hyperparameters
uv run mina-train --config runs/tune/best_hyperparameters.yaml --epochs 100
```

### DON'T Assume Tuning Will Always Help

**Tuning helps when:**
- ✅ You have a decent-sized dataset (500+ images)
- ✅ Model is underperforming with default settings
- ✅ You've exhausted other improvements (data quality, augmentation)

**Tuning might NOT help when:**
- ❌ Dataset is tiny (<100 images) — get more data instead
- ❌ Dataset is noisy/mislabeled — fix labels instead
- ❌ Model architecture is wrong — try different model

**Rule of thumb:**
> If your model is getting <70% mAP, fix your data first. If it's getting 70-85%, tuning might help. If it's getting >90%, diminishing returns.

---

## Final Thoughts

**What PR #11 gives you:**
> A systematic way to find optimal hyperparameters instead of guessing

**The cost:**
> ~12-15 days of GPU time (or significantly longer on CPU)

**The benefit:**
> Potentially 3-5% improvement in mAP, which could mean better disease detection in production

**When to use it:**
1. After you've collected good training data
2. Before final model training for production
3. When you need to squeeze every bit of performance
4. When you have time and compute budget

**When NOT to use it:**
1. During initial prototyping
2. When data quality is poor
3. When you're on a tight deadline
4. When compute budget is limited

**Remember:**
> Hyperparameter tuning is **optimization**, not **magic**. It won't fix bad data, wrong model architecture, or fundamental issues. But when you have a solid foundation, it can take you from "good" to "great".

---

**Happy tuning! 🎛️**
