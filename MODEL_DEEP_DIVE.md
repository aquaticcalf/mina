# 📘 MINA Model Deep Dive

> **Auto-generated Documentation for the Fish Disease Detection Model Pipeline**
> 
> This document explains the complete ML pipeline in `model/` — from raw images to a mobile-ready model.

---

## 1. Big Picture Architecture

### What Problem Does This Model Solve?

Fish farmers and aquarium enthusiasts need to **quickly identify diseases** in their fish. Waiting for an expert or sending images to a cloud service is too slow and unreliable. This model solves that by:

1. Running **entirely on your phone** (no internet needed)
2. Detecting 5 disease classes in **under 2 seconds**
3. Providing **precise bounding boxes** showing exactly where the problem is

**Think of it like this:** Instead of calling a plumber every time you suspect a leak, you get a leak detector that works instantly, offline, in your hands.

### What is YOLO Doing Here?

**YOLO** = "You Only Look Once" — it's an object detection algorithm.

**In traditional terms:**
- **Classification**: "This image contains a fish" ✅/❌
- **Object Detection**: "This image has a fish at coordinates (x, y, w, h) and it has bacterial infection with 87% confidence"

**Why YOLO specifically?**
- **Speed**: Designed for real-time detection (perfect for mobile)
- **Accuracy**: Detects multiple objects in one pass
- **Small models available**: YOLOv8n ("nano") is optimized for mobile devices

**What it does in one forward pass:**
1. Takes a 640x640 image
2. Runs it through a neural network
3. Outputs: "I see 2 fish, one healthy (90% conf) at box A, one with white_tail (75% conf) at box B"

### Why This Architecture?

```
┌─────────────────────────────────────────────────────────────┐
│                    DESIGN DECISIONS                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Python (Training)           TypeScript (Mobile App)         │
│                                                               │
│  ┌─────────────┐            ┌─────────────┐                 │
│  │ YOLOv8n     │  Export    │  TFLite     │                 │
│  │ (PyTorch)   │ ---------> │  (int8)     │  Bundled        │
│  │ ~6MB        │            │  ~3MB       │  in app         │
│  └─────────────┘            └─────────────┘                 │
│                                                               │
│  Why PyTorch?                Why TFLite?                     │
│  - Training ecosystem        - Mobile optimized              │
│  - Ultralytics library       - Small size (~3MB)             │
│  - Easy export               - Fast on-device inference      │
│                              - Works offline                 │
│                                                               │
│  Why int8 quantization?      Why NMS disabled?               │
│  - 50% smaller model         - onnx2tf has TopK issues       │
│  - Faster inference          - We implement NMS in the app   │
│  - Minimal accuracy loss     - More control, less bugs       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Critical Design Decision:**
> The model is **trained once** in Python, **exported to TFLite**, then **bundled with the app**.
> Users NEVER download the model separately. It's like shipping a calculator app with the calculator logic baked in.

---

## 2. Folder & File Walkthrough

```
model/
├── mina/                    # Core Python package
│   ├── core/                # Shared utilities and types
│   │   ├── __init__.py
│   │   ├── constants.py     # Disease classes, thresholds, paths
│   │   ├── types.py         # BoundingBox, Detection (Python types)
│   │   ├── dataset.py       # YAML file generation
│   │   └── model.py         # Model loading utilities
│   │
│   ├── train.py             # Training logic
│   ├── export.py            # TFLite export logic
│   ├── inference.py         # Inference + detection conversion
│   ├── evaluate.py          # Test set evaluation
│   └── dataset.py           # Roboflow download
│
├── cli/                     # Command-line interface scripts
│   ├── train.py             # CLI: mina-train
│   ├── export.py            # CLI: mina-export
│   ├── infer.py             # CLI: mina-infer
│   ├── evaluate.py          # CLI: mina-evaluate
│   └── download.py          # CLI: mina-download
│
├── tests/                   # Property-based tests
│   ├── test_training.py
│   ├── test_inference.py    # Property 2, 3, 5
│   └── test_export.py       # Property 8 (TFLite equivalence)
│
├── pyproject.toml           # Python dependencies + CLI entry points
└── README.md                # Quick start guide
```

### Key Files Explained

#### `mina/core/constants.py`
**What it does:** Central source of truth for configuration.

```python
DISEASE_CLASSES = [
    "bacterial_infection",
    "fungal_infection", 
    "healthy",
    "parasite",
    "white_tail"
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.3
DEFAULT_IMAGE_SIZE = 640
```

**Why it exists:**
- **Single source of truth**: Change the threshold once, it updates everywhere
- **Prevents bugs**: Class order MUST match model training. If someone adds a class out of order, everything breaks.

**⚠️ CRITICAL:** The order of `DISEASE_CLASSES` is **locked to the model's class indices**. Index 0 = bacterial_infection, Index 1 = fungal_infection, etc. **NEVER reorder this list** without retraining the model.

#### `mina/core/types.py`
**What it does:** Defines the Python data structures that match the app's TypeScript types.

```python
class BoundingBox(NamedTuple):
    x: float       # Top-left x (0.0 to 1.0)
    y: float       # Top-left y (0.0 to 1.0)
    width: float   # Box width (0.0 to 1.0)
    height: float  # Box height (0.0 to 1.0)

class Detection(NamedTuple):
    id: str
    disease_class: str       # Must be in DISEASE_CLASSES
    confidence: float        # 0.0 to 1.0
    bounding_box: BoundingBox
```

**Why it exists:**
- **Contract with the app**: The TypeScript app expects this exact structure
- **Validation**: Each type has a `.validate()` method to catch bugs early
- **Immutability**: Using `NamedTuple` prevents accidental mutations

**How it connects:**
- `inference.py` creates `Detection` objects from YOLO outputs
- Tests validate that every detection matches this structure
- The app expects JSON that mirrors this structure

#### `mina/core/dataset.py`
**What it does:** Generates the `data.yaml` file that YOLO needs for training.

```python
def create_data_yaml(data_dir: Path) -> Path:
    yaml_content = f"""
path: {data_dir.absolute()}
train: images/train
val: images/val

names:
  0: bacterial_infection
  1: fungal_infection
  2: healthy
  3: parasite
  4: white_tail

nc: 5
"""
```

**Why it exists:**
- YOLO expects a YAML config file to know where images/labels are
- Hardcodes the class mapping (index → name)
- Used by `train.py` to pass dataset info to YOLO

#### `mina/core/model.py`
**What it does:** Helper functions to find and load trained models.

```python
def find_best_weights() -> Path | None:
    # Searches runs/detect/* for most recent best.pt
    
def find_tflite_weights() -> tuple[Path | None, Path | None]:
    # Finds both PyTorch and TFLite versions
```

**Why it exists:**
- **Auto-discovery**: You don't have to remember where you saved your model
- **Testing**: Tests can auto-load the latest model without hardcoded paths
- **Convenience**: `mina-export` can just say "export the latest training run"

---

## 3. Training Pipeline (`train.py`)

### What is Training?

**Simple terms:** You show the model thousands of labeled fish images, and it learns to recognize patterns.

**Technical terms:** Training adjusts the neural network's weights using backpropagation to minimize the loss function (how wrong the model's predictions are).

**Analogy:** Like teaching a kid to identify animals:
- Show them 1000 pictures: "This is a dog, this is a cat, this is a bird"
- Test them: "What's this?" → "Dog!" → "Correct! ✅"
- Repeat until they get good at it

### How Dataset is Loaded

```python
def train(epochs=100, batch=16, imgsz=640, ...):
    data_yaml = get_data_yaml_path()  # Finds data.yaml
    model = YOLO("yolov8n.pt")        # Load pretrained YOLOv8 nano
    
    results = model.train(
        data=str(data_yaml),          # Where to find images/labels
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        # ...
    )
```

**What happens under the hood:**

1. **YOLO reads `data.yaml`**: "Ah, training images are in `data/images/train/`"
2. **Loads images + labels**: For each image, there's a `.txt` file with bounding box annotations
3. **Batches them**: Groups 16 images together to process at once (batch=16)
4. **Resizes to 640x640**: All images must be the same size for the neural network

**Label format (YOLO):**
```
# bacterial_infection_001.txt
0 0.5 0.3 0.4 0.35    # class_id cx cy w h (all normalized 0-1)
```
This means: "Class 0 (bacterial_infection) centered at (0.5, 0.3) with width 0.4, height 0.35"

### What These Terms Mean

#### `epochs=100`
- **One epoch** = The model sees every training image once
- **100 epochs** = The model cycles through the entire dataset 100 times
- **Why repeat?** The model gets better each time it sees the data (up to a point)

**Analogy:** Reading a textbook once vs. reading it 100 times. You remember more each pass.

#### `imgsz=640`
- Input image size (640x640 pixels)
- YOLO requires square images
- **Larger = more detail** but slower training/inference
- **640 is the sweet spot** for mobile: fast enough, accurate enough

**⚠️ CRITICAL:** The exported model is **locked to 640x640**. If you change this, you must retrain AND re-export.

#### `batch=16`
- Process 16 images at once (in parallel on GPU)
- **Larger batch** = faster training, more GPU memory needed
- **Smaller batch** = slower, less memory

**Analogy:** Washing dishes one at a time vs. loading 16 into a dishwasher.

#### `weights="yolov8n.pt"`
- Starting point for training (pretrained model)
- **"yolov8n.pt"** is pretrained on COCO dataset (everyday objects)
- We **fine-tune** it for fish diseases

**Why start from pretrained?**
- The model already knows basic vision (edges, shapes, textures)
- We just teach it to recognize fish-specific diseases
- Training from scratch would take 10x longer and need way more data

#### `device="0"` or `device="cpu"`
- **"0"** = Use first GPU (CUDA)
- **"cpu"** = Use CPU (slower, no GPU needed)
- Auto-detected in our code

#### `patience=20`
- **Early stopping**: If validation loss doesn't improve for 20 epochs, stop training
- Prevents overfitting (model memorizing training data instead of learning patterns)

### Training Step-by-Step

```
┌───────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                           │
├───────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load pretrained YOLOv8n weights                            │
│     ↓                                                           │
│  2. For each epoch (0 to 100):                                 │
│     ├─ Load batch of 16 images                                 │
│     ├─ Resize to 640x640                                       │
│     ├─ Apply augmentations (flip, rotate, color jitter)        │
│     ├─ Forward pass: Model predicts bounding boxes             │
│     ├─ Calculate loss: How wrong were the predictions?         │
│     ├─ Backward pass: Adjust weights to reduce loss            │
│     └─ Repeat for all batches                                  │
│     ↓                                                           │
│  3. After each epoch:                                          │
│     ├─ Run validation (on val/ images)                         │
│     ├─ Calculate mAP (mean Average Precision)                  │
│     ├─ If best mAP so far, save as "best.pt"                   │
│     └─ If no improvement for 20 epochs, stop early             │
│     ↓                                                           │
│  4. Training complete!                                         │
│     └─ Best model saved to runs/detect/fish_disease/weights/   │
│                                                                 │
└───────────────────────────────────────────────────────────────┘
```

### Augmentations Explained

```python
# From train.py
hsv_h=0.015      # Vary hue (color) slightly
hsv_s=0.7        # Vary saturation (color intensity)
hsv_v=0.4        # Vary brightness
degrees=10.0     # Rotate up to 10 degrees
translate=0.1    # Shift image by up to 10%
scale=0.5        # Zoom in/out randomly
flipud=0.5       # 50% chance of vertical flip
fliplr=0.5       # 50% chance of horizontal flip
mosaic=1.0       # Mosaic augmentation (combines 4 images)
mixup=0.1        # 10% chance of blending two images
```

**Why augment?**
- **Prevents overfitting**: Model learns to handle variations
- **Simulates real-world conditions**: Different lighting, angles, orientations
- **Makes the model robust**: Works even if user takes a tilted photo

**Analogy:** Teaching a kid to recognize dogs by showing them:
- Dogs from different angles
- Dogs in different lighting
- Photos that are slightly blurry or rotated

### What Gets Saved

After training, you get:

```
runs/detect/fish_disease/
├── weights/
│   ├── best.pt      # Best model (highest validation mAP)
│   └── last.pt      # Model from last epoch
├── results.png      # Training/validation loss curves
├── confusion_matrix.png
└── args.yaml        # Hyperparameters used
```

**⚠️ CRITICAL:** We **only export `best.pt`**, not `last.pt`. "Best" means highest validation accuracy, not most recent.

---

## 4. Inference Pipeline (`inference.py`)

### What is Inference?

**Simple terms:** Running the trained model on a new image to get predictions.

**Technical terms:** A forward pass through the neural network that converts input pixels to output detections.

**Training vs Inference:**
- **Training**: Model learns (weights change)
- **Inference**: Model predicts (weights frozen, no learning)

**Analogy:**
- **Training** = Studying for an exam
- **Inference** = Taking the exam

### How Images Become Detections

```python
def run_inference(model, image_path, min_confidence=0.3):
    # 1. Load image
    results = model(str(image_path), verbose=False)
    
    # 2. Convert raw outputs to Detection objects
    detections = convert_to_detections(results, min_confidence)
    
    # 3. Return sorted, filtered detections
    return detections
```

**Step-by-step:**

```
┌────────────────────────────────────────────────────────────┐
│              INFERENCE PIPELINE                             │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  Image File (fish.jpg)                                      │
│    ↓                                                         │
│  1. Load & Preprocess                                       │
│     - Read image pixels                                     │
│     - Resize to 640x640                                     │
│     - Normalize pixel values (0-255 → 0-1)                  │
│     - Convert to tensor [1, 3, 640, 640]                    │
│    ↓                                                         │
│  2. Run Model (Forward Pass)                                │
│     - Feed tensor through neural network                    │
│     - Get raw predictions [8400 anchors x (4+1+5)]          │
│       • 4 bbox coords (cx, cy, w, h)                        │
│       • 1 objectness score                                  │
│       • 5 class probabilities                               │
│    ↓                                                         │
│  3. Post-Processing                                         │
│     - Filter by confidence (keep >= 0.3)                    │
│     - Convert YOLO format to Detection objects              │
│     - Sort by confidence (highest first)                    │
│    ↓                                                         │
│  Detection[] (Python objects)                               │
│                                                              │
└────────────────────────────────────────────────────────────┘
```

### What Raw Outputs Look Like

YOLO outputs a **massive array** of predictions:

```
Shape: [8400, 9]

Where:
- 8400 = Number of "anchor boxes" (possible detection locations)
- 9 = [cx, cy, w, h, objectness, p0, p1, p2, p3, p4]
  - cx, cy, w, h: Center x/y, width, height (normalized 0-1)
  - objectness: Is there an object here? (0-1)
  - p0-p4: Probability for each of 5 classes (0-1)
```

**Example raw output (one anchor):**
```
[0.52, 0.31, 0.35, 0.28, 0.92, 0.05, 0.87, 0.02, 0.01, 0.05]
 └─cx  └─cy  └─w   └─h   └obj  └─p0  └─p1  └─p2  └─p3  └─p4

Interpretation:
- Bounding box centered at (0.52, 0.31) with size (0.35, 0.28)
- 92% confident there's an object here
- Class probabilities: [5%, 87%, 2%, 1%, 5%]
- → Most likely class 1 (fungal_infection) with 87%
- → Final confidence = objectness × class_prob = 0.92 × 0.87 = 0.80
```

### How Detections Are Constructed

```python
def convert_to_detections(results, min_confidence=0.3):
    detections = []
    
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            confidence = float(boxes.conf[i].item())
            
            # Filter by threshold
            if confidence < min_confidence:
                continue
            
            class_id = int(boxes.cls[i].item())
            disease_class = DISEASE_CLASSES[class_id]
            
            # Get bounding box (YOLO gives xyxy, convert to xywh)
            xyxy = boxes.xyxyn[i].tolist()  # Normalized
            x1, y1, x2, y2 = xyxy
            
            bbox = BoundingBox(
                x=x1,           # Top-left x
                y=y1,           # Top-left y
                width=x2 - x1,
                height=y2 - y1
            )
            
            detection = Detection(
                id=f"det_{len(detections):03d}",
                disease_class=disease_class,
                confidence=confidence,
                bounding_box=bbox
            )
            detections.append(detection)
    
    # Sort by confidence (highest first)
    detections.sort(key=lambda d: -d.confidence)
    return detections
```

**Key conversions:**

1. **Class ID → Disease Name:**
   ```python
   class_id = 1  # From model output
   disease_class = DISEASE_CLASSES[1]  # → "fungal_infection"
   ```

2. **YOLO xyxy → Our xywh:**
   ```
   YOLO gives: [x1, y1, x2, y2] (top-left and bottom-right corners)
   We want:    [x, y, w, h]     (top-left corner + dimensions)
   
   Conversion:
   x = x1
   y = y1
   w = x2 - x1
   h = y2 - y1
   ```

3. **Normalized coordinates:**
   ```
   All values are 0.0 to 1.0 (percentage of image dimensions)
   To get pixel coordinates:
     pixel_x = x * image_width
     pixel_y = y * image_height
   ```

### Confidence Filtering (Property 2)

```python
if confidence < 0.3:
    continue  # Skip low-confidence detections
```

**Why filter?**
- **Reduces false positives**: Low-confidence detections are usually wrong
- **User experience**: Only show confident predictions
- **Performance**: Fewer detections to process

**⚠️ WARNING:** Changing this threshold affects what users see. Too high = miss real diseases. Too low = show false alarms.

### Detection Sorting (Property 3)

```python
detections.sort(key=lambda d: -d.confidence)
```

**Why sort?**
- **User sees most important first**: Highest confidence at top
- **Consistency**: Same ordering across Python and TypeScript
- **Testing**: Property tests verify this always happens

---

## 5. Export Pipeline (`export.py`)

### What is TFLite?

**TensorFlow Lite** = A version of TensorFlow optimized for mobile and embedded devices.

**Think of it like this:**
- **PyTorch (.pt file)** = Full desktop app (big, feature-rich)
- **TFLite (.tflite file)** = Mobile app version (small, fast, essential features only)

**Why convert?**
- PyTorch models are **6-10 MB**, TFLite models are **~3 MB**
- PyTorch needs the full PyTorch runtime (100+ MB), TFLite needs minimal runtime (~1 MB)
- TFLite is optimized for mobile CPUs/GPUs

### What is Quantization?

**Simple explanation:** Compressing the model by using smaller numbers.

**Technical explanation:**
- Neural networks use **float32** (32-bit floating point) by default
- Quantization converts to **int8** (8-bit integers)
- This makes the model **4x smaller** and **2-3x faster**

**Analogy:**
```
Original (float32):  3.14159265359
Quantized (int8):    3

You lose some precision, but for most tasks it doesn't matter.
```

### What int8 Means

```
float32: -3.402823e38 to +3.402823e38 (huge range, high precision)
int8:    -128 to +127 (tiny range, low precision)

How it works:
1. Find min/max values in model weights: [-5.2, +8.7]
2. Map that range to [-128, 127]
3. Store as integers instead of floats

Space savings:
- float32: 4 bytes per weight
- int8:    1 byte per weight
- → 75% size reduction!
```

**Does accuracy suffer?**
- Typically **1-2% mAP drop**
- For our use case, going from 89% to 87% mAP is acceptable
- The speed/size benefits far outweigh the tiny accuracy loss

### Why NMS is Disabled

```python
export_path = model.export(
    format="tflite",
    int8=int8,
    imgsz=imgsz,
    nms=False,  # ⚠️ CRITICAL: Do not change!
)
```

**What is NMS (Non-Maximum Suppression)?**

When YOLO detects objects, it often predicts **multiple overlapping boxes** for the same object:

```
Image of a fish:
┌───────────┐
│  ┌──────┐ │  Box A: 92% confident (bacterial_infection)
│  │      │ │  Box B: 87% confident (bacterial_infection)  
│  └──────┘ │  Box C: 81% confident (bacterial_infection)
└───────────┘
All three boxes are for the same fish!
```

**NMS removes duplicates:**
1. Keep the box with highest confidence (A: 92%)
2. Remove any boxes that overlap >50% with A
3. Result: Only box A remains ✅

**Why we disable it in export:**

```python
# From export.py comments:
# nms=False because onnx2tf has issues with TopK operations
# used in YOLO NMS. NMS should be handled in the mobile app instead.
```

**The problem:**
- YOLO's NMS uses a "TopK" operation (select top K boxes)
- The export tool (onnx2tf) **sometimes fails** when converting TopK to TFLite
- It's a known bug in the conversion pipeline

**The solution:**
- Export the "raw" model without NMS
- Implement NMS ourselves in the TypeScript app
- See `app/lib/model/inference.ts` → `applyNMS()` function

**⚠️ CRITICAL:** If you enable `nms=True`, the export **may fail silently** or produce a broken model. Always keep it `False`.

### What Exactly is Exported

```
Input:  best.pt (PyTorch weights, ~6 MB)
Output: best_full_integer_quant.tflite (~3 MB)

The TFLite file contains:
✅ Model architecture (layers, connections)
✅ Quantized weights (int8)
✅ Input/output tensor specifications
✅ Optimization metadata

It does NOT contain:
❌ Training data
❌ Augmentation pipeline
❌ Optimizer state
❌ NMS logic (we handle that in the app)
```

### Export Step-by-Step

```python
def export_tflite(weights_path, int8=True, imgsz=640, nms=False):
    # 1. Load PyTorch model
    model = YOLO(str(weights_path))
    
    # 2. Export to TFLite
    export_path = model.export(
        format="tflite",    # Target format
        int8=int8,          # Apply quantization
        imgsz=imgsz,        # Input size
        simplify=True,      # Optimize graph
        nms=nms,            # NMS (disabled)
    )
    
    # 3. Return path to .tflite file
    return export_path
```

**What happens internally:**
```
PyTorch (.pt)
    ↓
  Export to ONNX (intermediate format)
    ↓
  Convert ONNX to TensorFlow SavedModel
    ↓
  Quantize to int8 (calibration on sample data)
    ↓
  Convert to TFLite
    ↓
TFLite (.tflite) ✅
```

---

## 6. Model Types & Data Structures

### Python Types (Server/Training)

```python
# mina/core/types.py

class BoundingBox(NamedTuple):
    """
    Normalized bounding box coordinates.
    All values are 0.0 to 1.0 (percentage of image dimensions).
    """
    x: float       # Left edge (0.0 = far left, 1.0 = far right)
    y: float       # Top edge (0.0 = top, 1.0 = bottom)
    width: float   # Box width (0.0 to 1.0)
    height: float  # Box height (0.0 to 1.0)

class Detection(NamedTuple):
    """A single disease detection."""
    id: str                  # Unique ID (e.g., "det_001")
    disease_class: str       # One of DISEASE_CLASSES
    confidence: float        # 0.0 to 1.0
    bounding_box: BoundingBox
```

### TypeScript Types (Mobile App)

```typescript
// app/lib/model/types.ts

interface BoundingBox {
    x: number       // Same as Python
    y: number
    width: number
    height: number
}

interface Detection {
    id: string
    diseaseClass: DiseaseClass  // Type-safe enum
    confidence: number
    boundingBox: BoundingBox
}

interface DetectionSession {
    id: string
    imageUri: string           // file:///path/to/image.jpg
    detections: Detection[]
    timestamp: number          // Unix ms
}
```

### How They Map to YOLO Outputs

```
┌─────────────────────────────────────────────────────────────┐
│              YOLO → Our Types Mapping                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  YOLO Output:                                                │
│  [cx, cy, w, h, objectness, p0, p1, p2, p3, p4]             │
│   │   │   │  │      │        └──────┬──────┘                │
│   │   │   │  │      │               │                       │
│   │   │   │  │      │          Class probs                  │
│   │   │   │  │      │                                       │
│   │   │   │  │   Objectness score                           │
│   │   │   │  │                                               │
│   │   │   │  └─ height (normalized)                         │
│   │   │   └──── width (normalized)                          │
│   │   └──────── center y (normalized)                       │
│   └──────────── center x (normalized)                       │
│                                                               │
│  ↓ Convert to ↓                                              │
│                                                               │
│  Our Detection:                                              │
│  {                                                            │
│    id: "det_001",                                            │
│    diseaseClass: DISEASE_CLASSES[argmax(p0..p4)],           │
│    confidence: objectness * max(p0..p4),                     │
│    boundingBox: {                                            │
│      x: cx - w/2,    // Convert center to top-left          │
│      y: cy - h/2,                                            │
│      width: w,                                               │
│      height: h                                               │
│    }                                                          │
│  }                                                            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### JSON Serialization

```python
# Python can't directly send objects to TypeScript
# So we serialize to JSON:

{
  "id": "session_123",
  "imageUri": "file:///data/image.jpg",
  "detections": [
    {
      "id": "det_001",
      "diseaseClass": "bacterial_infection",
      "confidence": 0.87,
      "boundingBox": {
        "x": 0.25,
        "y": 0.30,
        "width": 0.40,
        "height": 0.35
      }
    }
  ],
  "timestamp": 1705852800000
}
```

**Why this matters:**
- TypeScript can `JSON.parse()` this directly
- Field names MUST match exactly (camelCase in both)
- Types are validated on both sides

---

## 7. Testing Strategy

### Property-Based Testing Explained

**Traditional testing:**
```python
def test_confidence_filter():
    detections = [Detection(conf=0.9), Detection(conf=0.2)]
    filtered = filter_by_confidence(detections, threshold=0.3)
    assert len(filtered) == 1  # Only 0.9 should remain
```

**Property-based testing:**
```python
@given(confidences=st.lists(st.floats(0.0, 1.0)))  # Generate random data
def test_confidence_filter_property(confidences):
    # Test with ANY random list of confidences
    detections = [Detection(conf=c) for c in confidences]
    filtered = filter_by_confidence(detections, threshold=0.3)
    
    # Verify the PROPERTY holds:
    for d in filtered:
        assert d.confidence >= 0.3  # ✅ This MUST be true
```

**Why it's better:**
- Automatically tests **hundreds of random cases**
- Finds edge cases you'd never think of
- Proves the property holds **for all inputs**, not just your examples

### Property 5: Detection Result Structure

```python
# test_inference.py

@given(
    disease_class=st.sampled_from(DISEASE_CLASSES),
    confidence=st.floats(0.0, 1.0),
    x=st.floats(0.0, 0.5),
    y=st.floats(0.0, 0.5),
    width=st.floats(0.1, 0.5),
    height=st.floats(0.1, 0.5),
)
def test_valid_detection_passes_validation(...):
    detection = Detection(
        id="test_001",
        disease_class=disease_class,
        confidence=confidence,
        bounding_box=BoundingBox(x, y, width, height)
    )
    
    errors = detection.validate()
    assert len(errors) == 0
```

**What this guarantees:**
- **For ANY valid inputs**, a Detection can be created
- **Disease class** is always one of the 5 valid classes
- **Confidence** is always 0.0 to 1.0
- **Bounding box** coordinates are always 0.0 to 1.0

**Why this matters:**
If this test passes, we KNOW the app can never receive invalid data from the model.

### Property 2: Confidence Filtering

```python
@given(confidences=st.lists(st.floats(0.0, 1.0)))
def test_confidence_filtering(confidences):
    detections = [Detection(conf=c) for c in confidences]
    filtered = filter_by_confidence(detections, threshold=0.3)
    
    # Property: All kept detections >= threshold
    for d in filtered:
        assert d.confidence >= 0.3
    
    # Property: All removed detections < threshold
    removed = set(detections) - set(filtered)
    for d in removed:
        assert d.confidence < 0.3
```

**What this guarantees:**
- **No false positives**: Nothing below 0.3 sneaks through
- **No false negatives**: Nothing >= 0.3 gets filtered out
- **Works for ANY list of confidences**

### Property 3: Detection Sorting

```python
@given(confidences=st.lists(st.floats(0.0, 1.0), min_size=2))
def test_detection_sorting(confidences):
    detections = [Detection(conf=c) for c in confidences]
    sorted_detections = sorted(detections, key=lambda d: -d.confidence)
    
    # Property: Descending order
    for i in range(len(sorted_detections) - 1):
        assert sorted_detections[i].confidence >= sorted_detections[i+1].confidence
```

**What this guarantees:**
The list is ALWAYS sorted highest-to-lowest, regardless of input order.

### Property 8: TFLite Export Equivalence

```python
# test_export.py

@given(
    width=st.integers(320, 640),
    height=st.integers(320, 640),
    seed=st.integers(0, 2**32 - 1)
)
def test_detection_equivalence(width, height, seed):
    # Create random test image
    test_image = create_test_image(width, height, seed)
    
    # Run both models
    pt_results = pytorch_model(test_image)
    tflite_results = tflite_model(test_image)
    
    # Extract top detection from each
    pt_top = pt_results[0]
    tflite_top = tflite_results[0]
    
    # Property: Same class
    assert pt_top.class_id == tflite_top.class_id
    
    # Property: Confidence within 0.05
    conf_diff = abs(pt_top.confidence - tflite_top.confidence)
    assert conf_diff <= 0.05
```

**What this guarantees:**
- **PyTorch and TFLite models produce the same results**
- **Quantization doesn't break the model**
- **Export process is correct**

**Why 0.05 tolerance?**
- Quantization introduces small rounding errors
- 5% difference is acceptable (89% vs 84% confidence doesn't change user experience)
- Anything larger indicates a bug in export

**⚠️ CRITICAL:** If this test fails, the TFLite model is broken. DO NOT ship it.

---

## 8. Full End-to-End Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                    END-TO-END DATA FLOW                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DATASET                                                         │
│     └─ Roboflow (1000+ labeled fish images)                        │
│        ↓ mina-download                                              │
│     └─ data/images/train/ + labels/                                │
│        └─ data/images/val/ + labels/                               │
│                                                                      │
│  2. TRAINING (Python)                                               │
│     └─ mina-train --epochs 100 --batch 16                          │
│        ├─ Load YOLOv8n pretrained weights                          │
│        ├─ Fine-tune on fish disease data                           │
│        └─ Save best.pt (highest validation mAP)                    │
│        ↓                                                             │
│     └─ runs/detect/fish_disease/weights/best.pt (~6 MB)            │
│                                                                      │
│  3. EXPORT (Python)                                                 │
│     └─ mina-export                                                  │
│        ├─ Load best.pt                                              │
│        ├─ Convert PyTorch → ONNX → TFLite                          │
│        ├─ Apply int8 quantization                                  │
│        └─ Save best_full_integer_quant.tflite (~3 MB)              │
│        ↓                                                             │
│     └─ best_full_integer_quant.tflite                              │
│                                                                      │
│  4. RELEASE (Manual)                                                │
│     └─ Upload .tflite to GitHub Releases                           │
│        ├─ Tag: "dev" or "prod"                                      │
│        ├─ Asset: best_full_integer_quant.tflite                    │
│        └─ Release notes: Updated date, model metrics               │
│        ↓                                                             │
│     └─ https://github.com/.../releases/download/dev/*.tflite       │
│                                                                      │
│  5. APP DOWNLOAD (TypeScript)                                       │
│     └─ ModelProvider initializes on app launch                     │
│        ├─ Check if model exists locally                            │
│        ├─ If not, download from GitHub release                     │
│        ├─ Save to device storage (~3 MB)                           │
│        └─ Load into TFLite runtime                                 │
│        ↓                                                             │
│     └─ Model ready for inference ✅                                │
│                                                                      │
│  6. USER FLOW (Mobile App)                                          │
│     └─ User opens camera                                            │
│        ↓                                                             │
│     └─ User taps "Capture"                                          │
│        ↓                                                             │
│     └─ Image saved: file:///data/fish_123.jpg                      │
│        ↓                                                             │
│     └─ InferenceService.runInference(imageUri)                     │
│        ├─ Load image as tensor [1, 3, 640, 640]                    │
│        ├─ Run TFLite model (forward pass)                          │
│        ├─ Get raw output [8400, 9]                                 │
│        ├─ Parse detections                                          │
│        ├─ Apply NMS (remove duplicates)                            │
│        ├─ Filter confidence >= 0.3                                 │
│        └─ Sort by confidence                                        │
│        ↓                                                             │
│     └─ Detection[] objects                                          │
│        ↓                                                             │
│     └─ Create DetectionSession                                      │
│        ├─ id: "session_abc123"                                      │
│        ├─ imageUri: "file:///data/fish_123.jpg"                    │
│        ├─ detections: [...]                                         │
│        └─ timestamp: 1705852800000                                  │
│        ↓                                                             │
│     └─ Save to AsyncStorage (JSON)                                 │
│        ↓                                                             │
│     └─ Show results screen                                          │
│        ├─ Image with bounding boxes overlaid                       │
│        ├─ List of detections (sorted by confidence)                │
│        └─ Disease info (symptoms, treatments)                      │
│                                                                      │
│  7. HISTORY (Mobile App)                                            │
│     └─ User opens history screen                                    │
│        ↓                                                             │
│     └─ Load all sessions from AsyncStorage                          │
│        ↓                                                             │
│     └─ Sort by timestamp (newest first)                             │
│        ↓                                                             │
│     └─ Display list of past scans                                   │
│                                                                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 9. Integration With The App

### Exactly How the App Consumes This Model

The TypeScript app expects a specific **input format** and **output format**.

#### Input (What App Sends to Model)

```typescript
// app/lib/model/inference.ts

const imageUri = "file:///data/user/0/.../fish.jpg"

// 1. Load image
const image = await loadImage(imageUri)

// 2. Resize to 640x640
const resized = resizeImage(image, 640, 640)

// 3. Convert to tensor
const input = imageToTensor(resized)  // Float32Array[1,3,640,640]

// 4. Run inference
const output = model.run(input)  // Float32Array[8400,9]
```

**⚠️ CRITICAL:** The model is **frozen at 640x640 input**. If you change `imgsz` during training/export, you must update the app's inference code.

#### Output (What Model Returns to App)

```typescript
// Raw output from TFLite
Float32Array[8400, 9]
// Each row: [cx, cy, w, h, objectness, p0, p1, p2, p3, p4]

// App parses this into:
interface RawDetection {
    classIndex: number      // 0-4
    confidence: number      // objectness × max(p0..p4)
    boundingBox: BoundingBox
}
```

Then applies:
1. **NMS** (remove duplicates)
2. **Confidence filtering** (>= 0.3)
3. **Sorting** (highest first)
4. **Conversion to Detection objects**

### The Contract Between Python and TypeScript

```
┌──────────────────────────────────────────────────────────────┐
│                      THE CONTRACT                             │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  Python Side (Training/Export):                               │
│  ✅ Must export 640x640 input size                           │
│  ✅ Must export 5 classes (in exact order)                   │
│  ✅ Must export without NMS                                  │
│  ✅ Must use int8 quantization                               │
│  ✅ Output shape: [8400, 9]                                  │
│                                                                │
│  TypeScript Side (App):                                       │
│  ✅ Must resize images to 640x640                            │
│  ✅ Must implement NMS                                        │
│  ✅ Must filter confidence >= 0.3                            │
│  ✅ Must map class indices 0-4 to DISEASE_CLASSES            │
│  ✅ Must normalize bbox coords to 0-1                        │
│                                                                │
│  Shared Constants:                                            │
│  DISEASE_CLASSES = [                                          │
│    0: "bacterial_infection",                                  │
│    1: "fungal_infection",                                     │
│    2: "healthy",                                              │
│    3: "parasite",                                             │
│    4: "white_tail"                                            │
│  ]                                                             │
│                                                                │
│  CONFIDENCE_THRESHOLD = 0.3                                   │
│  IMAGE_SIZE = 640                                             │
│  IOU_THRESHOLD = 0.45 (for NMS)                              │
│                                                                │
└──────────────────────────────────────────────────────────────┘
```

### What Assumptions Must NEVER Break

1. **Class Order is Sacred**
   ```python
   # ⛔ NEVER DO THIS:
   DISEASE_CLASSES = [
       "healthy",              # Changed from index 2 to 0
       "bacterial_infection",  # Changed from index 0 to 1
       ...
   ]
   # This would make the model predict wrong classes!
   ```

2. **Image Size is Fixed**
   ```python
   # If you export with imgsz=320:
   export_tflite(weights, imgsz=320)
   
   # The app MUST also use 320:
   # app/lib/model/inference.ts
   const IMAGE_SIZE = 320  // Must match export!
   ```

3. **NMS is Always Disabled in Model**
   ```python
   # ⛔ NEVER ENABLE THIS:
   export_tflite(weights, nms=True)  # Will break!
   
   # ✅ Always keep disabled:
   export_tflite(weights, nms=False)
   ```

4. **Bounding Box Format**
   ```typescript
   // App expects top-left (x,y) + (w,h)
   // NOT center (cx,cy) + (w,h)
   
   // Model outputs center format, app converts:
   const x = cx - w/2
   const y = cy - h/2
   ```

5. **Confidence Threshold**
   ```python
   # Python: DEFAULT_CONFIDENCE_THRESHOLD = 0.3
   # TypeScript: CONFIDENCE_THRESHOLD = 0.3
   # These MUST match!
   ```

---

## 10. Glossary of ML Terms (From This Project)

### Anchor Boxes
**What:** Predefined boxes at different positions and scales.  
**In our code:** YOLO generates 8400 anchor boxes across the 640x640 image.  
**Why:** Instead of checking every pixel, we check 8400 strategic locations.  
**Think of it as:** A grid of possible detection locations.

### Augmentation
**What:** Artificially modifying training images.  
**In our code:** `hsv_h=0.015, flipud=0.5, mosaic=1.0, etc.`  
**Why:** Makes the model robust to variations (lighting, angle, etc.).  
**Think of it as:** Teaching the model to recognize fish even if the photo is rotated or poorly lit.

### Batch
**What:** Number of images processed together.  
**In our code:** `batch=16` means 16 images at once.  
**Why:** Faster training, better GPU utilization.  
**Think of it as:** Assembly line processing vs. one-at-a-time.

### Bounding Box
**What:** Rectangle around a detected object.  
**In our code:** `{ x: 0.25, y: 0.30, width: 0.40, height: 0.35 }`  
**Format:** Top-left corner (x,y) + dimensions (w,h), all normalized 0-1.  
**Think of it as:** A crop rectangle in an image editor.

### Class
**What:** Category the model can detect.  
**In our code:** 5 classes (bacterial_infection, fungal_infection, healthy, parasite, white_tail).  
**Why:** The model learns to distinguish between these categories.  
**Think of it as:** Labels you can tag objects with.

### Confidence
**What:** How sure the model is about a prediction.  
**In our code:** `confidence: 0.87` means 87% sure.  
**Range:** 0.0 (totally unsure) to 1.0 (completely certain).  
**Think of it as:** Probability that the detection is correct.

### Epoch
**What:** One complete pass through the entire training dataset.  
**In our code:** `epochs=100` means we cycle through all images 100 times.  
**Why:** Model gets better each epoch (up to a point).  
**Think of it as:** Reading a textbook once vs. 100 times.

### Export
**What:** Converting a trained model to a different format.  
**In our code:** PyTorch (.pt) → TFLite (.tflite).  
**Why:** PyTorch is for training, TFLite is for mobile inference.  
**Think of it as:** Compiling source code to an executable.

### Fine-Tuning
**What:** Starting from a pretrained model and training on your data.  
**In our code:** Start with `yolov8n.pt` (pretrained on COCO), fine-tune on fish diseases.  
**Why:** Faster training, better results with less data.  
**Think of it as:** Teaching a multilingual person a new language vs. teaching someone who only knows one.

### Inference
**What:** Running a trained model on new data to get predictions.  
**In our code:** `model(image_path)` returns detections.  
**Not training:** Weights are frozen, no learning happens.  
**Think of it as:** Using a calculator (inference) vs. learning math (training).

### int8 Quantization
**What:** Converting 32-bit floats to 8-bit integers.  
**In our code:** `int8=True` in export.  
**Why:** 4x smaller model, 2-3x faster inference.  
**Trade-off:** ~1-2% accuracy loss.  
**Think of it as:** Compressing an image from PNG to JPEG.

### IoU (Intersection over Union)
**What:** Measure of how much two boxes overlap.  
**In our code:** Used in NMS, threshold = 0.45.  
**Formula:** `area(overlap) / area(union)`  
**Range:** 0.0 (no overlap) to 1.0 (perfect overlap).  
**Think of it as:** Venn diagram overlap percentage.

### Loss
**What:** How wrong the model's predictions are.  
**In our code:** YOLO uses a combined loss (bbox + class + objectness).  
**Why:** Training adjusts weights to minimize loss.  
**Think of it as:** Error score (lower is better).

### mAP (mean Average Precision)
**What:** Standard metric for object detection accuracy.  
**In our code:** `mAP50 = 0.89` means 89% accuracy at 50% IoU threshold.  
**Why:** Single number to compare model versions.  
**Think of it as:** Overall grade on an exam.

### NMS (Non-Maximum Suppression)
**What:** Removing duplicate/overlapping detections.  
**In our code:** Implemented in TypeScript (`applyNMS()`), not in model.  
**How:** Keep highest-confidence box, remove overlapping boxes.  
**Think of it as:** Deduplicating search results.

### Objectness
**What:** Probability that a box contains any object.  
**In our code:** Part of YOLO output, combined with class probability.  
**Range:** 0.0 to 1.0.  
**Think of it as:** "Is there something here?" before asking "What is it?"

### Pretrained
**What:** Model already trained on a large dataset.  
**In our code:** `yolov8n.pt` is pretrained on COCO (80 everyday object classes).  
**Why:** Better starting point than random weights.  
**Think of it as:** Hiring an experienced developer vs. training a junior from scratch.

### Quantization
**What:** Reducing numerical precision to save space.  
**In our code:** float32 → int8.  
**Why:** Smaller model, faster inference, minimal accuracy loss.  
**Think of it as:** Rounding numbers (3.14159 → 3).

### Tensor
**What:** Multi-dimensional array (like a matrix but any number of dimensions).  
**In our code:** Image = `[1, 3, 640, 640]` (batch, channels, height, width).  
**Why:** Neural networks operate on tensors.  
**Think of it as:** A fancy array.

### TFLite (TensorFlow Lite)
**What:** Mobile-optimized version of TensorFlow.  
**In our code:** The `.tflite` file we export.  
**Why:** Designed for phones (small, fast, efficient).  
**Think of it as:** Mobile app vs. desktop app.

### Threshold
**What:** Cutoff value for filtering.  
**In our code:** `confidence >= 0.3` means only keep detections above 30% confidence.  
**Why:** Reduce false positives.  
**Think of it as:** Spam filter threshold.

### Validation Set
**What:** Held-out data used to evaluate model during training.  
**In our code:** `data/images/val/` (separate from training data).  
**Why:** Detect overfitting, choose best model.  
**Think of it as:** Practice exam before the real exam.

### Weights
**What:** Learned parameters of the neural network.  
**In our code:** Stored in `best.pt` (PyTorch) or `.tflite` (TFLite).  
**Why:** The weights ARE the model (architecture + weights = trained model).  
**Think of it as:** The "brain" of the model.

### YOLO (You Only Look Once)
**What:** Real-time object detection algorithm.  
**In our code:** YOLOv8n (version 8, nano variant).  
**Why:** Fast enough for mobile, accurate enough for our use case.  
**Think of it as:** A specific type of AI for finding objects in images.

### YOLOv8n
**What:** Nano variant of YOLOv8 (smallest, fastest).  
**In our code:** The base model we fine-tune.  
**Why:** Optimized for mobile/edge devices.  
**Sizes:** nano < small < medium < large < xlarge (we use nano).  
**Think of it as:** Economy car (small, efficient) vs. luxury SUV (big, powerful).

---

## 🚨 Critical Warnings & Performance Notes

### Things That Will Break Everything

1. **Changing `DISEASE_CLASSES` order**
   ```python
   # ⛔ DO NOT REORDER:
   DISEASE_CLASSES = ["healthy", "bacterial_infection", ...]  # WRONG!
   
   # ✅ ALWAYS KEEP:
   DISEASE_CLASSES = ["bacterial_infection", "fungal_infection", ...]
   ```
   **Why:** Class indices are baked into the trained model. Reordering would make the model predict wrong classes.

2. **Enabling NMS in export**
   ```python
   # ⛔ DO NOT DO:
   export_tflite(weights, nms=True)  # May fail or produce broken model
   
   # ✅ ALWAYS USE:
   export_tflite(weights, nms=False)
   ```
   **Why:** onnx2tf has known issues with TopK operations. The app handles NMS instead.

3. **Changing image size inconsistently**
   ```python
   # If you export with:
   export_tflite(weights, imgsz=320)
   
   # You MUST update app:
   // app/lib/model/inference.ts
   const IMAGE_SIZE = 320  // Must match!
   ```
   **Why:** The model's input layer is fixed at export time.

4. **Training without validation data**
   ```
   data/
     images/train/  ✅
     images/val/    ⚠️ REQUIRED!
   ```
   **Why:** Without validation, you can't detect overfitting or choose the best model.

### Performance-Sensitive Code

1. **Inference speed depends on:**
   - Image size: 640x640 is the sweet spot
   - Batch size: Always 1 for mobile (no batching)
   - Quantization: int8 is 2-3x faster than float32
   - Device: GPU/NPU much faster than CPU

2. **Model size matters:**
   - Too big: App download size increases, users complain
   - Too small: Accuracy suffers
   - Our sweet spot: ~3 MB (int8 quantized)

3. **Confidence threshold trade-offs:**
   - Too low (< 0.2): Many false positives
   - Too high (> 0.5): Miss real diseases
   - Our choice: 0.3 (balanced)

### When to Retrain

You MUST retrain if:
- You get more labeled data (accuracy will improve)
- You add/remove disease classes (CRITICAL: breaks everything)
- You change image size (640 → 320 requires retrain)
- You switch YOLO versions (v8 → v9)

You DON'T need to retrain if:
- You change confidence threshold (inference-time parameter)
- You update the app UI
- You fix bugs in the app's NMS implementation

---

## Final Notes

This model pipeline is **the product's core value**. Everything else (UI, storage, etc.) exists to deliver the model's predictions to users. 

**Key takeaways:**
1. **Training** happens once (or when you get better data)
2. **Export** converts the model for mobile use
3. **The app** bundles the model and runs inference on-device
4. **The contract** between Python and TypeScript is sacred

**If you remember nothing else, remember this:**
> The model is trained in Python, exported to TFLite, and consumed by TypeScript. The `DISEASE_CLASSES` order and image size (640x640) are **immutable contracts** that both sides must honor.

Good luck! 🐟
