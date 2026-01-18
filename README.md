# MINA

> Offline-first, on-device fish disease detection for mobile.

MINA is a React Native (Expo) app that uses a YOLOv8 model (exported to TFLite) to detect common fish diseases from a camera capture or gallery image. The goal is **instant, private, offline** detection with clear results + recommended next actions.

This repo is currently in the **initiation phase**: the overall plan is defined, but most core features are not implemented yet.

## What It Does

- ✓ YOLOv8n model trained to detect 5 classes: bacterial_infection, fungal_infection, healthy, parasite, white_tail
- Capture a photo using the device camera (or select from gallery)
- Run YOLOv8 inference **on-device** (no network required)
- Display bounding boxes + disease names with confidence scores
- Save scans locally and show a history view
- Provide disease descriptions, symptoms, and treatment suggestions

## Key Goals

- Offline-first: the model is bundled with the app
- Fast: target under ~2 seconds from capture → results on supported devices
- Clear results: bounding boxes + sorted detections + “healthy” state when none

## Repository Layout

- `app/`: Expo (React Native) app workspace (Expo Router + NativeWind, etc.)
- `model/`: Python workspace with trained YOLOv8n model + TFLite export

## Roadmap

High-level phases (✓ = done):

1. ✓ Python training pipeline (ultralytics YOLOv8n, export to int8 TFLite)
2. Expo foundation (types, JSON serialization, property tests)
3. Storage service (AsyncStorage) + history sorting
4. Inference service (react-native-fast-tflite) + confidence filtering
5. UI screens: camera → results → history → disease info + onboarding

## Getting Started (Development)

### Mobile app (`app/`)

```bash
cd app
bun install
bun run dev
```

Other useful scripts:

```bash
bun run android
```

```bash
bun run ios
```

```bash
bun run web
```

```bash
bun run fix
```

(`bun run fix` formats via Biome.)

### Python model pipeline (`model/`)

The model is trained and released! You can download the TFLite model directly:

- **Dev release (latest):** [best_full_integer_quant.tflite](https://github.com/fishcareyolo/fishcareyolo/releases/download/dev/best_full_integer_quant.tflite) (~3MB, Int8 quantized)
- **Prod release (stable):** [best_full_integer_quant.tflite](https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/best_full_integer_quant.tflite)

For details on the model or to retrain, see [`model/`](model/) and the [Colab notebook](https://colab.research.google.com/github/fishcareyolo/fishcareyolo/blob/main/model/mina.ipynb).

## Testing

### Model tests (`model/tests/`)

The Python pipeline includes property-based tests using `hypothesis`:

- **Training config:** disease classes, dataset YAML, training params
- **Detection structure (Property 5):** valid classes, confidence 0-1, bbox coordinates
- **Confidence filtering (Property 2):** ≥0.3 threshold applied correctly
- **Detection sorting (Property 3):** descending by confidence
- **TFLite equivalence (Property 8):** PyTorch ↔ TFLite outputs match within 0.05 tolerance

```bash
cd model
pytest tests/ -v
```

### App tests (planned)

Expo app tests use `bun:test` for correctness properties:
- Serialization round-trip
- Detection sorting/filtering

## Notes / Disclaimer

This project provides informational guidance only and is not a substitute for veterinary advice. When in doubt, consult a qualified aquatic veterinarian.

