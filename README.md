# Fish Care (FishCareYOLO)

Offline-first, on-device fish disease detection for mobile.

Fish Care is a React Native (Expo) app that uses a YOLOv8 model (exported to TFLite) to detect common fish diseases from a camera capture or gallery image. The goal is **instant, private, offline** detection with clear results + recommended next actions.

This repo is currently in the **initiation phase**: the overall plan is defined, but most core features are not implemented yet.

## What It Will Do

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

- `expo-app/`: Expo (React Native) app workspace (Expo Router + NativeWind, etc.)
- `python-model/`: Python workspace for model training/export (scaffolded)
## Roadmap

High-level phases:

1. Python training pipeline (ultralytics YOLOv8n, export to int8 TFLite)
2. Expo foundation (types, JSON serialization, property tests)
3. Storage service (AsyncStorage) + history sorting
4. Inference service (react-native-fast-tflite) + confidence filtering
5. UI screens: camera → results → history → disease info + onboarding

## Getting Started (Development)

### Mobile app (`expo-app/`)

```bash
cd expo-app
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

### Python model pipeline (`python-model/`)

`python-model/pyproject.toml` exists, but dependencies/scripts are not wired up yet.

When it’s ready, the intended workflow is via `uv`:

```bash
cd python-model
uv sync
```

For now, consider it a placeholder for:

- training a YOLOv8n model with `ultralytics`
- exporting an int8 `.tflite` model for bundling into the app
## Testing (Planned)

Planned tests include both unit tests and property-based tests:

- Expo app: `fast-check` for correctness properties (serialization round-trip, sorting, filtering)
- Python pipeline: `hypothesis` for export/inference equivalence

The test suite is not implemented yet.
## Notes / Disclaimer

This project provides informational guidance only and is not a substitute for veterinary advice. When in doubt, consult a qualified aquatic veterinarian.

