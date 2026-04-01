# PWA Offline Support Gaps And Fix Suggestions

## Scope

This document describes the remaining gaps between the current web PWA and a true offline-capable disease detection experience.

It assumes the current phase is:

- online PWA first
- full offline support later
- model auto-update later

## Current State

The web app now has the basic ingredients for offline support:

1. App shell is cached by the generated service worker
2. Scan history is stored locally in IndexedDB
3. The ONNX model is served from the same origin at `/model/best.onnx`
4. The generated production build precaches the ONNX model and the ONNX Runtime WASM file

Even with that progress, full offline support is still not complete.

## Main Gaps

### 1. No explicit offline readiness state

The app does not clearly tell the user whether the device is ready for offline analysis.

Current problem:

- the user cannot tell whether the model, worker assets, and runtime files are already stored locally
- the app only shows a generic loading state during analysis

Suggested fix:

1. Add a model readiness status in settings or onboarding
2. Show clear states such as:
   - `Not downloaded`
   - `Downloading`
   - `Ready for offline use`
   - `Update available`
3. Persist the readiness state locally

### 2. No deliberate offline warm-up flow

Right now offline readiness depends on the user naturally loading the right assets during usage.

Current problem:

- first offline use may still fail if the needed assets were not fully cached before the device went offline
- there is no explicit warm-up step that verifies all required assets are available

Suggested fix:

1. Add a startup or settings action to warm up offline assets
2. Explicitly verify availability of:
   - `/model/best.onnx`
   - ONNX Runtime WASM asset
   - worker bundle
   - app shell routes
3. Mark the app as offline-ready only after the warm-up succeeds

### 3. No versioned model strategy

The current model path is a single fixed filename:

- `/model/best.onnx`

Current problem:

- replacing the file in place makes updates hard to reason about
- old caches and new deployments can conflict
- rollback safety is weak

Suggested fix:

1. Move to versioned model filenames, for example:
   - `/model/best-v1.onnx`
   - `/model/best-v2.onnx`
2. Track the active version in a small manifest file
3. Keep the old model until the new model is fully downloaded and verified

### 4. No model manifest or metadata contract in the web app

There is no lightweight runtime metadata file that tells the web app what model version should be used.

Current problem:

- the app cannot check whether a newer model exists
- the app cannot distinguish app version from model version

Suggested fix:

1. Add a same-origin metadata file such as:
   - `/model/manifest.json`
2. Include fields like:
   - model version
   - filename
   - sha256
   - release date
   - compatibility version
3. Make the web app read this manifest before switching models

### 5. No explicit update flow for connected users

When a user comes back online later, there is no controlled mechanism to refresh the model.

Current problem:

- users may stay on an old model indefinitely
- there is no safe background update sequence

Suggested fix:

1. On reconnect or app launch, fetch the model manifest
2. If a newer compatible model exists:
   - download it in the background
   - verify hash/integrity
   - store it
   - switch to it only after success
3. Show an unobtrusive `Model updated` or `Update available` message

### 6. No integrity verification for downloaded model artifacts

The app currently trusts the model file if it is reachable.

Current problem:

- no checksum validation
- no protection against partial or corrupted downloads

Suggested fix:

1. Include a hash in the model manifest
2. Verify the downloaded file before activating it
3. Keep the previous model if verification fails

### 7. No dedicated storage strategy for model assets

At the moment the model is handled through static asset serving and service worker caching.

Current problem:

- there is no app-controlled model lifecycle
- there is no explicit delete/swap behavior
- debugging cache state is harder than it needs to be

Suggested fix:

Use one of these approaches:

1. Keep model assets in Cache Storage but manage them through a model manager layer
2. Or store model binaries in IndexedDB for stricter app-level control

Recommendation:

- for a large binary like the ONNX model, keep using HTTP caching/Cache Storage for delivery
- add an app-level model manager that knows which version should be active

### 8. No offline-specific error messaging

Offline failures will currently look too similar to normal model-load failures.

Current problem:

- users do not know whether the issue is missing offline assets, missing network, or a bad deployment

Suggested fix:

Differentiate these cases in the UI:

1. No internet and offline assets not prepared
2. Model missing from local storage
3. Model update failed, using previous version
4. Runtime asset missing or corrupted

### 9. No automated tests for offline behavior

Current tests only cover validation logic and history storage.

Current problem:

- no automated proof that install, cache, model warm-up, and offline inference work

Suggested fix:

Add browser-level tests for:

1. PWA installability
2. App shell offline navigation
3. Successful model warm-up
4. Offline inference after warm-up
5. Update flow when a newer model manifest is available

### 10. Results flow is not fully reload-safe

Some transient UI state still lives in memory only.

Current problem:

- a page refresh can lose the current result view even though history is stored

Suggested fix:

1. Persist the latest scan/session reference
2. Rehydrate the latest result from IndexedDB or route state
3. Make the results page recover gracefully after reload

## Recommended Offline Architecture

When the project is ready for the full offline phase, the recommended approach is:

1. Keep the web app, worker, WASM, and icons precached by the service worker
2. Use a versioned same-origin model manifest
3. Use versioned same-origin model filenames
4. Add a model manager in the app that:
   - checks readiness
   - warms up assets
   - verifies versions
   - activates new models safely
5. Add user-facing offline/update status

## Suggested Implementation Order

### Phase A: Offline readiness UX

1. Add offline/model readiness status
2. Add warm-up check
3. Add better offline-specific errors

### Phase B: Versioned model delivery

1. Add model manifest
2. Add versioned model filenames
3. Add app logic to resolve the active model

### Phase C: Safe updates

1. Check for manifest changes on reconnect
2. Download newer model in background
3. Verify checksum
4. Swap model only after success

### Phase D: Hardening

1. Add offline E2E tests
2. Add recovery/rollback behavior
3. Make results flow reload-safe

## Summary

The current web PWA is much closer to offline support than before, but it still does not provide a complete offline product experience.

The biggest missing pieces are:

1. explicit offline readiness
2. deliberate warm-up
3. versioned model delivery
4. safe update logic
5. automated offline testing
