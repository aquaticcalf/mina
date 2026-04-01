# Phase 1 Plan: Make The PWA Work Reliably Online

## Goal

Make the installed web PWA work correctly while the user is connected to the internet.

This phase does not attempt to complete full offline inference, background model updates, or model version management. It only fixes the current production failure and makes the online path stable.

## Current Failure

The production web app loads the ONNX model from:

- `https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/best.onnx`

That works in development only because Vite proxies `/model/...` requests to GitHub. In production and in the installed PWA, the browser worker fetches GitHub directly. GitHub Releases does not return the CORS headers required for browser access from the app origin, so the request is blocked.

## Root Cause

1. The browser fetches the model directly from GitHub Releases in production.
2. `onnxruntime-web` uses normal browser fetch semantics in the worker.
3. GitHub Release assets do not provide permissive CORS headers for this usage.
4. Workbox then reports `no-response` because the underlying network request already failed.

## Phase 1 Fix

Serve the ONNX model from the same origin as the web app.

### Required changes

1. Host the model at a same-origin path such as `/model/best.onnx`
2. Change the inference service to use the same-origin model URL in both development and production
3. Update PWA caching rules to target same-origin model assets instead of GitHub
4. Add deployment guidance so the web build always includes `web/public/model/best.onnx`
5. Improve model-load error messaging enough to diagnose missing assets or broken deployment

## Non-goals For This Phase

- Full offline-first support
- Automatic model update checks
- Background model downloads
- Versioned model manifest design
- Model integrity verification

## Expected Result

After this phase:

1. The web app and installed PWA should load the ONNX model successfully while online
2. Development and production should use the same model URL strategy
3. Browser CORS should no longer block inference

## Remaining Work After Phase 1

The later offline/update phase should cover:

1. Precache or otherwise guarantee local availability of model and WASM assets
2. Versioned model releases and migration logic
3. Update UX when a newer model becomes available
4. Explicit install/download instructions in the app UI
