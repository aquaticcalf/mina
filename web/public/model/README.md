The web app expects the production ONNX model here as:

- `web/public/model/best.onnx`

The web app serves this file at:

- `/model/best.onnx`

This same-origin path is required for production browser inference because fetching GitHub release assets directly from the client fails due to CORS.

If the model is missing, the app will fail during analysis with a model download/load error.
