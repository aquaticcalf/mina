# Model Loading Strategy

The ONNX models are **no longer bundled** in this public directory.

Instead of bundling 18MB of models into every deployment, the app dynamically loads the models from GitHub Releases:
- `https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/best.onnx`
- `https://github.com/fishcareyolo/fishcareyolo/releases/download/prod/fish_gate.onnx`

### How it works
1. **Production (Vercel)**: A Vercel Edge Function (`web/api/model-proxy.ts`) proxies the download from GitHub to bypass CORS restrictions.
2. **Local Development**: Vite's built-in `server.proxy` (`web/vite.config.ts`) handles the same `/api/model-proxy` endpoint and routes it to GitHub Releases, bypassing browser CORS locally.

The frontend natively fetches from `/api/model-proxy?file=...` in all environments.
