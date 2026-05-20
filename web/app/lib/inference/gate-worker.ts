import * as ort from "onnxruntime-web"

// In development, Vite natively resolves the ONNX modules from node_modules.
// In production, vite-plugin-static-copy copies the WASM files to the build root (/).
if (import.meta.env.PROD) {
  ort.env.wasm.wasmPaths = "/"
}
export interface GateRequest {
  id: string
  type: "load" | "run" | "release"
  data?: number[] | string | { url: string; force?: boolean }
}

export interface GateResponse {
  id: string
  type: "result" | "ready" | "error" | "loading"
  data?: { isFish: boolean; confidence: number }
  error?: string
  progress?: number
}

const GATE_SIZE = 224
const GATE_THRESHOLD = 0.45

// ImageNet normalisation constants (must match training preprocessing)
const IMAGENET_MEAN = [0.485, 0.456, 0.406]
const IMAGENET_STD = [0.229, 0.224, 0.225]

const ctx = self as unknown as Worker

let session: ort.InferenceSession | null = null

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

/**
 * Normalise a 224×224 RGBA ImageData into a CHW Float32Array using ImageNet stats.
 * Output layout: [R-plane (224*224), G-plane (224*224), B-plane (224*224)]
 */
function preprocessImage(imageData: ImageData): Float32Array {
  const data = new Float32Array(3 * GATE_SIZE * GATE_SIZE)
  const pixels = imageData.data

  for (let i = 0; i < GATE_SIZE * GATE_SIZE; i++) {
    const r = pixels[i * 4] / 255
    const g = pixels[i * 4 + 1] / 255
    const b = pixels[i * 4 + 2] / 255

    data[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    data[i + GATE_SIZE * GATE_SIZE] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    data[i + 2 * GATE_SIZE * GATE_SIZE] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
  }

  return data
}

async function loadModel(modelUrl: string, forceUpdate: boolean = false): Promise<void> {
  console.log(`[GateWorker] Starting model load from: ${modelUrl}${forceUpdate ? " (Force Update)" : ""}`)
  ctx.postMessage({ type: "loading", progress: 0 } as GateResponse)

  try {
    ctx.postMessage({ type: "loading", progress: 10 } as GateResponse)

    const cache = await caches.open("mina-models-v1")
    let response = await cache.match(modelUrl)

    if (!response || forceUpdate) {
      console.log("[GateWorker] Downloading model from network...")
      response = await fetch(modelUrl, {
        cache: "no-store",
        headers: { "Cache-Control": "no-cache" }
      })

      if (!response.ok) {
        throw new Error(
          `Gate model download failed (${response.status} ${response.statusText}) from ${modelUrl}`,
        )
      }

      await cache.put(modelUrl, response.clone())
    } else {
      console.log("[GateWorker] Loading model from browser cache...")
    }

    const modelBuffer = await response.arrayBuffer()

    console.log("[GateWorker] Loading model via ONNX Runtime...")
    ctx.postMessage({ type: "loading", progress: 60 } as GateResponse)

    const startTime = Date.now()
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
    })

    const loadTime = ((Date.now() - startTime) / 1000).toFixed(2)
    console.log(`[GateWorker] Model loaded successfully in ${loadTime}s`)
    console.log("[GateWorker] Input names:", session.inputNames)
    console.log("[GateWorker] Output names:", session.outputNames)

    ctx.postMessage({ type: "ready" } as GateResponse)
  } catch (e) {
    console.error("[GateWorker] Model load failed:", e)
    ctx.postMessage({
      type: "error",
      error: e instanceof Error ? e.message : "Failed to load gate model",
    } as GateResponse)
  }
}

async function runGate(id: string, imageData: ImageData): Promise<void> {
  if (!session) {
    ctx.postMessage({
      id,
      type: "error",
      error: "Gate model not loaded",
    } as GateResponse)
    return
  }

  try {
    const inputData = preprocessImage(imageData)
    const inputTensor = new ort.Tensor("float32", inputData, [1, 3, GATE_SIZE, GATE_SIZE])

    const inferenceStart = performance.now()
    const results = await session.run({ images: inputTensor })
    const inferenceMs = (performance.now() - inferenceStart).toFixed(2)

    // Output is a raw logit with shape [1, 1]
    const outputTensor = results["output"] ?? results[Object.keys(results)[0]]
    const logit = (outputTensor.data as Float32Array)[0]
    const confidence = sigmoid(logit)
    const isFish = confidence > GATE_THRESHOLD

    console.log(
      `[GateWorker] isFish=${isFish} confidence=${confidence.toFixed(4)} (${inferenceMs}ms)`,
    )

    ctx.postMessage({
      id,
      type: "result",
      data: { isFish, confidence },
    } as GateResponse)
  } catch (e) {
    console.error("[GateWorker] Inference failed:", e)
    ctx.postMessage({
      id,
      type: "error",
      error: e instanceof Error ? e.message : "Gate inference failed",
    } as GateResponse)
  }
}

ctx.onmessage = async (event: MessageEvent<GateRequest>) => {
  const { type, id, data } = event.data

  if (type === "load") {
    if (typeof data === "string") {
      await loadModel(data)
    } else {
      const payload = data as { url: string; force?: boolean }
      await loadModel(payload.url, payload.force)
    }
  } else if (type === "run") {
    const imageData = new ImageData(
      new Uint8ClampedArray(data as number[]),
      GATE_SIZE,
      GATE_SIZE,
    )
    await runGate(id, imageData)
  } else if (type === "release") {
    session = null
    ctx.postMessage({ id, type: "ready" } as GateResponse)
  }
}
