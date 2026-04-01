import * as ort from "onnxruntime-web"

export interface InferenceRequest {
  id: string
  type: "load" | "run" | "release"
  data?: number[] | string
  dims?: number[]
}

export interface InferenceResponse {
  id: string
  type: "result" | "ready" | "error" | "loading"
  data?: InferenceResult[]
  error?: string
  progress?: number
}

export interface InferenceResult {
  class: string
  confidence: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
}

const DISEASE_CLASSES = [
  "bacterial_infection",
  "fungal_infection",
  "healthy",
  "parasite",
  "white_tail",
]

const IMAGE_SIZE = 640
const CONFIDENCE_THRESHOLD = 0.3
const IOU_THRESHOLD = 0.6

// Use self to reference the worker global scope
const ctx = self as unknown as Worker

let session: ort.InferenceSession | null = null

function xywh2xyxy(boxes: number[]): number[] {
  const result = []
  for (let i = 0; i < boxes.length; i += 4) {
    const x = boxes[i]
    const y = boxes[i + 1]
    const w = boxes[i + 2]
    const h = boxes[i + 3]
    result.push(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
  }
  return result
}

function computeIoU(box1: number[], box2: number[]): number {
  const x1 = Math.max(box1[0], box2[0])
  const y1 = Math.max(box1[1], box2[1])
  const x2 = Math.min(box1[2], box2[2])
  const y2 = Math.min(box1[3], box2[3])

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
  const area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
  const union = area1 + area2 - intersection

  return intersection / (union + 1e-6)
}

function nms(boxes: number[][], scores: number[], iouThreshold: number): number[] {
  const indices = scores
    .map((score, idx) => ({ score, idx }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.idx)

  const keep: number[] = []
  while (indices.length > 0) {
    const current = indices.shift()!
    keep.push(current)

    const remaining: number[] = []
    for (const idx of indices) {
      const iou = computeIoU(boxes[current], boxes[idx])
      if (iou < iouThreshold) {
        remaining.push(idx)
      }
    }
    indices.length = 0
    indices.push(...remaining)
  }

  return keep
}

function nmsByClass(
  boxes: number[][],
  scores: number[],
  classIds: number[],
  iouThreshold: number,
): number[] {
  const perClass = new Map<number, number[]>()

  for (let i = 0; i < classIds.length; i++) {
    const cls = classIds[i]
    const indices = perClass.get(cls)
    if (indices) {
      indices.push(i)
    } else {
      perClass.set(cls, [i])
    }
  }

  const keep: number[] = []
  for (const indices of perClass.values()) {
    const classBoxes = indices.map((idx) => boxes[idx])
    const classScores = indices.map((idx) => scores[idx])
    const classKeep = nms(classBoxes, classScores, iouThreshold)
    keep.push(...classKeep.map((localIdx) => indices[localIdx]))
  }

  keep.sort((a, b) => scores[b] - scores[a])
  return keep
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x))
}

function preprocessImage(imageData: ImageData): Float32Array {
  const data = new Float32Array(3 * IMAGE_SIZE * IMAGE_SIZE)
  const pixels = imageData.data

  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    const r = pixels[i * 4] / 255
    const g = pixels[i * 4 + 1] / 255
    const b = pixels[i * 4 + 2] / 255

    data[i] = r
    data[i + IMAGE_SIZE * IMAGE_SIZE] = g
    data[i + 2 * IMAGE_SIZE * IMAGE_SIZE] = b
  }

  return data
}

function postprocessOutput(output: Float32Array, dims: number[]): InferenceResult[] {
  const numClasses = DISEASE_CLASSES.length

  let numAnchors = 0
  let featureStride = 0
  let classOffset = 0
  let hasObjectness = false
  let readValue: (anchorIdx: number, featureIdx: number) => number

  // YOLO exports can differ by layout:
  // - [1, features, anchors] (common for YOLOv8 ONNX)
  // - [1, anchors, features]
  // And features may be:
  // - 4 + classes (no objectness)
  // - 5 + classes (with objectness)
  if (dims.length === 3 && (dims[1] === numClasses + 4 || dims[1] === numClasses + 5)) {
    featureStride = dims[1]
    numAnchors = dims[2]
    hasObjectness = featureStride === numClasses + 5
    classOffset = hasObjectness ? 5 : 4
    readValue = (anchorIdx: number, featureIdx: number) =>
      output[featureIdx * numAnchors + anchorIdx]
  } else if (dims.length === 3 && (dims[2] === numClasses + 4 || dims[2] === numClasses + 5)) {
    featureStride = dims[2]
    numAnchors = dims[1]
    hasObjectness = featureStride === numClasses + 5
    classOffset = hasObjectness ? 5 : 4
    readValue = (anchorIdx: number, featureIdx: number) =>
      output[anchorIdx * featureStride + featureIdx]
  } else {
    throw new Error(`Unsupported output shape: [${dims.join(", ")}]`)
  }

  const boxes: number[][] = []
  const scores: number[] = []
  const classIds: number[] = []

  for (let i = 0; i < numAnchors; i++) {
    const cx = readValue(i, 0)
    const cy = readValue(i, 1)
    const w = readValue(i, 2)
    const h = readValue(i, 3)

    if (w <= 0 || h <= 0) {
      continue
    }

    const rawObjConf = hasObjectness ? readValue(i, 4) : 1
    const objConf =
      hasObjectness && (rawObjConf < 0 || rawObjConf > 1) ? sigmoid(rawObjConf) : rawObjConf

    let maxScore = 0
    let classId = 0
    for (let c = 0; c < numClasses; c++) {
      const rawClassConf = readValue(i, classOffset + c)
      const classConf = rawClassConf < 0 || rawClassConf > 1 ? sigmoid(rawClassConf) : rawClassConf
      const score = objConf * classConf
      if (score > maxScore) {
        maxScore = score
        classId = c
      }
    }

    if (maxScore > CONFIDENCE_THRESHOLD) {
      boxes.push([cx, cy, w, h])
      scores.push(maxScore)
      classIds.push(classId)
    }
  }

  if (boxes.length === 0) return []

  const xyxyBoxes = xywh2xyxy(boxes.flat())
  const boxArray: number[][] = []
  for (let i = 0; i < xyxyBoxes.length; i += 4) {
    boxArray.push(xyxyBoxes.slice(i, i + 4))
  }

  const keepIndices = nmsByClass(boxArray, scores, classIds, IOU_THRESHOLD)

  const maxCoord = boxArray.reduce((m, b) => Math.max(m, b[0], b[1], b[2], b[3]), 0)
  const coordScale = maxCoord <= 2 ? 1 : IMAGE_SIZE
  const coordMax = coordScale === 1 ? 1 : IMAGE_SIZE

  const results: InferenceResult[] = []
  for (const idx of keepIndices) {
    const [rawX1, rawY1, rawX2, rawY2] = boxArray[idx]
    const x1 = Math.max(0, Math.min(coordMax, rawX1))
    const y1 = Math.max(0, Math.min(coordMax, rawY1))
    const x2 = Math.max(0, Math.min(coordMax, rawX2))
    const y2 = Math.max(0, Math.min(coordMax, rawY2))

    if (x2 <= x1 || y2 <= y1) {
      continue
    }

    results.push({
      class: DISEASE_CLASSES[classIds[idx]],
      confidence: scores[idx],
      bbox: {
        x: x1 / coordScale,
        y: y1 / coordScale,
        width: (x2 - x1) / coordScale,
        height: (y2 - y1) / coordScale,
      },
    })
  }

  return results
}

async function loadModel(modelUrl: string): Promise<void> {
  console.log("[Worker] Starting model load from:", modelUrl)
  ctx.postMessage({ type: "loading", progress: 0 } as InferenceResponse)

  try {
    console.log("[Worker] Downloading model...")
    ctx.postMessage({ type: "loading", progress: 10 } as InferenceResponse)

    const response = await fetch(modelUrl, { credentials: "same-origin" })

    if (!response.ok) {
      throw new Error(
        `Model download failed (${response.status} ${response.statusText}) from ${modelUrl}`,
      )
    }

    const modelBuffer = await response.arrayBuffer()

    console.log("[Worker] Loading model via ONNX Runtime...")
    ctx.postMessage({ type: "loading", progress: 60 } as InferenceResponse)

    const startTime = Date.now()
    session = await ort.InferenceSession.create(modelBuffer, {
      executionProviders: ["wasm"],
    })

    const loadTime = ((Date.now() - startTime) / 1000).toFixed(2)
    console.log(`[Worker] Model loaded successfully in ${loadTime}s`)
    console.log("[Worker] Model input names:", session.inputNames)
    console.log("[Worker] Model output names:", session.outputNames)
    ctx.postMessage({ type: "ready" } as InferenceResponse)
  } catch (e) {
    console.error("[Worker] Model load failed:", e)
    ctx.postMessage({
      type: "error",
      error: e instanceof Error ? e.message : "Failed to load model",
    } as InferenceResponse)
  }
}

async function runInference(id: string, imageData: ImageData): Promise<void> {
  console.log("[Worker] Starting inference for request:", id)

  if (!session) {
    console.error("[Worker] Model not loaded, cannot run inference")
    ctx.postMessage({
      id,
      type: "error",
      error: "Model not loaded",
    } as InferenceResponse)
    return
  }

  try {
    console.log("[Worker] Preprocessing image...")
    const preprocessStart = performance.now()
    const inputData = preprocessImage(imageData)
    console.log(
      `[Worker] Preprocessing completed in ${(performance.now() - preprocessStart).toFixed(2)}ms`,
    )

    console.log("[Worker] Creating input tensor...")
    const inputTensor = new ort.Tensor("float32", inputData, [1, 3, IMAGE_SIZE, IMAGE_SIZE])

    console.log("[Worker] Running ONNX inference...")
    const inferenceStart = performance.now()
    const results = await session.run({ images: inputTensor })
    console.log(
      `[Worker] ONNX inference completed in ${(performance.now() - inferenceStart).toFixed(2)}ms`,
    )

    // Get the first output tensor (YOLOv8 uses 'output0')
    const outputNames = Object.keys(results)
    console.log("[Worker] Result output names:", outputNames)
    const outputTensor = results[outputNames[0]].data as Float32Array
    const outputDims = results[outputNames[0]].dims as number[]

    console.log("[Worker] Postprocessing output, dims:", outputDims)
    const postprocessStart = performance.now()
    const detections = postprocessOutput(outputTensor, outputDims)
    console.log(
      `[Worker] Postprocessing completed in ${(performance.now() - postprocessStart).toFixed(2)}ms`,
    )
    console.log(`[Worker] Found ${detections.length} detections`)

    ctx.postMessage({
      id,
      type: "result",
      data: detections,
    } as InferenceResponse)
    console.log("[Worker] Sent result for request:", id)
  } catch (e) {
    console.error("[Worker] Inference failed:", e)
    ctx.postMessage({
      id,
      type: "error",
      error: e instanceof Error ? e.message : "Inference failed",
    } as InferenceResponse)
  }
}

ctx.onmessage = async (event: MessageEvent<InferenceRequest>) => {
  const { type, id, data } = event.data
  console.log("[Worker] Received message:", type, "id:", id)

  if (type === "load") {
    await loadModel(data as unknown as string)
  } else if (type === "run") {
    console.log("[Worker] Creating ImageData from", (data as number[]).length, "bytes")
    const imageData = new ImageData(new Uint8ClampedArray(data as number[]), IMAGE_SIZE, IMAGE_SIZE)
    await runInference(id, imageData)
  } else if (type === "release") {
    console.log("[Worker] Releasing model")
    session = null
    ctx.postMessage({ id, type: "ready" } as InferenceResponse)
  }
}
