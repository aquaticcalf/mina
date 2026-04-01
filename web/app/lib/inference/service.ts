import type { InferenceResponse, InferenceResult } from "./worker"
import InferenceWorker from "./worker?worker"

function getDefaultModelUrl(): string {
  const configured = import.meta.env.VITE_MODEL_URL

  if (typeof configured === "string" && configured.trim().length > 0) {
    return configured.trim()
  }

  return `${import.meta.env.BASE_URL}model/best.onnx`
}

const MODEL_URL = getDefaultModelUrl()

export type InferenceStatus = "idle" | "downloading" | "loading" | "ready" | "error"

export interface InferenceState {
  status: InferenceStatus
  progress: number
  error: string | null
}

export type StatusCallback = (state: InferenceState) => void

interface LetterboxMeta {
  srcWidth: number
  srcHeight: number
  scale: number
  padX: number
  padY: number
}

function unletterboxDetections(
  detections: InferenceResult[],
  meta: LetterboxMeta,
): InferenceResult[] {
  const { srcWidth, srcHeight, scale, padX, padY } = meta

  const remapped: InferenceResult[] = []

  for (const det of detections) {
    const x = det.bbox.x * 640
    const y = det.bbox.y * 640
    const w = det.bbox.width * 640
    const h = det.bbox.height * 640

    const x1 = (x - padX) / scale
    const y1 = (y - padY) / scale
    const x2 = (x + w - padX) / scale
    const y2 = (y + h - padY) / scale

    const clampedX1 = Math.max(0, Math.min(srcWidth, x1))
    const clampedY1 = Math.max(0, Math.min(srcHeight, y1))
    const clampedX2 = Math.max(0, Math.min(srcWidth, x2))
    const clampedY2 = Math.max(0, Math.min(srcHeight, y2))

    if (clampedX2 <= clampedX1 || clampedY2 <= clampedY1) {
      continue
    }

    remapped.push({
      ...det,
      bbox: {
        x: clampedX1 / srcWidth,
        y: clampedY1 / srcHeight,
        width: (clampedX2 - clampedX1) / srcWidth,
        height: (clampedY2 - clampedY1) / srcHeight,
      },
    })
  }

  return remapped
}

class InferenceService {
  private worker: Worker | null = null
  private pendingRequests: Map<
    string,
    {
      resolve: (data: InferenceResult[]) => void
      reject: (err: Error) => void
      letterbox: LetterboxMeta
    }
  > = new Map()
  private statusCallbacks: Set<StatusCallback> = new Set()
  private state: InferenceState = { status: "idle", progress: 0, error: null }
  private modelUrl: string = MODEL_URL

  private updateState(partial: Partial<InferenceState>) {
    this.state = { ...this.state, ...partial }
    this.statusCallbacks.forEach((cb) => cb(this.state))
  }

  onStatusChange(callback: StatusCallback) {
    this.statusCallbacks.add(callback)
    callback(this.state)
    return () => this.statusCallbacks.delete(callback)
  }

  getStatus(): InferenceState {
    return this.state
  }

  setModelUrl(url: string) {
    this.modelUrl = url
  }

  async serve(): Promise<void> {
    if (this.state.status === "ready") {
      return
    }

    this.updateState({ status: "loading", progress: 0, error: null })
    this.initWorker(this.modelUrl)
  }

  private initWorker(url: string) {
    if (this.worker) {
      this.worker.terminate()
    }

    this.worker = new InferenceWorker()

    this.worker.onerror = (event: ErrorEvent) => {
      const message =
        event.message && event.message.length > 0
          ? event.message
          : "Inference worker crashed while loading the model"

      this.updateState({
        status: "error",
        error: message,
      })
    }

    this.worker.onmessageerror = () => {
      this.updateState({
        status: "error",
        error: "Inference worker failed to process a message",
      })
    }

    this.worker.onmessage = (event: MessageEvent<InferenceResponse>) => {
      const msg = event.data

      if (msg.type === "loading") {
        this.updateState({ progress: msg.progress || 0 })
      } else if (msg.type === "ready") {
        this.updateState({ status: "ready", progress: 100 })
      } else if (msg.type === "result") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.resolve(unletterboxDetections(msg.data || [], req.letterbox))
          this.pendingRequests.delete(msg.id)
        }
      } else if (msg.type === "error") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.reject(new Error(msg.error || "Inference failed"))
          this.pendingRequests.delete(msg.id)
        }
        this.updateState({
          status: "error",
          error: msg.error || "Inference failed",
        })
      }
    }

    this.worker.postMessage({ type: "load", id: "load", data: url })
  }

  async run(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<InferenceResult[]> {
    if (this.state.status !== "ready" || !this.worker) {
      throw new Error("Model not ready. Call serve() first.")
    }

    const canvas = document.createElement("canvas")
    canvas.width = 640
    canvas.height = 640
    const ctx = canvas.getContext("2d")
    if (!ctx) {
      throw new Error("Failed to get canvas context")
    }

    const srcWidth =
      imageElement instanceof HTMLImageElement ? imageElement.naturalWidth : imageElement.width
    const srcHeight =
      imageElement instanceof HTMLImageElement ? imageElement.naturalHeight : imageElement.height

    const scale = Math.min(640 / srcWidth, 640 / srcHeight)
    const drawWidth = srcWidth * scale
    const drawHeight = srcHeight * scale
    const padX = (640 - drawWidth) / 2
    const padY = (640 - drawHeight) / 2

    ctx.fillStyle = "#000"
    ctx.fillRect(0, 0, 640, 640)
    ctx.drawImage(imageElement, padX, padY, drawWidth, drawHeight)
    const imageData = ctx.getImageData(0, 0, 640, 640)

    const id = crypto.randomUUID()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, {
        resolve,
        reject,
        letterbox: { srcWidth, srcHeight, scale, padX, padY },
      })

      this.worker!.postMessage({
        type: "run",
        id,
        data: Array.from(imageData.data),
      })
    })
  }

  unserved(): void {
    if (this.worker) {
      this.worker.postMessage({ type: "release", id: "release", data: [] })
      this.worker.terminate()
      this.worker = null
    }
    this.updateState({ status: "idle", progress: 0, error: null })
  }
}

export const inferenceService = new InferenceService()
