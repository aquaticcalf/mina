import type { GateResponse } from "./gate-worker"
import GateWorker from "./gate-worker?worker"

export interface GateResult {
  isFish: boolean
  confidence: number
}

export type GateStatus = "idle" | "loading" | "ready" | "error"

export interface GateState {
  status: GateStatus
  progress: number
  error: string | null
}

export type GateStatusCallback = (state: GateState) => void

function getDefaultGateModelUrl(): string {
  const configured = import.meta.env.VITE_GATE_MODEL_URL

  if (typeof configured === "string" && configured.trim().length > 0) {
    return configured.trim()
  }

  // Use the API proxy mapping to GitHub Releases to bypass CORS
  return `/api/model-proxy?file=fish_gate.onnx`
}

const GATE_MODEL_URL = getDefaultGateModelUrl()
const GATE_SIZE = 224
// Torchvision default: resize shortest edge to this before cropping
const RESIZE_TO = 256

class GateService {
  private worker: Worker | null = null
  private pendingRequests: Map<
    string,
    {
      resolve: (data: GateResult) => void
      reject: (err: Error) => void
    }
  > = new Map()
  private statusCallbacks: Set<GateStatusCallback> = new Set()
  private state: GateState = { status: "idle", progress: 0, error: null }
  private modelUrl: string = GATE_MODEL_URL

  private updateState(partial: Partial<GateState>) {
    this.state = { ...this.state, ...partial }
    this.statusCallbacks.forEach((cb) => cb(this.state))
  }

  onStatusChange(callback: GateStatusCallback) {
    this.statusCallbacks.add(callback)
    callback(this.state)
    return () => this.statusCallbacks.delete(callback)
  }

  getStatus(): GateState {
    return this.state
  }

  setModelUrl(url: string) {
    this.modelUrl = url
  }

  async serve(force: boolean = false): Promise<void> {
    if (this.state.status === "ready" && !force) {
      return
    }

    this.updateState({ status: "loading", progress: 0, error: null })
    this.initWorker(this.modelUrl, force)
  }

  private initWorker(url: string, force: boolean = false) {
    if (this.worker) {
      this.worker.terminate()
    }

    this.worker = new GateWorker()

    this.worker.onerror = (event: ErrorEvent) => {
      const message =
        event.message && event.message.length > 0
          ? event.message
          : "Gate worker crashed while loading the model"

      this.updateState({ status: "error", error: message })
    }

    this.worker.onmessageerror = () => {
      this.updateState({
        status: "error",
        error: "Gate worker failed to process a message",
      })
    }

    this.worker.onmessage = (event: MessageEvent<GateResponse>) => {
      const msg = event.data

      if (msg.type === "loading") {
        this.updateState({ progress: msg.progress ?? 0 })
      } else if (msg.type === "ready") {
        this.updateState({ status: "ready", progress: 100 })
      } else if (msg.type === "result") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.resolve(msg.data!)
          this.pendingRequests.delete(msg.id)
        }
      } else if (msg.type === "error") {
        const req = this.pendingRequests.get(msg.id)
        if (req) {
          req.reject(new Error(msg.error || "Gate inference failed"))
          this.pendingRequests.delete(msg.id)
        }
        this.updateState({
          status: "error",
          error: msg.error || "Gate inference failed",
        })
      }
    }

    this.worker.postMessage({ type: "load", id: "load", data: { url, force } })
  }

  async run(imageElement: HTMLImageElement | HTMLCanvasElement): Promise<GateResult> {
    if (this.state.status !== "ready" || !this.worker) {
      throw new Error("Gate model not ready. Call serve() first.")
    }

    // ── Preprocessing: match torchvision Resize(256) + CenterCrop(224) ────
    //
    // Standard MobileNetV3 fine-tuning uses:
    //   transforms.Resize(256)      → scale shortest edge to 256, keep AR
    //   transforms.CenterCrop(224)  → take center 224×224 square
    //
    // We replicate both steps in a single drawImage() call by computing
    // the source rectangle in the *original* image's coordinate space:
    //
    //   scale     = 256 / min(srcW, srcH)
    //   srcCropSz = 224 / scale            ← size of crop in original pixels
    //   srcCropX  = (srcW - srcCropSz) / 2 ← centered horizontally
    //   srcCropY  = (srcH - srcCropSz) / 2 ← centered vertically
    //
    // drawImage(src, srcX, srcY, srcW, srcH, dstX, dstY, dstW, dstH)
    // scales the source rect into the destination rect in one GPU-accelerated op.

    const srcW =
      "naturalWidth" in imageElement ? imageElement.naturalWidth : imageElement.width
    const srcH =
      "naturalWidth" in imageElement ? imageElement.naturalHeight : imageElement.height

    const scale = RESIZE_TO / Math.min(srcW, srcH)
    const srcCropSz = GATE_SIZE / scale          // 224 back-projected into source space
    const srcCropX = (srcW - srcCropSz) / 2
    const srcCropY = (srcH - srcCropSz) / 2

    const canvas = document.createElement("canvas")
    canvas.width = GATE_SIZE
    canvas.height = GATE_SIZE
    const canvasCtx = canvas.getContext("2d")
    if (!canvasCtx) {
      throw new Error("Failed to get canvas context for gate preprocessing")
    }

    // Single-pass: crop center square (aspect-ratio-preserving) and scale to 224×224
    canvasCtx.drawImage(
      imageElement,
      srcCropX, srcCropY, srcCropSz, srcCropSz, // source: center square in original
      0, 0, GATE_SIZE, GATE_SIZE,               // dest: full 224×224 canvas
    )
    const imageData = canvasCtx.getImageData(0, 0, GATE_SIZE, GATE_SIZE)

    const id = crypto.randomUUID()

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject })

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

export const gateService = new GateService()
