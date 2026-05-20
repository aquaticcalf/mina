import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useCameraContext } from "@/lib/camera/context"
import { useDetectionContext } from "@/lib/detection/context"
import { inferenceService, gateService, transformResults } from "@/lib/inference"
import { saveHistoryItem } from "@/lib/history"
import { loadImageFromBlob, createAnnotatedImage } from "@/lib/utils/image"
import { cn } from "@/lib/utils"

type AnalysisStep =
  | "loading-image"
  | "running-gate"
  | "running-inference"
  | "processing-results"
  | "saving"

const STEP_LABELS: Record<AnalysisStep, string> = {
  "loading-image": "Loading image data",
  "running-gate": "Checking for fish",
  "running-inference": "Running disease detection",
  "processing-results": "Processing detections",
  saving: "Saving to history",
}

const STEPS: AnalysisStep[] = [
  "loading-image",
  "running-gate",
  "running-inference",
  "processing-results",
  "saving",
]

/** Wait until a service reaches "ready" (or reject on "error"). */
function waitForReady(
  service: typeof inferenceService | typeof gateService,
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const unsub = service.onStatusChange((state) => {
      if (state.status === "ready") {
        unsub()
        resolve()
      } else if (state.status === "error") {
        unsub()
        reject(new Error(state.error || "Failed to load model"))
      }
    })
  })
}

export default function AnalysisPage() {
  const { capturedImage } = useCameraContext()
  const { setCurrentOutcome, bypassGate, setBypassGate } = useDetectionContext()
  const navigate = useNavigate()
  const ran = useRef(false)
  const [currentStep, setCurrentStep] = useState<AnalysisStep>("loading-image")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!capturedImage) {
      navigate("/", { replace: true })
      return
    }
    if (ran.current) return
    ran.current = true

    runAnalysis(capturedImage)
  }, [capturedImage, navigate, setCurrentOutcome])

  async function runAnalysis(imageBlob: Blob) {
    try {
      // ── Step 1: Load image ────────────────────────────────────────────────
      setCurrentStep("loading-image")
      const img = await loadImageFromBlob(imageBlob)

      // ── Step 2: Ensure models are ready ──────────────────────────────────
      if (inferenceService.getStatus().status !== "ready") await inferenceService.serve()
      if (gateService.getStatus().status !== "ready") await gateService.serve()
      
      await Promise.all([
        inferenceService.getStatus().status === "ready" ? Promise.resolve() : waitForReady(inferenceService),
        gateService.getStatus().status === "ready" ? Promise.resolve() : waitForReady(gateService),
      ])

      // ── Step 3: Run gate check ────────────────────────────────────────────
      setCurrentStep("running-gate")

      // bypassGate is set when the user taps "Analyse anyway" on the no-fish page
      const skipGate = bypassGate
      if (skipGate) setBypassGate(false) // consume the flag immediately

      if (!skipGate) {
        const gateResult = await gateService.run(img)
        if (!gateResult.isFish) {
          // Not a fish — show the no-fish page; do NOT save to history
          setCurrentOutcome({ kind: "no_fish", gateConfidence: gateResult.confidence })
          navigate("/no-fish", { replace: true })
          return
        }
      }

      // ── Step 4: Run disease inference (gate passed) ───────────────────────
      setCurrentStep("running-inference")
      const startTime = performance.now()
      const rawResults = await inferenceService.run(img)
      const inferenceTimeMs = performance.now() - startTime

      // ── Step 5: Transform results ─────────────────────────────────────────
      setCurrentStep("processing-results")
      const result = transformResults(rawResults, inferenceTimeMs)

      // ── Step 6: Create annotated image and save to history ────────────────
      setCurrentStep("saving")
      const annotatedImage = await createAnnotatedImage(imageBlob, result.detections)

      await saveHistoryItem({
        timestamp: Date.now(),
        originalImage: imageBlob,
        processedImage: annotatedImage,
        results: result,
      })

      // Set outcome in context and navigate to results page
      setCurrentOutcome({ kind: "detections", result })
      navigate("/results", { replace: true })
    } catch (err) {
      console.error("Analysis failed:", err)
      setError(err instanceof Error ? err.message : "Analysis failed")

      setTimeout(() => {
        navigate("/preview", { replace: true })
      }, 2000)
    }
  }

  const currentStepIndex = STEPS.indexOf(currentStep)

  if (error) {
    return (
      <div
        className="flex h-screen w-full flex-col items-center justify-center bg-background p-6 transition-colors duration-300"
        role="alert"
      >
        <div className="flex w-full max-w-[320px] flex-col items-center gap-6 rounded-[2rem] border border-destructive/20 bg-destructive/10 p-10 shadow-lg backdrop-blur-md">
          <div
            className="relative flex size-20 items-center justify-center rounded-full bg-destructive/20"
            aria-hidden="true"
          >
            <div className="text-4xl font-light text-destructive">!</div>
            {/* Pulsing error ring */}
            <div className="absolute inset-0 rounded-full border border-destructive/30 animate-ping" />
          </div>

          <div className="flex flex-col items-center gap-2 text-center">
            <p className="text-xl font-semibold tracking-wide text-foreground">Analysis Failed</p>
            <p className="text-sm leading-relaxed text-muted-foreground">{error}</p>
          </div>

          <div className="mt-4 rounded-full bg-muted/50 px-4 py-2">
            <p className="text-xs font-medium tracking-wide text-muted-foreground">
              Returning to preview...
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex h-screen w-full flex-col items-center justify-center bg-background p-6 transition-colors duration-300"
      role="status"
      aria-label="Analysing image, please wait"
    >
      <div className="flex w-full max-w-[340px] flex-col items-center gap-10">
        {/* Modern Dual-Ring Scanner */}
        <div className="relative flex flex-col items-center gap-6">
          <div className="relative size-[88px]" aria-hidden="true">
            {/* Outer Ring */}
            <div className="absolute inset-0 rounded-full border-[3px] border-muted/20 border-t-foreground/80 animate-[spin_1.5s_linear_infinite]" />
            {/* Inner Ring */}
            <div className="absolute inset-2.5 rounded-full border-[3px] border-muted/20 border-b-foreground/50 animate-[spin_2s_linear_infinite_reverse]" />
            {/* Center Pulse */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="size-2 rounded-full bg-foreground animate-pulse" />
            </div>
          </div>
          <p className="text-2xl font-medium tracking-widest text-foreground">ANALYSING</p>
        </div>

        {/* Glassmorphism Steps Card */}
        <div className="w-full rounded-[2rem] border border-border bg-card p-8 shadow-lg">
          <ul className="flex w-full flex-col gap-5" aria-hidden="true">
            {STEPS.map((step, i) => {
              const isActive = i === currentStepIndex
              const isCompleted = i < currentStepIndex

              return (
                <li
                  key={step}
                  className={cn(
                    "flex items-center gap-4 transition-all duration-500",
                    isCompleted && "text-muted-foreground",
                    isActive && "text-foreground scale-[1.02] transform",
                    !isCompleted && !isActive && "text-muted-foreground/40",
                  )}
                >
                  <div className="relative flex size-3 shrink-0 items-center justify-center">
                    {/* Status Dot */}
                    <span
                      className={cn(
                        "absolute size-full rounded-full transition-all duration-500",
                        isCompleted && "bg-muted-foreground/40 scale-75",
                        isActive && "bg-foreground shadow-[0_0_12px_rgba(var(--foreground),0.8)]",
                        !isCompleted &&
                          !isActive &&
                          "border border-muted-foreground/20 bg-transparent",
                      )}
                    />
                    {/* Active Ping Effect */}
                    {isActive && (
                      <span className="absolute size-full rounded-full bg-foreground/50 animate-ping" />
                    )}
                  </div>

                  <div className="flex flex-col">
                    <span className="text-sm font-medium tracking-wide">{STEP_LABELS[step]}</span>
                  </div>
                </li>
              )
            })}
          </ul>
        </div>

        {/* Privacy Badge */}
        <div className="mt-4 rounded-full border border-border/50 bg-muted/30 px-5 py-2.5">
          <p className="text-center text-xs font-medium tracking-wide text-muted-foreground">
            Running on-device — no data is transmitted
          </p>
        </div>
      </div>
    </div>
  )
}
