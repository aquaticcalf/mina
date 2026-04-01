import { useEffect, useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import { useCameraContext } from "@/lib/camera/context"
import { useDetectionContext } from "@/lib/detection/context"
import { inferenceService, transformResults } from "@/lib/inference"
import type { InferenceStatus } from "@/lib/inference"
import { saveHistoryItem } from "@/lib/history"
import { loadImageFromBlob, createAnnotatedImage } from "@/lib/utils/image"
import { cn } from "@/lib/utils"

type AnalysisStep =
  | "loading-image"
  | "loading-model"
  | "running-inference"
  | "processing-results"
  | "saving"

const STEP_LABELS: Record<AnalysisStep, string> = {
  "loading-image": "Loading image data",
  "loading-model": "Loading disease detection model",
  "running-inference": "Running disease detection",
  "processing-results": "Processing detections",
  saving: "Saving to history",
}

const STEPS: AnalysisStep[] = [
  "loading-image",
  "loading-model",
  "running-inference",
  "processing-results",
  "saving",
]

export default function AnalysisPage() {
  const { capturedImage } = useCameraContext()
  const { setCurrentResult } = useDetectionContext()
  const navigate = useNavigate()
  const ran = useRef(false)
  const [currentStep, setCurrentStep] = useState<AnalysisStep>("loading-image")
  const [modelStatus, setModelStatus] = useState<InferenceStatus>("idle")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!capturedImage) {
      navigate("/", { replace: true })
      return
    }
    if (ran.current) return
    ran.current = true

    runAnalysis(capturedImage)
  }, [capturedImage, navigate, setCurrentResult])

  async function runAnalysis(imageBlob: Blob) {
    try {
      // Step 1: Load image
      setCurrentStep("loading-image")
      const img = await loadImageFromBlob(imageBlob)

      // Step 2: Ensure model is loaded
      setCurrentStep("loading-model")

      // Subscribe to model loading status
      const unsubscribe = inferenceService.onStatusChange((state) => {
        setModelStatus(state.status)
      })

      try {
        await inferenceService.serve()

        // Wait for model to be ready
        const status = inferenceService.getStatus()
        if (status.status !== "ready") {
          // Wait for ready status
          await new Promise<void>((resolve, reject) => {
            const checkStatus = inferenceService.onStatusChange((state) => {
              if (state.status === "ready") {
                checkStatus()
                resolve()
              } else if (state.status === "error") {
                checkStatus()
                reject(new Error(state.error || "Failed to load model"))
              }
            })
          })
        }
      } finally {
        unsubscribe()
      }

      // Step 3: Run inference
      setCurrentStep("running-inference")
      const startTime = performance.now()
      const rawResults = await inferenceService.run(img)
      const inferenceTimeMs = performance.now() - startTime

      // Step 4: Transform results
      setCurrentStep("processing-results")
      const result = transformResults(rawResults, inferenceTimeMs)

      // Step 5: Create annotated image and save to history
      setCurrentStep("saving")
      const annotatedImage = await createAnnotatedImage(imageBlob, result.detections)

      await saveHistoryItem({
        timestamp: Date.now(),
        originalImage: imageBlob,
        processedImage: annotatedImage,
        results: result,
      })

      // Set result in context and navigate to results page
      setCurrentResult(result)
      navigate("/results", { replace: true })
    } catch (err) {
      console.error("Analysis failed:", err)
      setError(err instanceof Error ? err.message : "Analysis failed")

      // Navigate back to preview after a short delay so user can see error
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
                    {isActive && step === "loading-model" && modelStatus === "loading" && (
                      <span className="mt-1 text-xs tracking-wider text-muted-foreground uppercase">
                        Downloading...
                      </span>
                    )}
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
