import { Loader2, AlertCircle, RefreshCw } from "lucide-react"
import { useModelLoader } from "@/lib/inference/model-loader-context"

export function ModelDownloadBanner() {
  const { modelsReady, diseaseState, gateState, retryDownload } = useModelLoader()

  if (modelsReady) return null

  const gateLoading = gateState.status === "loading"
  const diseaseLoading = diseaseState.status === "loading" || diseaseState.status === "downloading"
  
  const hasError = gateState.status === "error" || diseaseState.status === "error"

  const combinedProgress = Math.round(((gateState.progress || 0) + (diseaseState.progress || 0)) / 2)

  return (
    <div className="absolute top-20 left-4 right-4 z-40 flex justify-center">
      <div className="flex w-full max-w-sm flex-col items-center gap-3 rounded-2xl border border-border/50 bg-background/80 p-5 shadow-2xl backdrop-blur-xl">
        {hasError ? (
          <>
            <div className="flex w-full items-start gap-3 text-red-500">
              <AlertCircle size={20} className="shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-semibold tracking-wide">Failed to download AI models</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Please check your internet connection and try again. Models are required to use the camera.
                </p>
              </div>
            </div>
            <button
              onClick={retryDownload}
              className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl bg-muted/50 py-2.5 text-sm font-medium transition-colors hover:bg-muted"
            >
              <RefreshCw size={16} />
              Retry Download
            </button>
          </>
        ) : (
          <>
            <div className="flex w-full items-center justify-between">
              <div className="flex items-center gap-2">
                <Loader2 size={16} className="animate-spin text-foreground" />
                <span className="text-sm font-medium tracking-wide">
                  {(gateLoading && diseaseLoading) || (!gateLoading && !diseaseLoading)
                    ? "Preparing AI models…"
                    : gateLoading
                      ? "Downloading gate model…"
                      : "Downloading disease model…"}
                </span>
              </div>
              <span className="text-xs font-semibold text-muted-foreground">{combinedProgress}%</span>
            </div>
            
            {/* Progress Bar Container */}
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted/50">
              <div
                className="h-full bg-foreground transition-all duration-300 ease-out"
                style={{ width: `${combinedProgress}%` }}
              />
            </div>
            <p className="w-full text-center text-[10px] uppercase tracking-widest text-muted-foreground/60 mt-1">
              One-time initial download (~18MB)
            </p>
          </>
        )}
      </div>
    </div>
  )
}
