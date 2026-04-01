import { useNavigate } from "react-router-dom"
import { Camera } from "lucide-react"
import { useDetectionContext } from "@/lib/detection/context"
import { useCameraContext } from "@/lib/camera/context"
import { ResultsView } from "@/components/detection/results-view"
import { Button } from "@/components/ui/button"

export default function ResultsPage() {
  const { currentResult } = useDetectionContext()
  const { capturedImageUrl } = useCameraContext()
  const navigate = useNavigate()

  // Premium Empty/Fallback State
  if (!currentResult || !capturedImageUrl) {
    return (
      <div className="flex h-screen w-full flex-col items-center justify-center bg-background p-6 transition-colors duration-300">
        <div className="flex w-full max-w-[320px] flex-col items-center gap-6 rounded-[2rem] border border-border bg-card p-10 text-center shadow-lg">
          <div className="flex size-20 items-center justify-center rounded-full bg-muted/50 border border-border/50">
            <Camera className="size-8 text-muted-foreground" strokeWidth={1.5} />
          </div>

          <div className="space-y-2">
            <p className="text-xl font-medium tracking-wide text-foreground">No Results</p>
            <p className="text-sm leading-relaxed text-muted-foreground">
              There is no analysis data available. Please capture an image first.
            </p>
          </div>

          <Button
            onClick={() => navigate("/")}
            className="mt-4 flex h-14 w-full items-center justify-center gap-2 rounded-2xl bg-foreground text-background font-semibold tracking-wide shadow-lg transition-all hover:opacity-90 active:scale-95"
          >
            <Camera size={18} aria-hidden="true" />
            <span>Go to Camera</span>
          </Button>
        </div>
      </div>
    )
  }

  // Main Results View
  return (
    <div className="flex min-h-screen w-full flex-col bg-background transition-colors duration-300">
      {/* Sticky Glassmorphism Header */}
      <header className="shrink-0 flex items-center justify-between border-b border-border bg-background/80 px-6 py-2.5 md:px-12 md:py-3.5 transition-colors duration-300">
        <h1 className="text-lg font-semibold tracking-widest text-foreground uppercase">Results</h1>

        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            className="group flex h-10 items-center gap-2 rounded-full border border-border bg-muted/50 px-5 text-sm font-medium text-foreground shadow-sm transition-all hover:bg-muted active:scale-95"
            onClick={() => navigate("/")}
          >
            <Camera
              size={16}
              className="transition-transform duration-300 group-hover:scale-110"
              aria-hidden="true"
            />
            <span>New Scan</span>
          </Button>
        </div>
      </header>

      {/* Content Area (Letting ResultsView handle its own internal layout, but with a sleek wrapper) */}
      <main className="flex-1 w-full relative">
        <ResultsView imageUrl={capturedImageUrl} result={currentResult} />
      </main>
    </div>
  )
}
