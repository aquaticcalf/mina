import { useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { Camera, FlaskConical } from "lucide-react"
import { useDetectionContext } from "@/lib/detection/context"
import { Button } from "@/components/ui/button"

const REDIRECT_DELAY_MS = 3000

export default function NoFishPage() {
  const { currentOutcome, setBypassGate } = useDetectionContext()
  const navigate = useNavigate()

  // True when there is no gate context (deep-link / refresh / back navigation)
  const isContextless = currentOutcome === null || currentOutcome.kind !== "no_fish"

  const confidence =
    currentOutcome?.kind === "no_fish"
      ? Math.round((1 - currentOutcome.gateConfidence) * 100)
      : null

  // Auto-redirect to home when context is missing
  const [redirecting, setRedirecting] = useState(false)
  useEffect(() => {
    if (!isContextless) return
    setRedirecting(true)
    const timer = setTimeout(() => {
      navigate("/", { replace: true })
    }, REDIRECT_DELAY_MS)
    return () => clearTimeout(timer)
  }, [isContextless, navigate])

  return (
    <div
      className="flex h-screen w-full flex-col items-center justify-center bg-background p-6 transition-colors duration-300"
      role="alert"
      aria-label="No fish detected in the submitted image"
    >
      <div className="flex w-full max-w-[340px] flex-col items-center gap-8 rounded-[2rem] border border-border bg-card p-10 shadow-lg">
        {/* Icon */}
        <div className="relative flex size-24 items-center justify-center" aria-hidden="true">
          {/* Soft ambient glow */}
          <div className="absolute inset-0 rounded-full bg-muted/60 blur-md" />
          {/* Icon ring */}
          <div className="relative flex size-24 items-center justify-center rounded-full border border-border bg-muted/50">
            {/* Fish emoji rendered as text for maximum compatibility */}
            <span className="text-4xl select-none" role="img" aria-label="fish">🐟</span>
          </div>
        </div>

        {/* Text */}
        <div className="flex flex-col items-center gap-3 text-center">
          <h1 className="text-2xl font-semibold tracking-wide text-foreground">
            No fish detected
          </h1>
          <p className="text-sm leading-relaxed text-muted-foreground">
            The image doesn't appear to contain a fish. Please photograph your fish directly
            in well-lit surroundings.
          </p>
          {confidence !== null && (
            <p className="text-xs font-mono text-muted-foreground/60">
              not-fish confidence: {confidence}%
            </p>
          )}
          {/* Shown only on context-less visits (deep-link / refresh) */}
          {redirecting && (
            <p className="text-xs font-mono text-muted-foreground/50 animate-pulse">
              Redirecting to home…
            </p>
          )}
        </div>

        {/* Tips */}
        <ul className="w-full space-y-2 rounded-2xl border border-border/50 bg-muted/30 p-4">
          {[
            "Make sure the fish fills most of the frame",
            "Use good lighting — avoid shadows",
            "Keep the camera steady and close",
          ].map((tip) => (
            <li key={tip} className="flex items-start gap-2 text-xs text-muted-foreground">
              <span className="mt-0.5 shrink-0 text-foreground/40">•</span>
              <span>{tip}</span>
            </li>
          ))}
        </ul>

        {/* CTAs */}
        <div className="flex w-full flex-col gap-3">
          <Button
            id="no-fish-try-again"
            onClick={() => navigate("/")}
            className="flex h-14 w-full items-center justify-center gap-2 rounded-2xl bg-foreground text-background font-semibold tracking-wide shadow-lg transition-all hover:opacity-90 active:scale-95"
          >
            <Camera size={18} aria-hidden="true" />
            <span>Try Again</span>
          </Button>

          {/* Bypass button — shown only when we have a real gate result */}
          {!isContextless && (
            <Button
              id="no-fish-analyse-anyway"
              variant="ghost"
              onClick={() => {
                setBypassGate(true)
                navigate("/analysis", { replace: true })
              }}
              className="flex h-10 w-full items-center justify-center gap-2 rounded-2xl text-muted-foreground text-sm tracking-wide transition-all hover:text-foreground hover:bg-muted/50 active:scale-95"
            >
              <FlaskConical size={15} aria-hidden="true" />
              <span>Analyse anyway</span>
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
