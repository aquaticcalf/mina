import { useState } from "react"
import { ChevronDown, AlertCircle, AlertTriangle, CheckCircle } from "lucide-react"
import type { Detection } from "@/lib/model/types"
import { DISEASE_INFO } from "@/lib/model/disease"

interface Props {
  detection: Detection
}

export function DetectionCard({ detection }: Props) {
  const [expanded, setExpanded] = useState(false)
  const info = DISEASE_INFO[detection.diseaseClass]

  const severityColor = (() => {
    switch (info.severity) {
      case "healthy":
        return "var(--healthy)"
      case "low":
        return "var(--low)"
      case "medium":
      case "high":
        return "var(--medium)"
    }
  })()

  const Icon = (() => {
    switch (info.severity) {
      case "healthy":
        return CheckCircle
      case "low":
        return AlertTriangle
      case "medium":
      case "high":
        return AlertCircle
    }
  })()

  // For healthy, show "Healthy" instead of "healthy severity"
  const severityLabel = info.severity === "healthy" ? "Healthy" : `${info.severity} severity`

  return (
    <article className="relative overflow-hidden rounded-2xl border border-border bg-card shadow-sm transition-colors duration-300 hover:bg-card/80">
      <button
        className="flex min-h-[4.5rem] w-full items-center justify-between px-5 py-4 text-left outline-none"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls={`det-body-${detection.id}`}
      >
        <div className="flex min-w-0 flex-1 items-center gap-4">
          <div className="rounded-full bg-muted/50 p-1.5 shadow-inner border border-border/50">
            <Icon size={16} style={{ color: severityColor }} aria-hidden="true" />
          </div>

          <div className="flex min-w-0 flex-1 flex-col justify-center gap-0.5">
            <p className="truncate text-base font-semibold tracking-wide text-card-foreground">
              {info.displayName}
            </p>
            <span
              className="font-mono text-[10px] font-bold uppercase tracking-widest"
              style={{ color: severityColor }}
            >
              {severityLabel}
            </span>
          </div>
        </div>

        <div className="ml-4 flex shrink-0 items-center gap-3">
          <span
            className="font-mono text-lg font-light tracking-wider text-card-foreground"
            aria-label={`${(detection.confidence * 100).toFixed(0)}% confidence`}
          >
            {(detection.confidence * 100).toFixed(0)}%
          </span>
          <ChevronDown
            size={18}
            className={`text-muted-foreground transition-transform duration-300 ${expanded ? "rotate-180" : ""}`}
            aria-hidden="true"
          />
        </div>
      </button>

      {expanded && (
        <div
          className="flex flex-col gap-6 border-t border-border px-5 pb-6 pt-5 animate-in fade-in slide-in-from-top-2 duration-300"
          id={`det-body-${detection.id}`}
        >
          <p className="text-sm leading-relaxed text-muted-foreground">{info.description}</p>

          {info.symptoms.length > 0 && (
            <div className="flex flex-col gap-3">
              <h4 className="font-mono text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Symptoms
              </h4>
              <ul className="ml-4 flex flex-col gap-1.5 list-disc list-outside marker:text-muted-foreground/50">
                {info.symptoms.map((s: string, i: number) => (
                  <li key={i} className="text-sm leading-[1.6] text-foreground/80">
                    {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {info.treatments.length > 0 && (
            <div className="flex flex-col gap-3">
              <h4 className="font-mono text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                Treatment
              </h4>
              <ol className="ml-4 flex flex-col gap-1.5 list-decimal list-outside marker:text-muted-foreground/50 marker:font-mono marker:text-xs">
                {info.treatments.map((t: string, i: number) => (
                  <li key={i} className="text-sm leading-[1.6] text-foreground/80 pl-1">
                    {t}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}
    </article>
  )
}
