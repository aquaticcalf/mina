import { AlertCircle, CheckCircle } from "lucide-react"
import type { Detection } from "@/lib/model/types"
import { getDiseaseInfo } from "@/lib/model/disease"

interface Props {
  detections: Detection[]
}

export function ResultsSummary({ detections }: Props) {
  // Check if all detections are healthy (severity === "healthy")
  const allHealthy = detections.every((d) => {
    const info = getDiseaseInfo(d.diseaseClass)
    return info.severity === "healthy"
  })

  const highOrMediumCount = detections.filter((d) => {
    const info = getDiseaseInfo(d.diseaseClass)
    return info.severity === "high" || info.severity === "medium"
  }).length

  if (allHealthy) {
    return (
      <div className="flex items-center gap-5" role="status">
        <div className="flex size-12 shrink-0 items-center justify-center rounded-full border border-border/50 bg-muted/50 shadow-inner">
          <CheckCircle size={24} style={{ color: "var(--healthy)" }} aria-hidden="true" />
        </div>
        <div className="flex flex-col gap-1">
          <h2 className="text-lg font-semibold tracking-wide text-foreground">
            Fish appears healthy
          </h2>
          <span className="font-mono text-[10px] font-bold tracking-widest uppercase text-muted-foreground">
            No diseases detected
          </span>
        </div>
      </div>
    )
  }

  const iconColor = highOrMediumCount > 0 ? "var(--medium)" : "var(--low)"

  return (
    <div className="flex items-center gap-5" role="status">
      <div className="relative flex size-12 shrink-0 items-center justify-center rounded-full border border-border/50 bg-muted/50 shadow-inner">
        <AlertCircle size={24} style={{ color: iconColor }} aria-hidden="true" />
        {/* Subtle ping animation for urgent alerts */}
        {highOrMediumCount > 0 && (
          <span
            className="absolute inset-0 rounded-full border border-foreground/10 animate-ping opacity-50"
            style={{ borderColor: iconColor }}
            aria-hidden="true"
          />
        )}
      </div>

      <div className="flex flex-col gap-1">
        <h2 className="text-lg font-semibold tracking-wide text-foreground">
          {detections.length} Detection{detections.length !== 1 ? "s" : ""} Found
        </h2>

        {highOrMediumCount > 0 ? (
          <span
            className="font-mono text-[10px] font-bold tracking-widest uppercase"
            style={{ color: iconColor }}
          >
            {highOrMediumCount} Require urgent attention
          </span>
        ) : (
          <span className="font-mono text-[10px] font-bold tracking-widest uppercase text-muted-foreground">
            Review suggested treatments
          </span>
        )}
      </div>
    </div>
  )
}
