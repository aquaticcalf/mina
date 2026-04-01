import type { InferenceResult } from "@/lib/model/types"
import { AnnotatedImage } from "./annotated-image"
import { DetectionCard } from "./detection-card"

interface Props {
  imageUrl: string
  result: InferenceResult
  showTimestamp?: boolean
  timestamp?: number
}

function formatTimestamp(ts: number) {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(ts))
}

export function ResultsView({ imageUrl, result, showTimestamp, timestamp }: Props) {
  return (
    <div className="flex flex-1 flex-col w-full bg-background md:grid md:grid-cols-[1fr_400px] md:grid-rows-[auto_1fr] md:items-start">
      {/* Cleaned up Timestamp (No bulky background/border) */}
      {showTimestamp && timestamp && (
        <div className="px-6 pt-6 pb-2 md:col-span-2 md:px-10">
          <p className="font-mono text-[11px] font-medium uppercase tracking-widest text-muted-foreground">
            Scan • {formatTimestamp(timestamp)}
          </p>
        </div>
      )}

      {/* Left Column: Image Container */}
      <div className="flex w-full items-center justify-center p-6 md:col-start-1 md:row-start-2 md:self-stretch md:px-10 md:py-6">
        <div className="relative flex w-full max-w-md items-center justify-center overflow-hidden rounded-[2rem] border border-border bg-card shadow-lg md:max-w-full md:h-[calc(100vh-14rem)] transition-colors duration-300">
          <AnnotatedImage imageUrl={imageUrl} detections={result.detections} />
        </div>
      </div>

      {/* Right Column: Summary & Detections Scroll Area */}
      {/* Added native CSS hiding utilities to completely hide the scrollbar across all browsers */}
      <div className="flex flex-col gap-8 px-6 pb-32 md:col-start-2 md:row-start-2 md:max-h-[calc(100dvh-60px)] md:overflow-y-auto md:border-l md:border-border md:px-8 md:py-6 md:pb-12 [scrollbar-width:none] [-ms-overflow-style:none] [&::-webkit-scrollbar]:hidden transition-colors duration-300">
        <div className="flex flex-col gap-5">
          {/* Cleaned up Section Header (No pill badges) */}
          <div className="flex items-baseline gap-3">
            <h2 className="font-mono text-xs font-bold uppercase tracking-widest text-muted-foreground">
              Detections
            </h2>
            <span className="font-mono text-[10px] font-medium tracking-widest text-muted-foreground/60 uppercase">
              {result.detections.length} {result.detections.length === 1 ? "Found" : "Found"}
            </span>
          </div>

          {/* Cards List */}
          <div className="flex flex-col gap-3">
            {result.detections.map((det) => (
              <DetectionCard key={det.id} detection={det} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
