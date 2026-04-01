import { useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { Clock, ChevronRight, Scan } from "lucide-react"
import { getHistoryItems, revokeHistoryItemUrls } from "@/lib/history"
import type { HistoryItem } from "@/lib/history/types"
import { DISEASE_INFO } from "@/lib/model/disease"
import { getSeverityMeta, getWorstSeverity } from "@/lib/model/disease/severity"
import { Button } from "@/components/ui/button"

function formatDate(ts: number) {
  return new Intl.DateTimeFormat("en-GB", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(ts))
}

function getSummary(item: HistoryItem) {
  const diseases = item.results.detections.filter((d) => {
    const severity = DISEASE_INFO[d.diseaseClass].severity
    return severity !== "healthy" && severity !== "low"
  })
  if (diseases.length === 0) return "No diseases detected"
  if (diseases.length === 1) return DISEASE_INFO[diseases[0].diseaseClass].displayName
  return `${diseases.length} diseases detected`
}

export default function HistoryIndexPage() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    let mounted = true

    getHistoryItems()
      .then((items) => {
        if (mounted) {
          setHistory(items)
          setLoading(false)
        }
      })
      .catch((err) => {
        console.error("Failed to load history:", err)
        if (mounted) setLoading(false)
      })

    return () => {
      mounted = false
      // Revoke Object URLs to prevent memory leaks
      history.forEach(revokeHistoryItemUrls)
    }
  }, [])

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="relative size-10" aria-hidden="true">
          <div className="absolute inset-0 rounded-full border-[3px] border-muted/20 border-t-foreground/80 animate-spin" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-1 w-full flex-col bg-background md:pb-0 overflow-hidden transition-colors duration-300">
      {/* Sticky Glassmorphism Header */}
      <header className="shrink-0 flex items-center justify-between border-b border-border bg-background/80 px-6 py-4 backdrop-blur-xl md:px-12 md:py-5 transition-colors duration-300">
        <h1 className="text-lg font-semibold tracking-widest text-foreground uppercase">History</h1>
        {/* Cleaned up count text (no pill) */}
        <span className="font-mono text-[11px] font-medium tracking-widest text-muted-foreground uppercase">
          {history.length} {history.length === 1 ? "Scan" : "Scans"}
        </span>
      </header>

      {history.length === 0 ? (
        /* Premium Empty State */
        <div className="flex-1 overflow-y-auto flex items-center justify-center p-6">
          <div className="flex w-full max-w-[320px] flex-col items-center gap-6 rounded-[2rem] border border-border bg-card p-10 text-center shadow-lg">
            <div className="flex size-20 items-center justify-center rounded-full bg-muted/50 border border-border/50">
              <Clock className="size-8 text-muted-foreground" strokeWidth={1.5} />
            </div>

            <div className="space-y-2">
              <p className="text-xl font-medium tracking-wide text-foreground">No scans yet</p>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Your scan history will appear here after you analyse a fish photo.
              </p>
            </div>

            <Button
              onClick={() => navigate("/")}
              className="mt-4 flex h-14 w-full items-center justify-center gap-2 rounded-2xl bg-foreground text-background font-semibold tracking-wide shadow-lg transition-all hover:opacity-90 active:scale-95"
            >
              <Scan size={18} aria-hidden="true" />
              <span>Start scanning</span>
            </Button>
          </div>
        </div>
      ) : (
        /* History List as Glass Cards */
        <ul className="flex-1 overflow-y-auto flex flex-col gap-4 p-4 md:gap-5 md:p-8">
          {history.map((item) => {
            const severity = getWorstSeverity(item.results.detections)
            const meta = getSeverityMeta(severity)

            return (
              <li key={item.id}>
                <button
                  className="group flex min-h-[88px] w-full items-center gap-4 rounded-2xl border border-border bg-card p-3 text-left shadow-sm transition-all hover:scale-[1.01] hover:bg-card/80 active:scale-[0.98] md:p-4 md:gap-5"
                  onClick={() => navigate(`/history/${item.id}`)}
                  aria-label={`View scan from ${formatDate(item.timestamp)}: ${getSummary(item)}`}
                >
                  {/* Image Thumbnail */}
                  <div className="relative size-16 shrink-0 overflow-hidden rounded-xl border border-border bg-muted/50 md:size-20">
                    <img
                      src={item.processedImageUrl}
                      alt=""
                      className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-110"
                    />
                  </div>

                  {/* Cleaned up Meta Data (No Pills) */}
                  <div className="flex min-w-0 flex-1 flex-col justify-center gap-1.5">
                    <p className="truncate text-[15px] font-semibold tracking-wide text-card-foreground">
                      {getSummary(item)}
                    </p>

                    {/* Unified metadata line: Dot + Severity + Bullet + Time */}
                    <div className="flex items-center gap-2 mt-0.5">
                      <span
                        className="size-2 shrink-0 rounded-full shadow-[0_0_8px_currentColor]"
                        style={{ background: meta.color, color: meta.color }}
                      />
                      <span
                        className="font-mono text-[10px] font-bold uppercase tracking-widest"
                        style={{ color: meta.color }}
                      >
                        {meta.label}
                      </span>

                      <span className="text-muted-foreground/30 text-[10px]">•</span>

                      <time
                        className="font-mono text-[10px] font-medium uppercase tracking-widest text-muted-foreground truncate"
                        dateTime={new Date(item.timestamp).toISOString()}
                      >
                        {formatDate(item.timestamp)}
                      </time>
                    </div>
                  </div>

                  {/* Action Chevron */}
                  <div className="ml-2 flex size-8 shrink-0 items-center justify-center rounded-full bg-muted/50 border border-border/50 transition-colors group-hover:bg-muted">
                    <ChevronRight
                      size={16}
                      className="text-muted-foreground group-hover:text-foreground"
                    />
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
