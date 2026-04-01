import { useParams, useNavigate } from "react-router-dom"
import { useEffect, useState, useRef } from "react"
import { ArrowLeft, Trash2, Share2, X, Download, Copy, MessageCircle, Loader2 } from "lucide-react"
import { getHistoryItem, revokeHistoryItemUrls, deleteHistoryItem } from "@/lib/history"
import type { HistoryItem } from "@/lib/history/types"
import { ResultsView } from "@/components/detection/results-view"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { DISEASE_INFO } from "@/lib/model/disease"
import { getSeverityMeta, getWorstSeverity } from "@/lib/model/disease/severity"

function getSummary(item: HistoryItem) {
  const diseases = item.results.detections.filter((d) => {
    const severity = DISEASE_INFO[d.diseaseClass].severity
    return severity !== "healthy" && severity !== "low"
  })
  if (diseases.length === 0) return "No diseases detected"
  if (diseases.length === 1) return DISEASE_INFO[diseases[0].diseaseClass].displayName
  return `${diseases.length} diseases detected`
}

export default function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

  const [item, setItem] = useState<HistoryItem | null>(null)
  const itemRef = useRef<HistoryItem | null>(null)

  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [isSharing, setIsSharing] = useState(false)
  const [showShareModal, setShowShareModal] = useState(false)

  useEffect(() => {
    if (!id) {
      setNotFound(true)
      setLoading(false)
      return
    }

    let mounted = true

    getHistoryItem(id)
      .then((result) => {
        if (!mounted) return
        if (result) {
          itemRef.current = result
          setItem(result)
        } else {
          setNotFound(true)
        }
        setLoading(false)
      })
      .catch((err) => {
        console.error("Failed to load history item:", err)
        if (mounted) {
          setNotFound(true)
          setLoading(false)
        }
      })

    return () => {
      mounted = false
      if (itemRef.current) revokeHistoryItemUrls(itemRef.current)
    }
  }, [id])

  const handleDelete = async () => {
    if (!id) return
    if (window.confirm("Are you sure you want to delete this scan?")) {
      try {
        await deleteHistoryItem(id)
        navigate("/history", { replace: true })
      } catch (err) {
        console.error("Failed to delete history item:", err)
        alert("Failed to delete scan.")
      }
    }
  }

  // Generate shareable text
  const getShareText = () => {
    if (!item) return ""
    const severity = getWorstSeverity(item.results.detections)
    const meta = getSeverityMeta(severity)
    const summary = getSummary(item)
    return `🐟 *Fish Scan Result*\n\n*Diagnosis:* ${summary}\n*Severity:* ${meta.label}\n\n_Scanned using the Fish Disease Analysis App_`
  }

  // The Master Share Function
  const handleShare = async () => {
    if (!item) return
    setIsSharing(true)
    const shareText = getShareText()

    try {
      const response = await fetch(item.processedImageUrl)
      const blob = await response.blob()
      const file = new File([blob], "fish-scan.jpg", { type: "image/jpeg" })

      const shareData = {
        title: "Fish Scan Result",
        text: shareText,
        files: [file],
      }

      // Try native share (iOS/Android)
      if (navigator.canShare && navigator.canShare(shareData)) {
        await navigator.share(shareData)
        setIsSharing(false)
        return
      }

      // Fallback: Text only native share
      if (navigator.share) {
        await navigator.share({ title: "Fish Scan", text: shareText })
        setIsSharing(false)
        return
      }
    } catch (err: any) {
      if (err.name === "AbortError") {
        setIsSharing(false)
        return
      }
      console.warn("Native share failed, using fallback UI", err)
    }

    // Open Shadcn Fallback Modal
    setIsSharing(false)
    setShowShareModal(true)
  }

  // Fallback Modal Actions
  const downloadImage = () => {
    if (!item) return
    const a = document.createElement("a")
    a.href = item.processedImageUrl
    a.download = `fish-scan-${Date.now()}.jpg`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setShowShareModal(false)
  }

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(getShareText())
      alert("Results copied to clipboard!")
    } catch {
      alert("Failed to copy text.")
    }
    setShowShareModal(false)
  }

  const shareToWhatsApp = () => {
    const text = encodeURIComponent(getShareText())
    window.open(`https://wa.me/?text=${text}`, "_blank")
    setShowShareModal(false)
  }

  if (loading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background">
        <div className="relative size-10">
          <div className="absolute inset-0 rounded-full border-[3px] border-muted/20 border-t-foreground/80 animate-spin" />
        </div>
      </div>
    )
  }

  if (notFound || !item) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background p-6 transition-colors duration-300">
        <div className="flex w-full max-w-[320px] flex-col items-center gap-6 rounded-[2rem] border border-border bg-card p-10 text-center shadow-lg">
          <div className="flex size-20 items-center justify-center rounded-full bg-muted/50 border border-border/50">
            <ArrowLeft className="size-8 text-muted-foreground" />
          </div>
          <p className="text-xl font-medium text-foreground">Scan Not Found</p>
          <Button
            onClick={() => navigate("/history")}
            className="mt-4 flex h-14 w-full gap-2 rounded-2xl bg-foreground text-background font-semibold shadow-lg"
          >
            <ArrowLeft size={18} />
            <span>Back to History</span>
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-background transition-colors duration-300">
      {/* Sticky Glassmorphism Header */}
      <header className="shrink-0 flex items-center justify-between border-b border-border bg-background/80 px-6 py-2.5 md:px-12 md:py-3.5 transition-colors duration-300">
        <button
          className="group flex h-10 items-center gap-2 rounded-full border border-border bg-muted/50 px-4 text-sm font-medium tracking-wide text-foreground shadow-sm transition-all hover:bg-muted active:scale-95"
          onClick={() => navigate("/history")}
        >
          <ArrowLeft size={16} className="transition-transform group-hover:-translate-x-1" />
          <span className="hidden sm:inline">History</span>
        </button>

        <div className="flex items-center gap-3">
          <button
            className="group flex h-10 items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/15 px-4 text-sm font-medium tracking-wide text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.15)] transition-all hover:bg-blue-500/25 hover:text-blue-300 active:scale-95 disabled:opacity-50"
            onClick={handleShare}
            disabled={isSharing}
          >
            {isSharing ? (
              <Loader2 size={16} className="animate-spin text-blue-400" />
            ) : (
              <Share2
                size={16}
                className="transition-transform group-hover:-translate-y-0.5 text-blue-400"
              />
            )}
            <span className="hidden sm:inline">{isSharing ? "Preparing..." : "Share"}</span>
          </button>

          <button
            className="group flex h-10 items-center gap-2 rounded-full border border-destructive/20 bg-destructive/10 px-4 text-sm font-medium tracking-wide text-destructive shadow-sm transition-all hover:bg-destructive/20 active:scale-95"
            onClick={handleDelete}
          >
            <Trash2 size={16} className="transition-transform group-hover:scale-110" />
            <span className="hidden sm:inline">Delete</span>
          </button>
        </div>
      </header>

      <main className="flex-1 w-full relative">
        <ResultsView
          imageUrl={item.processedImageUrl}
          result={item.results}
          showTimestamp
          timestamp={item.timestamp}
        />
      </main>

      {/* Shadcn Share Modal Fallback */}
      <Dialog open={showShareModal} onOpenChange={setShowShareModal}>
        {/* [&>button]:hidden magically removes the default Shadcn X button! */}
        <DialogContent className="w-[90vw] max-w-sm rounded-[2rem] border border-border bg-card p-6 shadow-lg sm:rounded-[2rem] [&>button]:hidden">
          {/* Custom Header with Flexbox to align Title and Custom X perfectly */}
          <DialogHeader className="mb-4 flex flex-row items-center justify-between space-y-0 p-0">
            <DialogTitle className="text-lg font-semibold tracking-wide text-foreground">
              Share Results
            </DialogTitle>
            <button
              onClick={() => setShowShareModal(false)}
              className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted/50 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            >
              <X size={16} />
            </button>
          </DialogHeader>

          <div className="flex flex-col gap-3">
            <button
              onClick={downloadImage}
              className="flex h-14 w-full items-center gap-4 rounded-2xl border border-border bg-muted/30 px-5 text-left text-foreground transition-colors hover:bg-muted active:scale-[0.98]"
            >
              <div className="flex size-8 items-center justify-center rounded-full bg-muted/50">
                <Download size={16} />
              </div>
              <span className="font-medium tracking-wide">Save Image</span>
            </button>

            <button
              onClick={copyText}
              className="flex h-14 w-full items-center gap-4 rounded-2xl border border-border bg-muted/30 px-5 text-left text-foreground transition-colors hover:bg-muted active:scale-[0.98]"
            >
              <div className="flex size-8 items-center justify-center rounded-full bg-muted/50">
                <Copy size={16} />
              </div>
              <span className="font-medium tracking-wide">Copy Text Summary</span>
            </button>

            <button
              onClick={shareToWhatsApp}
              className="flex h-14 w-full items-center gap-4 rounded-2xl border border-green-500/20 bg-green-500/10 px-5 text-left text-green-400 transition-colors hover:bg-green-500/20 active:scale-[0.98]"
            >
              <div className="flex size-8 items-center justify-center rounded-full bg-green-500/20 text-green-400">
                <MessageCircle size={16} />
              </div>
              <span className="font-medium tracking-wide">Share text to WhatsApp</span>
            </button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
