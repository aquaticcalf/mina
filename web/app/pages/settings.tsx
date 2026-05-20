import { useState, useEffect } from "react"
import { Monitor, Moon, Sun, Trash2, CheckCircle2, Loader2, AlertCircle, Circle, Download, RefreshCw } from "lucide-react"
import { useTheme } from "@/components/theme-provider"
import { usePwaInstall } from "@/hooks/use-pwa-install"
import { clearHistory, getHistoryItems } from "@/lib/history"
import { inferenceService, gateService } from "@/lib/inference"
import type { InferenceStatus, GateStatus } from "@/lib/inference"

type Theme = "light" | "dark" | "system"

const THEME_OPTIONS: {
  value: Theme
  label: string
  icon: typeof Sun
}[] = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
]

type ModelStatus = InferenceStatus | GateStatus

function ModelStatusBadge({ status }: { status: ModelStatus }) {
  if (status === "ready") {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-600 dark:text-emerald-400">
        <CheckCircle2 size={12} aria-hidden="true" />
        Cached
      </span>
    )
  }
  if (status === "loading") {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-3 py-1 text-xs font-medium text-muted-foreground">
        <Loader2 size={12} className="animate-spin" aria-hidden="true" />
        Loading…
      </span>
    )
  }
  if (status === "error") {
    return (
      <span className="flex items-center gap-1.5 rounded-full border border-destructive/20 bg-destructive/10 px-3 py-1 text-xs font-medium text-destructive">
        <AlertCircle size={12} aria-hidden="true" />
        Error
      </span>
    )
  }
  // idle / downloading
  return (
    <span className="flex items-center gap-1.5 rounded-full border border-border bg-muted/30 px-3 py-1 text-xs font-medium text-muted-foreground/60">
      <Circle size={12} aria-hidden="true" />
      Not loaded
    </span>
  )
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { isInstallAvailable, promptInstall } = usePwaInstall()
  const [historyCount, setHistoryCount] = useState(0)
  const [diseaseModelStatus, setDiseaseModelStatus] = useState<ModelStatus>(
    inferenceService.getStatus().status,
  )
  const [gateModelStatus, setGateModelStatus] = useState<ModelStatus>(
    gateService.getStatus().status,
  )

  useEffect(() => {
    getHistoryItems()
      .then((items) => setHistoryCount(items.length))
      .catch((err) => console.error("Failed to load history count:", err))

    // Subscribe to model status updates
    const unsubDisease = inferenceService.onStatusChange((state) =>
      setDiseaseModelStatus(state.status),
    )
    const unsubGate = gateService.onStatusChange((state) =>
      setGateModelStatus(state.status),
    )

    return () => {
      unsubDisease()
      unsubGate()
    }
  }, [])

  const handleClearHistory = async () => {
    if (
      window.confirm(
        `Delete all ${historyCount} scan${historyCount !== 1 ? "s" : ""}? This cannot be undone.`,
      )
    ) {
      try {
        await clearHistory()
        setHistoryCount(0)
      } catch (err) {
        console.error("Failed to clear history:", err)
        alert("Failed to clear history. Please try again.")
      }
    }
  }

  return (
    <div className="flex min-h-screen w-full flex-col bg-background pb-24 transition-colors duration-300">
      {/* Sticky Theme-Aware Header */}
      <header className="sticky top-0 z-50 flex shrink-0 items-center justify-between border-b border-border bg-background/80 px-6 py-4 backdrop-blur-xl md:px-12 md:py-5">
        <h1 className="text-lg font-semibold tracking-widest text-foreground uppercase">
          Settings
        </h1>
      </header>

      <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 p-6 md:p-10">
        {/* Appearance Section */}
        <section className="flex flex-col gap-3" aria-labelledby="theme-heading">
          <h2
            className="pl-4 font-mono text-[11px] font-bold uppercase tracking-widest text-muted-foreground"
            id="theme-heading"
          >
            Appearance
          </h2>

          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-sm transition-colors duration-300">
            <div className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between md:px-8">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Theme
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Choose your preferred colour scheme
                </p>
              </div>

              {/* Premium Segmented Control (Theme Aware) */}
              <div
                className="flex shrink-0 gap-1 rounded-2xl border border-border/50 bg-muted/50 p-1.5 shadow-inner"
                role="radiogroup"
                aria-label="Theme selection"
              >
                {THEME_OPTIONS.map(({ value, label, icon: Icon }) => {
                  const isActive = theme === value
                  return (
                    <label
                      key={value}
                      className={`group relative flex min-w-[90px] cursor-pointer items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-xs font-medium transition-all duration-300 ${
                        isActive
                          ? "bg-background text-foreground shadow-sm ring-1 ring-border/50"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <input
                        type="radio"
                        name="theme"
                        value={value}
                        checked={isActive}
                        onChange={() => setTheme(value)}
                        className="pointer-events-none absolute h-0 w-0 opacity-0"
                        aria-label={`${label} theme`}
                      />
                      <Icon
                        size={16}
                        className={
                          isActive
                            ? "text-foreground"
                            : "text-muted-foreground group-hover:text-foreground"
                        }
                        aria-hidden="true"
                      />
                      <span className="tracking-wide">{label}</span>
                    </label>
                  )
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Models Section */}
        <section className="flex flex-col gap-3" aria-labelledby="models-heading">
          <h2
            className="pl-4 font-mono text-[11px] font-bold uppercase tracking-widest text-muted-foreground"
            id="models-heading"
          >
            AI Models
          </h2>

          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-sm transition-colors duration-300">
            {/* Gate model row */}
            <div className="flex items-center justify-between gap-4 p-6 md:px-8">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Fish Gate
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Validates that submitted images contain a fish · ~2.5 MB
                </p>
              </div>
              <ModelStatusBadge status={gateModelStatus} />
            </div>

            {/* Divider */}
            <div className="h-px w-full bg-border" aria-hidden="true" />

            {/* Disease model row */}
            <div className="flex items-center justify-between gap-4 p-6 md:px-8">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Disease Detector
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Identifies fish diseases from images · ~12 MB
                </p>
              </div>
              <ModelStatusBadge status={diseaseModelStatus} />
            </div>

            {/* Divider */}
            <div className="h-px w-full bg-border" aria-hidden="true" />

            {/* Force Update models row */}
            <div className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between md:px-8 bg-muted/10">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Update Models
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Fetch the latest AI models from the server.
                </p>
              </div>

              <button
                onClick={() => {
                  gateService.serve(true);
                  inferenceService.serve(true);
                }}
                disabled={diseaseModelStatus === "loading" || gateModelStatus === "loading"}
                className="group flex h-12 w-full items-center justify-center gap-2 rounded-2xl border border-border bg-background px-6 text-sm font-medium tracking-wide text-foreground shadow-sm transition-all hover:bg-muted active:scale-95 disabled:pointer-events-none disabled:opacity-50 md:w-auto"
              >
                <RefreshCw
                  size={18}
                  className={`transition-transform ${diseaseModelStatus === 'loading' || gateModelStatus === 'loading' ? 'animate-spin' : 'group-hover:rotate-180'}`}
                />
                <span>Force Update</span>
              </button>
            </div>
          </div>
        </section>

        {/* Data Section */}
        <section className="flex flex-col gap-3" aria-labelledby="data-heading">
          <h2
            className="pl-4 font-mono text-[11px] font-bold uppercase tracking-widest text-muted-foreground"
            id="data-heading"
          >
            Data
          </h2>

          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-sm transition-colors duration-300">
            <div className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between md:px-8">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Scan History
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {historyCount} {historyCount === 1 ? "scan" : "scans"} stored locally on this
                  device
                </p>
              </div>

              <button
                onClick={handleClearHistory}
                disabled={historyCount === 0}
                className="group flex h-12 w-full items-center justify-center gap-2 rounded-2xl border border-destructive/20 bg-destructive/10 px-6 text-sm font-medium tracking-wide text-destructive transition-all hover:bg-destructive/20 active:scale-95 disabled:pointer-events-none disabled:opacity-50 md:w-auto"
                aria-label="Clear all scan history"
              >
                <Trash2
                  size={18}
                  className="transition-transform group-hover:scale-110"
                  aria-hidden="true"
                />
                <span>Clear Data</span>
              </button>
            </div>
          </div>
        </section>

        {/* Install Section (Only shown if installable) */}
        {isInstallAvailable && (
          <section className="flex flex-col gap-3" aria-labelledby="install-heading">
            <h2
              className="pl-4 font-mono text-[11px] font-bold uppercase tracking-widest text-muted-foreground"
              id="install-heading"
            >
              App
            </h2>

            <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-sm transition-colors duration-300">
              <div className="flex flex-col gap-5 p-6 md:flex-row md:items-center md:justify-between md:px-8">
                <div className="flex-1 min-w-0">
                  <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                    Install Mina
                  </p>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Add to your home screen for fast, offline access.
                  </p>
                </div>

                <button
                  onClick={promptInstall}
                  className="group flex h-12 w-full items-center justify-center gap-2 rounded-2xl bg-foreground px-6 text-sm font-medium tracking-wide text-background shadow-xl transition-transform active:scale-95 md:w-auto"
                  aria-label="Install App"
                >
                  <Download
                    size={18}
                    className="transition-transform group-hover:translate-y-0.5"
                    aria-hidden="true"
                  />
                  <span>Install</span>
                </button>
              </div>
            </div>
          </section>
        )}

        {/* About Section */}
        <section className="flex flex-col gap-3" aria-labelledby="about-heading">
          <h2
            className="pl-4 font-mono text-[11px] font-bold uppercase tracking-widest text-muted-foreground"
            id="about-heading"
          >
            About
          </h2>

          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-sm transition-colors duration-300">
            {/* App Info Row */}
            <div className="flex items-center justify-between gap-4 p-6 md:px-8">
              <div className="flex-1 min-w-0">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  FishCare YOLO
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Fish disease detection — powered by on-device ML
                </p>
              </div>
              <span className="shrink-0 rounded-full border border-border bg-muted/50 px-3 py-1 font-mono text-[10px] font-bold tracking-widest text-muted-foreground shadow-inner">
                v1.0.0
              </span>
            </div>

            {/* Divider */}
            <div className="h-px w-full bg-border" aria-hidden="true" />

            {/* Privacy Row */}
            <div className="flex items-center gap-4 p-6 md:px-8">
              <div className="flex-1">
                <p className="mb-1 text-base font-semibold tracking-wide text-card-foreground">
                  Privacy First
                </p>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  All analysis runs entirely on your device. Your photos and data are never uploaded
                  to the cloud.
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
