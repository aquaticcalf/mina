import { Download, X } from "lucide-react"

interface PwaInstallBannerProps {
  onInstall: () => void
  onDismiss: () => void
}

export function PwaInstallBanner({ onInstall, onDismiss }: PwaInstallBannerProps) {
  return (
    <div className="absolute bottom-[calc(140px+env(safe-area-inset-bottom))] left-4 right-4 z-40 flex justify-center md:bottom-[160px] animate-in slide-in-from-bottom-5 duration-500">
      <div className="flex w-full max-w-sm items-center justify-between gap-4 rounded-2xl border border-border/50 bg-background/80 p-4 shadow-2xl backdrop-blur-xl">
        <div className="flex items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-foreground text-background shadow-inner">
            <Download size={20} />
          </div>
          <div className="flex flex-col">
            <p className="text-sm font-semibold tracking-wide">Install Mina</p>
            <p className="text-xs text-muted-foreground">Add to home screen for offline use</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={onInstall}
            className="rounded-xl bg-foreground px-4 py-2 text-xs font-semibold tracking-wide text-background transition-transform active:scale-95"
          >
            Install
          </button>
          <button
            onClick={onDismiss}
            className="flex size-8 items-center justify-center rounded-full bg-muted/50 text-muted-foreground transition-colors hover:bg-muted active:scale-95"
            aria-label="Dismiss install prompt"
          >
            <X size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}
