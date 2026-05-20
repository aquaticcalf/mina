import { useState, useEffect, useCallback } from "react"

// Types for the BeforeInstallPromptEvent
interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[]
  readonly userChoice: Promise<{
    outcome: "accepted" | "dismissed"
    platform: string
  }>
  prompt(): Promise<void>
}

declare global {
  interface WindowEventMap {
    beforeinstallprompt: BeforeInstallPromptEvent
  }
}

export function usePwaInstall() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null)
  const [isReadyForInstall, setIsReadyForInstall] = useState(false)

  // Track if user explicitly closed the banner
  const [isDismissed, setIsDismissed] = useState(() => {
    return localStorage.getItem("pwa-install-dismissed") === "true"
  })

  useEffect(() => {
    const handleBeforeInstallPrompt = (e: BeforeInstallPromptEvent) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault()
      // Stash the event so it can be triggered later.
      setDeferredPrompt(e)
      setIsReadyForInstall(true)
    }

    // Check if app is already installed
    if (window.matchMedia("(display-mode: standalone)").matches) {
      setIsReadyForInstall(false) // Already installed
    }

    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt)

    return () => {
      window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
    }
  }, [])

  const promptInstall = useCallback(async () => {
    if (!deferredPrompt) return

    // Show the install prompt
    await deferredPrompt.prompt()
    
    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice
    console.log(`User ${outcome} the A2HS prompt`)

    // We've used the prompt, and can't use it again, throw it away
    setDeferredPrompt(null)
    setIsReadyForInstall(false)
  }, [deferredPrompt])

  const dismissInstall = useCallback(() => {
    setIsDismissed(true)
    localStorage.setItem("pwa-install-dismissed", "true")
  }, [])

  return {
    canInstall: isReadyForInstall && !isDismissed,
    isInstallAvailable: isReadyForInstall, // Raw availability regardless of dismissal
    isDismissed,
    promptInstall,
    dismissInstall,
  }
}
