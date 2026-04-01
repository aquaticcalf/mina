import { useRef, useEffect, useState, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import { Camera, Image, AlertCircle, RefreshCw } from "lucide-react"
import { useCameraContext } from "@/lib/camera/context"
import { cn } from "@/lib/utils"

export default function CameraPage() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [permission, setPermission] = useState<"pending" | "granted" | "denied">("pending")
  const [cameraError, setCameraError] = useState<string | null>(null)

  const {
    setCapturedImage,
    cameraFacingMode: facingMode,
    setCameraFacingMode: setFacingMode,
  } = useCameraContext()
  const navigate = useNavigate()

  const requestInProgressRef = useRef(false)
  const pendingRequestRef = useRef<"environment" | "user" | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [])

  const startCamera = useCallback(async (facing: "environment" | "user") => {
    if (requestInProgressRef.current) {
      pendingRequestRef.current = facing
      return
    }
    requestInProgressRef.current = true
    let currentFacing = facing

    while (currentFacing) {
      try {
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop())
          if (videoRef.current) {
            videoRef.current.srcObject = null
          }
          streamRef.current = null
          // Give mobile devices a moment to release the camera hardware
          await new Promise((resolve) => setTimeout(resolve, 500))
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          if (mountedRef.current) {
            setPermission("denied")
            setCameraError(
              "Camera API is unavailable. This usually happens when the site is not served over a secure connection (HTTPS). Please ensure you are using the HTTPS URL.",
            )
          }
          break
        }

        let stream: MediaStream
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: currentFacing,
              width: { ideal: 1920 },
              height: { ideal: 1080 },
            },
          })
        } catch (err: any) {
          if (
            err.name === "NotReadableError" ||
            err.name === "TrackStartError" ||
            err.name === "OverconstrainedError"
          ) {
            await new Promise((resolve) => setTimeout(resolve, 500))
            stream = await navigator.mediaDevices.getUserMedia({
              video: { facingMode: currentFacing },
            })
          } else {
            throw err
          }
        }

        if (!mountedRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          break
        }

        if (pendingRequestRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          await new Promise((resolve) => setTimeout(resolve, 500))
        } else {
          streamRef.current = stream
          if (videoRef.current) {
            videoRef.current.srcObject = stream
          }
          setPermission("granted")
          setCameraError(null)
        }
      } catch (err: unknown) {
        if (!mountedRef.current) break
        if (!pendingRequestRef.current) {
          const error = err as { name?: string }
          if (error.name === "NotAllowedError") {
            setPermission("denied")
            setCameraError(
              "Camera permission was denied. Please allow camera access in your browser settings, or use the gallery option below.",
            )
          } else if (error.name === "NotFoundError") {
            setPermission("denied")
            setCameraError(
              "No camera found on this device. Please use the gallery option to select a photo.",
            )
          } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
            // Do not set permission to denied so the switch camera button remains visible
            setCameraError(
              "Camera is already in use by another application or tab. Please try switching cameras again.",
            )
          } else {
            setPermission("denied")
            setCameraError(
              "Unable to access camera. Please use the gallery option to select a photo.",
            )
          }
          break
        }
      }

      const nextMode = pendingRequestRef.current
      pendingRequestRef.current = null
      currentFacing = nextMode as typeof currentFacing
    }

    requestInProgressRef.current = false
  }, [])

  useEffect(() => {
    startCamera(facingMode)
  }, [facingMode, startCamera])

  const handleCapture = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const containerWidth = video.clientWidth
    const containerHeight = video.clientHeight

    if (!videoWidth || !videoHeight || !containerWidth || !containerHeight) return

    // Calculate aspect ratios
    const videoRatio = videoWidth / videoHeight
    const containerRatio = containerWidth / containerHeight

    let sWidth = videoWidth
    let sHeight = videoHeight
    let sx = 0
    let sy = 0

    // Math to replicate CSS object-fit: cover
    if (containerRatio > videoRatio) {
      // Container is wider. Video width fits perfectly, height is cropped top/bottom.
      sHeight = videoWidth / containerRatio
      sy = (videoHeight - sHeight) / 2
    } else {
      // Container is taller. Video height fits perfectly, width is cropped left/right.
      sWidth = videoHeight * containerRatio
      sx = (videoWidth - sWidth) / 2
    }

    // Set canvas to the cropped high-res dimensions
    canvas.width = sWidth
    canvas.height = sHeight

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Handle mirror effect for front camera
    if (facingMode === "user") {
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)
    }

    // Draw only the visible portion of the video to the canvas
    ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, sWidth, sHeight)

    // Convert to Blob
    const blob = await new Promise<Blob>((resolve, reject) => {
      canvas.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Blob creation failed"))),
        "image/jpeg",
        0.92,
      )
    })

    setCapturedImage(blob)
    navigate("/preview")
  }

  const handleGallery = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setCapturedImage(file)
    navigate("/preview")
  }

  const toggleCamera = () => {
    setFacingMode(facingMode === "environment" ? "user" : "environment")
  }

  return (
    <div className="relative flex flex-1 flex-col bg-foreground h-full max-h-dvh w-full">
      {/* Viewfinder */}
      <div className="relative flex-1 overflow-hidden bg-foreground">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={cn(
            "h-full w-full object-cover transition-opacity duration-500",
            facingMode === "user" && "scale-x-[-1]",
            permission === "granted" ? "opacity-100" : "opacity-0 hidden",
          )}
          aria-label="Live camera feed"
        />

        {permission !== "granted" && (
          <div
            className="flex h-full flex-col items-center justify-center gap-4 bg-background text-white"
            role="status"
          >
            <Camera size={48} className="opacity-50" aria-hidden="true" />
            <p className="text-sm font-medium tracking-wide">Camera unavailable</p>
          </div>
        )}
      </div>

      {/* Floating Error Banner */}
      {cameraError && (
        <div
          className="absolute left-4 right-4 top-4 z-40 flex items-start gap-3 rounded-2xl border border-red-500/30 bg-red-500/15 p-4 text-sm text-red-100 shadow-2xl backdrop-blur-md"
          role="alert"
        >
          <AlertCircle size={20} className="shrink-0 text-red-400" aria-hidden="true" />
          <p className="leading-relaxed">{cameraError}</p>
        </div>
      )}

      {/* Modern Floating Hint */}
      <div className="pointer-events-none absolute bottom-[calc(140px+env(safe-area-inset-bottom))] left-0 right-0 z-30 flex justify-center px-4 md:bottom-[160px]">
        <div className="rounded-full border border-border/50 bg-background/80 px-5 py-2.5 backdrop-blur-md shadow-xl">
          <p className="text-center text-xs font-medium tracking-wide text-foreground">
            Position the fish clearly in frame
          </p>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center justify-between bg-transparent px-8 pb-[calc(2rem+env(safe-area-inset-bottom))] pt-24 md:px-16">
        {/* Gallery Button */}
        <button
          className="group flex flex-col items-center justify-center gap-2 transition-transform hover:scale-105 active:scale-95"
          onClick={() => fileInputRef.current?.click()}
          aria-label="Choose from gallery"
        >
          <div className="flex size-14 items-center justify-center rounded-2xl bg-white/60 text-black shadow-xl transition-colors group-hover:bg-white/80">
            <Image size={24} aria-hidden="true" />
          </div>
        </button>

        {/* Capture Shutter Button */}
        <button
          className="group flex size-[80px] items-center justify-center rounded-full border-[4px] border-white/40 shadow-xl transition-all hover:border-white/70 active:scale-95 disabled:cursor-not-allowed disabled:opacity-30 md:size-[92px]"
          onClick={handleCapture}
          disabled={permission !== "granted"}
          aria-label="Capture photo for analysis"
        >
          <span
            className="size-[60px] rounded-full bg-white/80 transition-all group-hover:scale-[0.96] group-active:scale-90 md:size-[70px]"
            aria-hidden="true"
          />
        </button>

        {/* Flip Camera Button */}
        {permission === "granted" ? (
          <div className="flex flex-col items-center justify-center gap-2">
            <button
              className="group flex size-14 items-center justify-center rounded-2xl bg-white/60 text-black shadow-xl transition-all hover:scale-105 hover:bg-white/80 active:scale-95"
              onClick={toggleCamera}
              aria-label="Flip camera"
            >
              <RefreshCw
                size={24}
                className="transition-transform duration-500 group-hover:rotate-180"
                aria-hidden="true"
              />
            </button>
          </div>
        ) : (
          <div className="w-14" aria-hidden="true" />
        )}
      </div>

      <canvas ref={canvasRef} className="hidden" aria-hidden="true" />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleGallery}
        aria-label="Upload image from gallery"
      />
    </div>
  )
}
