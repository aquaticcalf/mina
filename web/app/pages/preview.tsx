import { useNavigate } from "react-router-dom"
import { RotateCcw, Scan } from "lucide-react"
import { useCameraContext } from "@/lib/camera/context"
import { useEffect } from "react"

export default function PreviewPage() {
  const { capturedImageUrl, setCapturedImage } = useCameraContext()
  const navigate = useNavigate()

  useEffect(() => {
    if (!capturedImageUrl) navigate("/", { replace: true })
  }, [capturedImageUrl, navigate])

  const handleRetake = () => {
    setCapturedImage(null)
    navigate("/")
  }

  const handleAnalyse = () => {
    navigate("/analysis")
  }

  if (!capturedImageUrl) return null

  return (
    <div className="relative flex flex-1 flex-col bg-foreground h-full max-h-dvh w-full">
      {/* Viewfinder-style Image Preview */}
      <div className="relative flex-1 overflow-hidden bg-foreground">
        <img
          src={capturedImageUrl}
          alt="Captured photo ready for analysis"
          className="h-full w-full object-contain"
        />
      </div>

      {/* Controls Bar (Matches CameraPage perfectly) */}
      <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center justify-between bg-transparent px-8 pb-[calc(2rem+env(safe-area-inset-bottom))] pt-24 md:px-16">
        {/* Retake Button (Matches Gallery/Flip buttons) */}
        <button
          className="group flex size-14 items-center justify-center rounded-2xl bg-white/60 text-black shadow-xl transition-all hover:scale-105 hover:bg-white/80 active:scale-95"
          onClick={handleRetake}
          aria-label="Retake photo"
        >
          <RotateCcw
            size={24}
            className="transition-transform duration-500 group-hover:-rotate-90"
            aria-hidden="true"
          />
        </button>

        {/* Analyse Button */}
        <button
          className="group flex h-14 items-center justify-center gap-3 rounded-2xl bg-white/60 px-8 text-black shadow-xl transition-all hover:scale-105 hover:opacity-80 active:scale-95"
          onClick={handleAnalyse}
          aria-label="Analyse photo"
        >
          <Scan size={20} aria-hidden="true" />
          <span className="font-bold tracking-wide">Analyse</span>
        </button>
      </div>
    </div>
  )
}
