import { useRef, useEffect, useState } from "react"
import type { Detection } from "@/lib/model/types"
import { getBoundingBoxColor, getDiseaseInfo } from "@/lib/model/disease"

interface Props {
  imageUrl: string
  detections: Detection[]
}

export function AnnotatedImage({ imageUrl, detections }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 })

  // Get the natural dimensions of the uploaded/captured image
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
    }
    img.src = imageUrl
  }, [imageUrl])

  // Draw bounding boxes accurately over the object-contain scaled image
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || imgSize.w === 0) return

    const drawAnnotations = () => {
      const ctx = canvas.getContext("2d")
      if (!ctx) return

      // Use the actual on-screen client dimensions of the canvas
      const cw = canvas.clientWidth
      const ch = canvas.clientHeight
      if (cw === 0 || ch === 0) return

      // Fix for high-DPI screens (Retina displays/mobile phones) so lines aren't blurry
      const dpr = window.devicePixelRatio || 1
      canvas.width = cw * dpr
      canvas.height = ch * dpr
      ctx.scale(dpr, dpr)
      ctx.clearRect(0, 0, cw, ch)

      // Calculate the "object-contain" math to find exactly where the image is rendered
      const imgRatio = imgSize.w / imgSize.h
      const containerRatio = cw / ch

      let renderW, renderH, offsetX, offsetY

      if (containerRatio > imgRatio) {
        // Container is wider. Image is constrained by height (pillarboxed).
        renderH = ch
        renderW = renderH * imgRatio
        offsetX = (cw - renderW) / 2
        offsetY = 0
      } else {
        // Container is taller. Image is constrained by width (letterboxed).
        renderW = cw
        renderH = renderW / imgRatio
        offsetX = 0
        offsetY = (ch - renderH) / 2
      }

      // Draw the detections using the calculated boundaries
      detections.forEach((det) => {
        const info = getDiseaseInfo(det.diseaseClass)
        const color = getBoundingBoxColor(det.diseaseClass)

        // Map the 0-1 percentage coordinates to the actual rendered image pixels
        const x = offsetX + det.boundingBox.x * renderW
        const y = offsetY + det.boundingBox.y * renderH
        const w = det.boundingBox.width * renderW
        const h = det.boundingBox.height * renderH

        // Draw Box
        ctx.strokeStyle = color
        ctx.lineWidth = 2.5
        ctx.strokeRect(x, y, w, h)

        // Calculate Label
        const confidence = (det.confidence * 100).toFixed(0)
        const labelText = `${info.displayName} ${confidence}%`
        ctx.font = "bold 12px monospace"
        const textW = ctx.measureText(labelText).width
        const labelH = 22

        // Push label below box if it hits the top edge
        const labelY = y > labelH + 4 ? y - labelH : y + h

        // Draw Label Background
        ctx.fillStyle = color
        ctx.fillRect(x, labelY, textW + 12, labelH)

        // Draw Label Text
        ctx.fillStyle = "#ffffff"
        ctx.strokeStyle = "rgba(0,0,0,0.5)"
        ctx.lineWidth = 2
        ctx.strokeText(labelText, x + 6, labelY + 15)
        ctx.fillText(labelText, x + 6, labelY + 15)
      })
    }

    // Run initial draw
    drawAnnotations()

    // Redraw if the user resizes the window or rotates their phone
    window.addEventListener("resize", drawAnnotations)
    return () => window.removeEventListener("resize", drawAnnotations)
  }, [detections, imgSize])

  return (
    <div
      className="relative flex h-full w-full items-center justify-center overflow-hidden rounded-xl bg-transparent"
      role="img"
      aria-label={`Fish scan with ${detections.length} detection${detections.length !== 1 ? "s" : ""} annotated`}
    >
      <img
        src={imageUrl}
        alt=""
        className="block h-full w-full object-contain pointer-events-none"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full pointer-events-none"
        aria-hidden="true"
      />
    </div>
  )
}
