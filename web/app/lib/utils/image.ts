/**
 * Image utilities for creating annotated images.
 */

import type { Detection } from "@/lib/model/types"
import { getBoundingBoxColor, getDiseaseInfo } from "@/lib/model/disease"

/**
 * Load a Blob as an HTMLImageElement.
 *
 * @param blob - Image blob to load
 * @returns Promise resolving to the loaded image element
 */
export function loadImageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    const url = URL.createObjectURL(blob)

    img.onload = () => {
      URL.revokeObjectURL(url)
      resolve(img)
    }

    img.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error("Failed to load image"))
    }

    img.src = url
  })
}

/**
 * Create an annotated image with bounding boxes drawn on it.
 *
 * @param imageBlob - Original image blob
 * @param detections - Array of detections to draw
 * @returns Promise resolving to the annotated image as a Blob
 */
export async function createAnnotatedImage(
  imageBlob: Blob,
  detections: Detection[],
): Promise<Blob> {
  const img = await loadImageFromBlob(imageBlob)

  const canvas = document.createElement("canvas")
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight

  const ctx = canvas.getContext("2d")
  if (!ctx) {
    throw new Error("Failed to get canvas context")
  }

  // Draw the original image
  ctx.drawImage(img, 0, 0)

  // Draw bounding boxes for each detection
  detections.forEach((det) => {
    const info = getDiseaseInfo(det.diseaseClass)
    const color = getBoundingBoxColor(det.diseaseClass)

    const x = det.boundingBox.x * canvas.width
    const y = det.boundingBox.y * canvas.height
    const w = det.boundingBox.width * canvas.width
    const h = det.boundingBox.height * canvas.height

    // Draw bounding box
    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) * 0.004)
    ctx.strokeRect(x, y, w, h)

    // Draw label background
    const confidence = (det.confidence * 100).toFixed(0)
    const labelText = `${info.displayName} ${confidence}%`
    const fontSize = Math.max(12, Math.min(canvas.width, canvas.height) * 0.02)
    ctx.font = `bold ${fontSize}px monospace`
    const textMetrics = ctx.measureText(labelText)
    const labelH = fontSize + 8
    const labelY = y > labelH + 4 ? y - labelH - 2 : y + h + 2

    ctx.fillStyle = color
    ctx.fillRect(x, labelY, textMetrics.width + 10, labelH)

    // Draw label text
    ctx.fillStyle = "#000"
    ctx.fillText(labelText, x + 5, labelY + fontSize + 2)
  })

  // Convert canvas to blob
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob) {
          resolve(blob)
        } else {
          reject(new Error("Failed to create image blob"))
        }
      },
      imageBlob.type || "image/jpeg",
      0.9,
    )
  })
}
