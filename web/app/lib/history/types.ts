import type { InferenceResult } from "@/lib/model/types"

export interface HistoryItem {
  id: string
  timestamp: number
  originalImageUrl: string
  processedImageUrl: string
  results: InferenceResult
}

export interface StoredHistoryItem {
  id: string
  timestamp: number
  originalImage: ArrayBuffer
  processedImage: ArrayBuffer
  originalImageType: string
  processedImageType: string
  results: InferenceResult
}

export interface HistoryItemInput {
  timestamp: number
  originalImage: Blob
  processedImage: Blob
  results: InferenceResult
}
