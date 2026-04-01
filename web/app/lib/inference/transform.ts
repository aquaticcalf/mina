/**
 * Transform raw inference worker results to app types.
 */

import type { InferenceResult, Detection, DiseaseClass } from "@/lib/model/types"
import { isValidDiseaseClass } from "@/lib/model/types"
import { generateUUID } from "@/lib/utils/uuid"

/**
 * Raw result from the inference worker.
 * This is the format returned by worker.ts
 */
export interface RawInferenceResult {
  class: string
  confidence: number
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
}

/**
 * Transform raw worker results to the InferenceResult type used by the app.
 *
 * @param rawResults - Array of raw detection results from the worker
 * @param inferenceTimeMs - Time taken for inference in milliseconds
 * @returns InferenceResult with properly typed detections
 */
export function transformResults(
  rawResults: RawInferenceResult[],
  inferenceTimeMs: number,
): InferenceResult {
  const detections: Detection[] = rawResults
    .filter((r) => isValidDiseaseClass(r.class))
    .map((r) => ({
      id: generateUUID(),
      diseaseClass: r.class as DiseaseClass,
      confidence: r.confidence,
      boundingBox: {
        x: r.bbox.x,
        y: r.bbox.y,
        width: r.bbox.width,
        height: r.bbox.height,
      },
    }))

  return {
    detections,
    inferenceTimeMs,
  }
}
