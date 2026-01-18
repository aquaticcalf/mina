/**
 * Inference service for running TFLite model and processing detections.
 *
 * Since the TFLite model is exported with nms=False, this service implements
 * Non-Maximum Suppression (NMS) to filter overlapping detections.
 */

import { DISEASE_CLASSES } from "@/lib/model/types"
import type { BoundingBox, Detection, DiseaseClass } from "@/lib/model/types"

const IMAGE_SIZE = 640
const CONFIDENCE_THRESHOLD = 0.3
const IOU_THRESHOLD = 0.45

interface RawDetection {
    classIndex: number
    confidence: number
    boundingBox: BoundingBox
}

export type { RawDetection }

function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
    const x1 = Math.max(box1.x, box2.x)
    const y1 = Math.max(box1.y, box2.y)
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width)
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height)

    const intersectionWidth = Math.max(0, x2 - x1)
    const intersectionHeight = Math.max(0, y2 - y1)
    const intersectionArea = intersectionWidth * intersectionHeight

    const area1 = box1.width * box1.height
    const area2 = box2.width * box2.height
    const unionArea = area1 + area2 - intersectionArea

    if (unionArea === 0) return 0
    return intersectionArea / unionArea
}

function xywhToBoundingBox(
    x: number,
    y: number,
    width: number,
    height: number,
): BoundingBox {
    return {
        x,
        y,
        width,
        height,
    }
}

function cxcywhToBoundingBox(
    cx: number,
    cy: number,
    w: number,
    h: number,
): BoundingBox {
    return {
        x: cx - w / 2,
        y: cy - h / 2,
        width: w,
        height: h,
    }
}

export function applyNMS(
    detections: RawDetection[],
    iouThreshold: number = IOU_THRESHOLD,
): RawDetection[] {
    if (detections.length === 0) return []

    const sortedDetections = [...detections].sort(
        (a, b) => b.confidence - a.confidence,
    )

    const keep: RawDetection[] = []

    while (sortedDetections.length > 0) {
        const best = sortedDetections.shift()!
        keep.push(best)

        const remaining: RawDetection[] = []

        for (const detection of sortedDetections) {
            const sameClass = detection.classIndex === best.classIndex
            const iou = sameClass
                ? calculateIoU(best.boundingBox, detection.boundingBox)
                : 0
            if (!sameClass || iou < iouThreshold) {
                remaining.push(detection)
            }
        }

        sortedDetections.length = 0
        sortedDetections.push(...remaining)
    }

    return keep
}

export function parseYolov8Output(
    output: Float32Array,
    inputWidth: number,
    inputHeight: number,
): RawDetection[] {
    const detections: RawDetection[] = []

    const numAnchors = output.length / (4 + 1 + DISEASE_CLASSES.length)

    for (let i = 0; i < numAnchors; i++) {
        const baseOffset = i * (4 + 1 + DISEASE_CLASSES.length)

        const cx = output[baseOffset + 0] / inputWidth
        const cy = output[baseOffset + 1] / inputHeight
        const w = output[baseOffset + 2] / inputWidth
        const h = output[baseOffset + 3] / inputHeight
        const objectness = output[baseOffset + 4]

        for (let c = 0; c < DISEASE_CLASSES.length; c++) {
            const classOffset = baseOffset + 5 + c
            const classProb = output[classOffset]
            const confidence = objectness * classProb

            if (confidence >= CONFIDENCE_THRESHOLD) {
                const bbox = cxcywhToBoundingBox(cx, cy, w, h)

                detections.push({
                    classIndex: c,
                    confidence,
                    boundingBox: bbox,
                })
            }
        }
    }

    return detections
}

export function filterByConfidence(
    detections: RawDetection[],
    threshold: number = CONFIDENCE_THRESHOLD,
): RawDetection[] {
    return detections.filter((d) => d.confidence >= threshold)
}

export function sortByConfidence(detections: RawDetection[]): RawDetection[] {
    return [...detections].sort((a, b) => b.confidence - a.confidence)
}

export function convertToDetections(
    rawDetections: RawDetection[],
    imageWidth: number,
    imageHeight: number,
): Detection[] {
    const scaledDetections: Detection[] = []

    for (let i = 0; i < rawDetections.length; i++) {
        const raw = rawDetections[i]
        const diseaseClass = DISEASE_CLASSES[raw.classIndex] as DiseaseClass

        const bbox = raw.boundingBox

        let x = Math.max(0, Math.min(1, bbox.x))
        let y = Math.max(0, Math.min(1, bbox.y))
        let width = Math.max(0, Math.min(1, bbox.width))
        let height = Math.max(0, Math.min(1, bbox.height))

        if (x + width > 1) {
            width = 1 - x
        }
        if (y + height > 1) {
            height = 1 - y
        }

        const detection: Detection = {
            id: `det_${i.toString().padStart(3, "0")}`,
            diseaseClass,
            confidence: raw.confidence,
            boundingBox: { x, y, width, height },
        }

        scaledDetections.push(detection)
    }

    return scaledDetections
}

export interface InferenceResult {
    detections: Detection[]
    inferenceTimeMs: number
}

const NUM_ANCHORS = 8400
