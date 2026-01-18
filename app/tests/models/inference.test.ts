import { describe, it, expect } from "bun:test"
import {
    applyNMS,
    filterByConfidence,
    sortByConfidence,
    convertToDetections,
} from "@/lib/model/inference"
import { DISEASE_CLASSES } from "@/lib/model/types"
import type { RawDetection } from "@/lib/model/inference"

function createRawDetection(
    classIndex: number,
    confidence: number,
    x: number,
    y: number,
    width: number,
    height: number,
): RawDetection {
    return {
        classIndex,
        confidence,
        boundingBox: { x, y, width, height },
    }
}

describe("**Feature: fish-disease-detection, Property 2: Confidence filtering**", () => {
    describe("filterByConfidence", () => {
        it("should filter out detections below threshold", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.15, 0.3, 0.3, 0.2, 0.2),
                createRawDetection(2, 0.5, 0.5, 0.5, 0.2, 0.2),
                createRawDetection(3, 0.24, 0.7, 0.7, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 0.25)
            expect(filtered.length).toBe(2)
            expect(filtered[0].confidence).toBe(0.9)
            expect(filtered[1].confidence).toBe(0.5)
        })

        it("should keep all detections above threshold", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.95, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.85, 0.3, 0.3, 0.2, 0.2),
                createRawDetection(2, 0.75, 0.5, 0.5, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 0.25)
            expect(filtered.length).toBe(3)
        })

        it("should return empty array when all below threshold", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.2, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.1, 0.3, 0.3, 0.2, 0.2),
                createRawDetection(2, 0.15, 0.5, 0.5, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 0.25)
            expect(filtered.length).toBe(0)
        })

        it("should handle empty input array", () => {
            const filtered = filterByConfidence([], 0.25)
            expect(filtered.length).toBe(0)
        })

        it("should handle exact threshold boundary", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.25, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.2499, 0.3, 0.3, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 0.25)
            expect(filtered.length).toBe(1)
            expect(filtered[0].confidence).toBe(0.25)
        })

        it("should work with threshold of 0", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.1, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.01, 0.3, 0.3, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 0)
            expect(filtered.length).toBe(2)
        })

        it("should work with threshold of 1.0", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 1.0, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.99, 0.3, 0.3, 0.2, 0.2),
            ]
            const filtered = filterByConfidence(detections, 1.0)
            expect(filtered.length).toBe(1)
            expect(filtered[0].confidence).toBe(1.0)
        })
    })
})

describe("**Feature: fish-disease-detection, Property 3: Detection sorting by confidence**", () => {
    describe("sortByConfidence", () => {
        it("should sort detections in descending order", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.5, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.9, 0.3, 0.3, 0.2, 0.2),
                createRawDetection(2, 0.3, 0.5, 0.5, 0.2, 0.2),
                createRawDetection(3, 0.7, 0.7, 0.7, 0.2, 0.2),
            ]
            const sorted = sortByConfidence(detections)
            expect(sorted[0].confidence).toBe(0.9)
            expect(sorted[1].confidence).toBe(0.7)
            expect(sorted[2].confidence).toBe(0.5)
            expect(sorted[3].confidence).toBe(0.3)
        })

        it("should handle single detection", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.8, 0.1, 0.1, 0.2, 0.2),
            ]
            const sorted = sortByConfidence(detections)
            expect(sorted.length).toBe(1)
            expect(sorted[0].confidence).toBe(0.8)
        })

        it("should handle empty array", () => {
            const sorted = sortByConfidence([])
            expect(sorted.length).toBe(0)
        })

        it("should work with all confidence values", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 1.0, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.0, 0.3, 0.3, 0.2, 0.2),
                createRawDetection(2, 0.5, 0.5, 0.5, 0.2, 0.2),
            ]
            const sorted = sortByConfidence(detections)
            expect(sorted[0].confidence).toBe(1.0)
            expect(sorted[1].confidence).toBe(0.5)
            expect(sorted[2].confidence).toBe(0.0)
        })
    })
})

describe("**Feature: fish-disease-detection, Property 9: NMS deduplication**", () => {
    describe("applyNMS", () => {
        it("should remove overlapping boxes with high IoU", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.3, 0.3),
                createRawDetection(0, 0.8, 0.12, 0.12, 0.3, 0.3),
                createRawDetection(0, 0.7, 0.14, 0.14, 0.3, 0.3),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
            expect(result[0].confidence).toBe(0.9)
        })

        it("should keep non-overlapping boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(0, 0.8, 0.6, 0.6, 0.2, 0.2),
                createRawDetection(0, 0.7, 0.3, 0.3, 0.2, 0.2),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(3)
        })

        it("should handle empty input", () => {
            const result = applyNMS([])
            expect(result.length).toBe(0)
        })

        it("should handle single detection", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.5, 0.5, 0.2, 0.2),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
        })

        it("should keep higher confidence box when IoU > threshold", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.6, 0.1, 0.1, 0.3, 0.3),
                createRawDetection(0, 0.95, 0.12, 0.12, 0.3, 0.3),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
            expect(result[0].confidence).toBe(0.95)
        })

        it("should keep boxes from different classes even if overlapping", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.3, 0.3),
                createRawDetection(1, 0.8, 0.12, 0.12, 0.3, 0.3),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(2)
        })

        it("should work with different IoU thresholds", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.3, 0.3),
                createRawDetection(0, 0.8, 0.15, 0.15, 0.3, 0.3),
                createRawDetection(0, 0.7, 0.2, 0.2, 0.3, 0.3),
            ]

            const resultHighThreshold = applyNMS(detections, 0.1)
            expect(resultHighThreshold.length).toBe(1)

            const resultLowThreshold = applyNMS(detections, 0.9)
            expect(resultLowThreshold.length).toBe(3)
        })

        it("should handle identical boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.5, 0.5, 0.2, 0.2),
                createRawDetection(0, 0.8, 0.5, 0.5, 0.2, 0.2),
                createRawDetection(0, 0.7, 0.5, 0.5, 0.2, 0.2),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
            expect(result[0].confidence).toBe(0.9)
        })

        it("should handle partially overlapping boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.1, 0.1, 0.4, 0.4),
                createRawDetection(0, 0.8, 0.15, 0.15, 0.4, 0.4),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
            expect(result[0].confidence).toBe(0.9)
        })

        it("should handle barely overlapping boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.0, 0.0, 0.3, 0.3),
                createRawDetection(0, 0.8, 0.3, 0.3, 0.3, 0.3),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(2)
        })

        it("should handle complex scenario with multiple boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.95, 0.1, 0.1, 0.2, 0.2),
                createRawDetection(1, 0.85, 0.12, 0.12, 0.2, 0.2),
                createRawDetection(0, 0.75, 0.14, 0.14, 0.2, 0.2),
                createRawDetection(2, 0.65, 0.5, 0.5, 0.2, 0.2),
                createRawDetection(3, 0.55, 0.6, 0.6, 0.2, 0.2),
                createRawDetection(2, 0.45, 0.52, 0.52, 0.2, 0.2),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(4)
        })

        it("should handle boxes at image boundaries", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.0, 0.0, 0.5, 0.5),
                createRawDetection(0, 0.8, 0.1, 0.1, 0.5, 0.5),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(1)
            expect(result[0].confidence).toBe(0.9)
        })

        it("should handle small boxes", () => {
            const detections: RawDetection[] = [
                createRawDetection(0, 0.9, 0.5, 0.5, 0.01, 0.01),
                createRawDetection(0, 0.8, 0.51, 0.51, 0.01, 0.01),
                createRawDetection(0, 0.7, 0.7, 0.7, 0.01, 0.01),
            ]
            const result = applyNMS(detections, 0.45)
            expect(result.length).toBe(3)
        })
    })
})

describe("convertToDetections", () => {
    it("should convert raw detections to Detection objects", () => {
        const rawDetections: RawDetection[] = [
            createRawDetection(0, 0.9, 0.1, 0.2, 0.3, 0.4),
            createRawDetection(2, 0.8, 0.5, 0.6, 0.2, 0.3),
        ]
        const result = convertToDetections(rawDetections, 640, 640)
        expect(result.length).toBe(2)
        expect(result[0].id).toBe("det_000")
        expect(result[0].diseaseClass).toBe(DISEASE_CLASSES[0])
        expect(result[0].confidence).toBe(0.9)
        expect(result[0].boundingBox.x).toBe(0.1)
        expect(result[0].boundingBox.y).toBe(0.2)
        expect(result[0].boundingBox.width).toBe(0.3)
        expect(result[0].boundingBox.height).toBe(0.4)
        expect(result[1].id).toBe("det_001")
        expect(result[1].diseaseClass).toBe(DISEASE_CLASSES[2])
    })

    it("should clamp bounding box values to valid range", () => {
        const rawDetections: RawDetection[] = [
            createRawDetection(0, 0.9, -0.1, 0.1, 1.5, 0.2),
        ]
        const result = convertToDetections(rawDetections, 640, 640)
        expect(result[0].boundingBox.x).toBe(0)
        expect(result[0].boundingBox.width).toBe(1)
    })

    it("should handle empty input", () => {
        const result = convertToDetections([], 640, 640)
        expect(result.length).toBe(0)
    })

    it("should assign unique IDs to each detection", () => {
        const rawDetections: RawDetection[] = [
            createRawDetection(0, 0.9, 0.1, 0.1, 0.2, 0.2),
            createRawDetection(1, 0.8, 0.3, 0.3, 0.2, 0.2),
            createRawDetection(2, 0.7, 0.5, 0.5, 0.2, 0.2),
        ]
        const result = convertToDetections(rawDetections, 640, 640)
        expect(result[0].id).toBe("det_000")
        expect(result[1].id).toBe("det_001")
        expect(result[2].id).toBe("det_002")
    })
})
