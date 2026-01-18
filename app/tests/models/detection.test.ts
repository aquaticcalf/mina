import { describe, it, expect } from "bun:test"
import {
    type BoundingBox,
    type Detection,
    type DiseaseClass,
    DISEASE_CLASSES,
    isValidBoundingBox,
    isValidConfidence,
    isValidDiseaseClass,
    validateBoundingBox,
    validateDetection,
    validateDetectionSession,
    validateDiseaseInfo,
} from "@/lib/model/types"

describe("**Feature: fish-disease-detection, Property 5: Detection result structure**", () => {
    describe("BoundingBox validation", () => {
        it("should validate valid bounding boxes", () => {
            const validBbox: BoundingBox = { x: 0.25, y: 0.3, width: 0.4, height: 0.35 }
            expect(isValidBoundingBox(validBbox)).toBe(true)
            expect(validateBoundingBox(validBbox)).toEqual([])
        })

        it("should reject bounding boxes with out-of-range values", () => {
            const invalidBbox: BoundingBox = { x: 1.5, y: 0.3, width: 0.4, height: 0.35 }
            expect(isValidBoundingBox(invalidBbox)).toBe(false)
            expect(validateBoundingBox(invalidBbox).length).toBeGreaterThan(0)
        })

        it("should reject bounding boxes that exceed image bounds", () => {
            const invalidBbox: BoundingBox = { x: 0.8, y: 0.3, width: 0.4, height: 0.35 }
            expect(isValidBoundingBox(invalidBbox)).toBe(false)
            const errors = validateBoundingBox(invalidBbox)
            expect(errors.some((e) => e.includes("exceeds"))).toBe(true)
        })

        it("should accept edge cases at boundaries", () => {
            const edgeBbox: BoundingBox = { x: 0, y: 0, width: 1, height: 1 }
            expect(isValidBoundingBox(edgeBbox)).toBe(true)
            expect(validateBoundingBox(edgeBbox)).toEqual([])
        })
    })

    describe("DiseaseClass validation", () => {
        it("should accept valid disease classes", () => {
            for (const diseaseClass of DISEASE_CLASSES) {
                expect(isValidDiseaseClass(diseaseClass)).toBe(true)
            }
        })

        it("should reject invalid disease classes", () => {
            expect(isValidDiseaseClass("unknown_disease")).toBe(false)
            expect(isValidDiseaseClass("")).toBe(false)
            expect(isValidDiseaseClass(123 as unknown as string)).toBe(false)
        })
    })

    describe("Confidence validation", () => {
        it("should accept valid confidence values", () => {
            expect(isValidConfidence(0.0)).toBe(true)
            expect(isValidConfidence(0.5)).toBe(true)
            expect(isValidConfidence(1.0)).toBe(true)
        })

        it("should reject out-of-range confidence values", () => {
            expect(isValidConfidence(-0.1)).toBe(false)
            expect(isValidConfidence(1.5)).toBe(false)
        })
    })

    describe("Detection validation", () => {
        it("should validate a correct detection", () => {
            const validDetection: Detection = {
                id: "det_001",
                diseaseClass: "bacterial_infection",
                confidence: 0.87,
                boundingBox: { x: 0.25, y: 0.30, width: 0.40, height: 0.35 },
            }
            expect(validateDetection(validDetection)).toEqual([])
        })

        it("should detect multiple validation errors", () => {
            const invalidDetection: Detection = {
                id: "det_001",
                diseaseClass: "invalid_class",
                confidence: 1.5,
                boundingBox: { x: 1.5, y: 0.3, width: 0.4, height: 0.35 },
            }
            const errors = validateDetection(invalidDetection)
            expect(errors.length).toBeGreaterThanOrEqual(3)
        })
    })

    describe("DetectionSession validation", () => {
        it("should validate a correct session", () => {
            const validSession = {
                id: "session_123",
                imageUri: "file:///path/to/image.jpg",
                detections: [
                    {
                        id: "det_001",
                        diseaseClass: "bacterial_infection" as DiseaseClass,
                        confidence: 0.87,
                        boundingBox: { x: 0.25, y: 0.30, width: 0.40, height: 0.35 },
                    },
                ],
                timestamp: 1732550400000,
            }
            expect(validateDetectionSession(validSession)).toEqual([])
        })

        it("should detect missing required fields", () => {
            const invalidSession = {
                id: "",
                imageUri: "",
                detections: [],
                timestamp: 0,
            }
            const errors = validateDetectionSession(invalidSession)
            expect(errors.length).toBeGreaterThanOrEqual(3)
        })
    })

    describe("DiseaseInfo validation", () => {
        it("should validate correct disease info", () => {
            const validInfo = {
                diseaseClass: "bacterial_infection" as DiseaseClass,
                displayName: "Bacterial Infection",
                description: "A bacterial infection affecting fish.",
                symptoms: ["white patches", "lethargy"],
                treatments: ["antibiotics", "water changes"],
                severity: "high" as const,
            }
            expect(validateDiseaseInfo(validInfo)).toEqual([])
        })

        it("should detect incomplete disease info", () => {
            const invalidInfo = {
                diseaseClass: "unknown" as DiseaseClass,
                displayName: "",
                description: "",
                symptoms: [],
                treatments: [],
                severity: "critical" as const,
            }
            const errors = validateDiseaseInfo(invalidInfo)
            expect(errors.length).toBeGreaterThanOrEqual(5)
        })
    })
})
