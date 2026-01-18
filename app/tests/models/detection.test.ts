import { describe, it, expect } from "bun:test"
import {
    type BoundingBox,
    type Detection,
    type DetectionSession,
    type DiseaseClass,
    type DiseaseInfo,
    DISEASE_CLASSES,
    isValidBoundingBox,
    isValidConfidence,
    isValidDiseaseClass,
    parseSession,
    serializeSession,
    validateBoundingBox,
    validateDetection,
    validateDetectionSession,
    validateDiseaseInfo,
} from "@/lib/model/types"

describe("**Feature: fish-disease-detection, Property 5: Detection result structure**", () => {
    describe("BoundingBox validation", () => {
        it("should validate valid bounding boxes", () => {
            const validBbox: BoundingBox = {
                x: 0.25,
                y: 0.3,
                width: 0.4,
                height: 0.35,
            }
            expect(isValidBoundingBox(validBbox)).toBe(true)
            expect(validateBoundingBox(validBbox)).toEqual([])
        })

        it("should reject bounding boxes with out-of-range values", () => {
            const invalidBbox: BoundingBox = {
                x: 1.5,
                y: 0.3,
                width: 0.4,
                height: 0.35,
            }
            expect(isValidBoundingBox(invalidBbox)).toBe(false)
            expect(validateBoundingBox(invalidBbox).length).toBeGreaterThan(0)
        })

        it("should reject bounding boxes that exceed image bounds", () => {
            const invalidBbox: BoundingBox = {
                x: 0.8,
                y: 0.3,
                width: 0.4,
                height: 0.35,
            }
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
                boundingBox: { x: 0.25, y: 0.3, width: 0.4, height: 0.35 },
            }
            expect(validateDetection(validDetection)).toEqual([])
        })

        it("should detect multiple validation errors", () => {
            const invalidDetection = {
                id: "det_001",
                diseaseClass: "invalid_class" as DiseaseClass,
                confidence: 1.5,
                boundingBox: { x: 1.5, y: 0.3, width: 0.4, height: 0.35 },
            } as Detection
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
                        boundingBox: {
                            x: 0.25,
                            y: 0.3,
                            width: 0.4,
                            height: 0.35,
                        },
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
            } as unknown as DiseaseInfo
            const errors = validateDiseaseInfo(invalidInfo)
            expect(errors.length).toBeGreaterThanOrEqual(5)
        })
    })
})

describe("**Feature: fish-disease-detection, Property 1: Detection session round-trip**", () => {
    describe("Serialization", () => {
        it("should serialize a valid session to JSON", () => {
            const session: DetectionSession = {
                id: "session_123",
                imageUri: "file:///path/to/image.jpg",
                detections: [
                    {
                        id: "det_001",
                        diseaseClass: "bacterial_infection",
                        confidence: 0.87,
                        boundingBox: {
                            x: 0.25,
                            y: 0.3,
                            width: 0.4,
                            height: 0.35,
                        },
                    },
                    {
                        id: "det_002",
                        diseaseClass: "healthy",
                        confidence: 0.95,
                        boundingBox: {
                            x: 0.1,
                            y: 0.1,
                            width: 0.2,
                            height: 0.2,
                        },
                    },
                ],
                timestamp: 1732550400000,
            }
            const json = serializeSession(session)
            expect(json).toContain("session_123")
            expect(json).toContain("file:///path/to/image.jpg")
            expect(json).toContain("bacterial_infection")
            expect(() => JSON.parse(json)).not.toThrow()
        })

        it("should handle sessions with empty detections", () => {
            const session: DetectionSession = {
                id: "session_empty",
                imageUri: "file:///path/to/empty.jpg",
                detections: [],
                timestamp: 1732550400000,
            }
            const json = serializeSession(session)
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.detections).toEqual([])
        })

        it("should handle sessions with single detection", () => {
            const session: DetectionSession = {
                id: "session_single",
                imageUri: "file:///path/to/single.jpg",
                detections: [
                    {
                        id: "det_single",
                        diseaseClass: "parasite",
                        confidence: 0.72,
                        boundingBox: {
                            x: 0.5,
                            y: 0.5,
                            width: 0.3,
                            height: 0.3,
                        },
                    },
                ],
                timestamp: 1732550400000,
            }
            const json = serializeSession(session)
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.detections.length).toBe(1)
        })
    })

    describe("Parsing", () => {
        it("should parse valid JSON back to session", () => {
            const original: DetectionSession = {
                id: "session_123",
                imageUri: "file:///path/to/image.jpg",
                detections: [
                    {
                        id: "det_001",
                        diseaseClass: "fungal_infection",
                        confidence: 0.65,
                        boundingBox: {
                            x: 0.2,
                            y: 0.25,
                            width: 0.35,
                            height: 0.4,
                        },
                    },
                ],
                timestamp: 1732550400000,
            }
            const json = serializeSession(original)
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.id).toBe(original.id)
            expect(parsed?.imageUri).toBe(original.imageUri)
            expect(parsed?.timestamp).toBe(original.timestamp)
            expect(parsed?.detections.length).toBe(original.detections.length)
            expect(parsed?.detections[0].id).toBe(original.detections[0].id)
            expect(parsed?.detections[0].diseaseClass).toBe(
                original.detections[0].diseaseClass,
            )
            expect(parsed?.detections[0].confidence).toBe(
                original.detections[0].confidence,
            )
            expect(parsed?.detections[0].boundingBox).toEqual(
                original.detections[0].boundingBox,
            )
        })

        it("should return null for malformed JSON", () => {
            expect(parseSession("invalid json")).toBeNull()
            expect(parseSession("{")).toBeNull()
            expect(parseSession("")).toBeNull()
        })

        it("should return null for JSON with missing required fields", () => {
            expect(parseSession('{"id": "test"}')).toBeNull()
            expect(parseSession('{"imageUri": "test.jpg"}')).toBeNull()
            expect(parseSession('{"timestamp": 123}')).toBeNull()
            expect(
                parseSession(
                    '{"id": "", "imageUri": "", "detections": [], "timestamp": 0}',
                ),
            ).toBeNull()
        })

        it("should return null for JSON with invalid detection data", () => {
            const invalidJson = JSON.stringify({
                id: "session_123",
                imageUri: "file:///test.jpg",
                detections: [
                    {
                        id: "det_001",
                        diseaseClass: "invalid_class",
                        confidence: 1.5,
                        boundingBox: {
                            x: 1.5,
                            y: 0.3,
                            width: 0.4,
                            height: 0.35,
                        },
                    },
                ],
                timestamp: 1732550400000,
            })
            const parsed = parseSession(invalidJson)
            expect(parsed).not.toBeNull()
            expect(parsed?.detections.length).toBe(0)
        })

        it("should filter out invalid detections but keep valid ones", () => {
            const json = JSON.stringify({
                id: "session_mixed",
                imageUri: "file:///test.jpg",
                detections: [
                    {
                        id: "det_valid",
                        diseaseClass: "white_tail",
                        confidence: 0.8,
                        boundingBox: {
                            x: 0.1,
                            y: 0.1,
                            width: 0.2,
                            height: 0.2,
                        },
                    },
                    {
                        id: "det_invalid",
                        diseaseClass: "not_a_disease",
                        confidence: 0.9,
                        boundingBox: {
                            x: 0.5,
                            y: 0.5,
                            width: 0.1,
                            height: 0.1,
                        },
                    },
                ],
                timestamp: 1732550400000,
            })
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.detections.length).toBe(1)
            expect(parsed?.detections[0].id).toBe("det_valid")
        })
    })

    describe("Round-trip", () => {
        it("should maintain data integrity through serialize/parse cycle", () => {
            const original: DetectionSession = {
                id: "session_roundtrip",
                imageUri: "file:///path/to/roundtrip.jpg",
                detections: [
                    {
                        id: "det_1",
                        diseaseClass: "bacterial_infection",
                        confidence: 0.87,
                        boundingBox: {
                            x: 0.25,
                            y: 0.3,
                            width: 0.4,
                            height: 0.35,
                        },
                    },
                    {
                        id: "det_2",
                        diseaseClass: "fungal_infection",
                        confidence: 0.72,
                        boundingBox: {
                            x: 0.6,
                            y: 0.7,
                            width: 0.25,
                            height: 0.2,
                        },
                    },
                    {
                        id: "det_3",
                        diseaseClass: "healthy",
                        confidence: 0.95,
                        boundingBox: {
                            x: 0.1,
                            y: 0.1,
                            width: 0.15,
                            height: 0.15,
                        },
                    },
                ],
                timestamp: 1732550400000,
            }
            const json = serializeSession(original)
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.id).toBe(original.id)
            expect(parsed?.imageUri).toBe(original.imageUri)
            expect(parsed?.timestamp).toBe(original.timestamp)
            expect(parsed?.detections).toHaveLength(original.detections.length)
            for (let i = 0; i < original.detections.length; i++) {
                expect(parsed?.detections[i].id).toBe(original.detections[i].id)
                expect(parsed?.detections[i].diseaseClass).toBe(
                    original.detections[i].diseaseClass,
                )
                expect(parsed?.detections[i].confidence).toBe(
                    original.detections[i].confidence,
                )
                expect(parsed?.detections[i].boundingBox.x).toBe(
                    original.detections[i].boundingBox.x,
                )
                expect(parsed?.detections[i].boundingBox.y).toBe(
                    original.detections[i].boundingBox.y,
                )
                expect(parsed?.detections[i].boundingBox.width).toBe(
                    original.detections[i].boundingBox.width,
                )
                expect(parsed?.detections[i].boundingBox.height).toBe(
                    original.detections[i].boundingBox.height,
                )
            }
        })

        it("should handle special characters in IDs and URIs", () => {
            const original: DetectionSession = {
                id: "session_with-special_chars-123",
                imageUri: "file:///path/with%20spaces/file%26name.jpg",
                detections: [
                    {
                        id: "det_特殊",
                        diseaseClass: "parasite",
                        confidence: 0.5,
                        boundingBox: {
                            x: 0.1,
                            y: 0.1,
                            width: 0.8,
                            height: 0.8,
                        },
                    },
                ],
                timestamp: 1732550400000,
            }
            const json = serializeSession(original)
            const parsed = parseSession(json)
            expect(parsed).not.toBeNull()
            expect(parsed?.id).toBe(original.id)
            expect(parsed?.imageUri).toBe(original.imageUri)
            expect(parsed?.detections[0].id).toBe(original.detections[0].id)
        })
    })
})
