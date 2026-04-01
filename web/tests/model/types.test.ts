import { describe, it, expect } from "bun:test"
import {
  type BoundingBox,
  DISEASE_CLASSES,
  isValidBoundingBox,
  isValidConfidence,
  isValidDiseaseClass,
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
    })

    it("should reject bounding boxes with out-of-range values", () => {
      const invalidBbox: BoundingBox = {
        x: 1.5,
        y: 0.3,
        width: 0.4,
        height: 0.35,
      }
      expect(isValidBoundingBox(invalidBbox)).toBe(false)
    })

    it("should reject bounding boxes with negative values", () => {
      const invalidBbox: BoundingBox = {
        x: -0.1,
        y: 0.3,
        width: 0.4,
        height: 0.35,
      }
      expect(isValidBoundingBox(invalidBbox)).toBe(false)
    })

    it("should reject bounding boxes that exceed image bounds", () => {
      const invalidBbox: BoundingBox = {
        x: 0.8,
        y: 0.3,
        width: 0.4,
        height: 0.35,
      }
      expect(isValidBoundingBox(invalidBbox)).toBe(false)
    })

    it("should accept edge cases at boundaries", () => {
      const edgeBbox: BoundingBox = { x: 0, y: 0, width: 1, height: 1 }
      expect(isValidBoundingBox(edgeBbox)).toBe(true)
    })

    it("should accept bounding box at top-left corner", () => {
      const bbox: BoundingBox = { x: 0, y: 0, width: 0.5, height: 0.5 }
      expect(isValidBoundingBox(bbox)).toBe(true)
    })

    it("should accept bounding box at bottom-right corner", () => {
      const bbox: BoundingBox = {
        x: 0.5,
        y: 0.5,
        width: 0.5,
        height: 0.5,
      }
      expect(isValidBoundingBox(bbox)).toBe(true)
    })
  })

  describe("DiseaseClass validation", () => {
    it("should accept all valid disease classes", () => {
      for (const diseaseClass of DISEASE_CLASSES) {
        expect(isValidDiseaseClass(diseaseClass)).toBe(true)
      }
    })

    it("should reject invalid disease classes", () => {
      expect(isValidDiseaseClass("unknown_disease")).toBe(false)
      expect(isValidDiseaseClass("")).toBe(false)
    })

    it("should reject non-string values", () => {
      expect(isValidDiseaseClass(123 as unknown as string)).toBe(false)
      expect(isValidDiseaseClass(null as unknown as string)).toBe(false)
      expect(isValidDiseaseClass(undefined as unknown as string)).toBe(false)
    })

    it("should be case-sensitive", () => {
      expect(isValidDiseaseClass("BACTERIAL_INFECTION")).toBe(false)
      expect(isValidDiseaseClass("Bacterial_Infection")).toBe(false)
    })
  })

  describe("Confidence validation", () => {
    it("should accept valid confidence values", () => {
      expect(isValidConfidence(0.0)).toBe(true)
      expect(isValidConfidence(0.5)).toBe(true)
      expect(isValidConfidence(1.0)).toBe(true)
    })

    it("should accept confidence at boundaries", () => {
      expect(isValidConfidence(0)).toBe(true)
      expect(isValidConfidence(1)).toBe(true)
    })

    it("should reject out-of-range confidence values", () => {
      expect(isValidConfidence(-0.1)).toBe(false)
      expect(isValidConfidence(1.5)).toBe(false)
      expect(isValidConfidence(-1)).toBe(false)
      expect(isValidConfidence(2)).toBe(false)
    })

    it("should reject non-numeric values", () => {
      expect(isValidConfidence(NaN)).toBe(false)
      expect(isValidConfidence(Number.POSITIVE_INFINITY)).toBe(false)
      expect(isValidConfidence(Number.NEGATIVE_INFINITY)).toBe(false)
    })
  })
})
