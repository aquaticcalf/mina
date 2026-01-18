/**
 * Type definitions for model management.
 */

/** Model distribution channel */
export type ModelChannel = "dev" | "prod"

/** Metadata about a model release */
export interface ModelMetadata {
    /** Release channel (dev or prod) */
    channel: ModelChannel
    /** ISO date string when the release was updated */
    updatedAt: string
    /** Model file size in bytes */
    sizeBytes: number
    /** URL the model was downloaded from */
    downloadUrl: string
}

/** Current state of model management */
export interface ModelState {
    /** Whether the model is ready for inference */
    isReady: boolean
    /** Whether a download/update is in progress */
    isLoading: boolean
    /** Download progress (0-1) */
    progress: number
    /** Error message if something went wrong */
    error: string | null
    /** Metadata about the currently loaded model */
    metadata: ModelMetadata | null
}

/** Result of checking for model updates */
export interface UpdateCheckResult {
    /** Whether an update is available */
    hasUpdate: boolean
    /** The new release date if update available */
    newDate: string | null
    /** Current local release date */
    currentDate: string | null
}

/** GitHub release API response (simplified) */
export interface GitHubRelease {
    tag_name: string
    name: string
    body: string
    published_at: string
    assets: GitHubAsset[]
}

/** GitHub release asset */
export interface GitHubAsset {
    name: string
    browser_download_url: string
    size: number
}

// Disease Classes (matches Python model/mina/core/constants.py)
export type DiseaseClass =
    | "bacterial_infection"
    | "fungal_infection"
    | "healthy"
    | "parasite"
    | "white_tail"

export const DISEASE_CLASSES: DiseaseClass[] = [
    "bacterial_infection",
    "fungal_infection",
    "healthy",
    "parasite",
    "white_tail",
]

// Bounding Box
export interface BoundingBox {
    x: number
    y: number
    width: number
    height: number
}

// Detection
export interface Detection {
    id: string
    diseaseClass: DiseaseClass
    confidence: number
    boundingBox: BoundingBox
}

// Detection Session
export interface DetectionSession {
    id: string
    imageUri: string
    detections: Detection[]
    timestamp: number
}

// Disease Severity
export type Severity = "low" | "medium" | "high"

// Disease Info
export interface DiseaseInfo {
    diseaseClass: DiseaseClass
    displayName: string
    description: string
    symptoms: string[]
    treatments: string[]
    severity: Severity
}

// Validation Functions
export function isValidBoundingBox(bbox: BoundingBox): boolean {
    return (
        bbox.x >= 0 &&
        bbox.x <= 1 &&
        bbox.y >= 0 &&
        bbox.y <= 1 &&
        bbox.width >= 0 &&
        bbox.width <= 1 &&
        bbox.height >= 0 &&
        bbox.height <= 1 &&
        bbox.x + bbox.width <= 1 + 1e-6 &&
        bbox.y + bbox.height <= 1 + 1e-6
    )
}

export function isValidDiseaseClass(value: string): value is DiseaseClass {
    return DISEASE_CLASSES.includes(value as DiseaseClass)
}

export function isValidConfidence(value: number): boolean {
    return value >= 0 && value <= 1
}

export function validateBoundingBox(bbox: BoundingBox): string[] {
    const errors: string[] = []
    if (!(bbox.x >= 0 && bbox.x <= 1)) errors.push(`BBox x out of range: ${bbox.x}`)
    if (!(bbox.y >= 0 && bbox.y <= 1)) errors.push(`BBox y out of range: ${bbox.y}`)
    if (!(bbox.width >= 0 && bbox.width <= 1))
        errors.push(`BBox width out of range: ${bbox.width}`)
    if (!(bbox.height >= 0 && bbox.height <= 1))
        errors.push(`BBox height out of range: ${bbox.height}`)
    if (bbox.x + bbox.width > 1.0 + 1e-6)
        errors.push(`BBox exceeds right edge: x=${bbox.x}, width=${bbox.width}`)
    if (bbox.y + bbox.height > 1.0 + 1e-6)
        errors.push(`BBox exceeds bottom edge: y=${bbox.y}, height=${bbox.height}`)
    return errors
}

export function validateDetection(detection: Detection): string[] {
    const errors: string[] = []
    if (!isValidDiseaseClass(detection.diseaseClass))
        errors.push(`Invalid disease class: ${detection.diseaseClass}`)
    if (!isValidConfidence(detection.confidence))
        errors.push(`Confidence out of range: ${detection.confidence}`)
    errors.push(...validateBoundingBox(detection.boundingBox))
    return errors
}

export function validateDetectionSession(session: DetectionSession): string[] {
    const errors: string[] = []
    if (!session.id) errors.push("Session missing id")
    if (!session.imageUri) errors.push("Session missing imageUri")
    if (!Array.isArray(session.detections)) errors.push("Session missing detections array")
    if (typeof session.timestamp !== "number" || session.timestamp <= 0)
        errors.push(`Invalid timestamp: ${session.timestamp}`)
    for (const detection of session.detections) {
        errors.push(...validateDetection(detection))
    }
    return errors
}

export function validateDiseaseInfo(info: DiseaseInfo): string[] {
    const errors: string[] = []
    if (!isValidDiseaseClass(info.diseaseClass))
        errors.push(`Invalid disease class: ${info.diseaseClass}`)
    if (!info.displayName || info.displayName.trim().length === 0)
        errors.push("Missing displayName")
    if (!info.description || info.description.trim().length === 0)
        errors.push("Missing description")
    if (!Array.isArray(info.symptoms) || info.symptoms.length === 0)
        errors.push("Missing or empty symptoms array")
    if (!Array.isArray(info.treatments) || info.treatments.length === 0)
        errors.push("Missing or empty treatments array")
    if (!["low", "medium", "high"].includes(info.severity))
        errors.push(`Invalid severity: ${info.severity}`)
    return errors
}
