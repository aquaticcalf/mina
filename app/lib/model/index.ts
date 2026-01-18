/**
 * Model management module.
 *
 * Handles downloading, updating, and loading TFLite models from GitHub releases.
 */

// Types
export type {
    BoundingBox,
    Detection,
    DetectionSession,
    DiseaseClass,
    DiseaseInfo,
    GitHubAsset,
    GitHubRelease,
    ModelChannel,
    ModelMetadata,
    ModelState,
    Severity,
    UpdateCheckResult,
} from "@/lib/model/types"

// Validation functions
export {
    isValidBoundingBox,
    isValidConfidence,
    isValidDiseaseClass,
    validateBoundingBox,
    validateDetection,
    validateDetectionSession,
    validateDiseaseInfo,
} from "@/lib/model/types"

// Disease classes constant
export { DISEASE_CLASSES } from "@/lib/model/types"

// Serialization functions
export { parseSession, serializeSession } from "@/lib/model/types"

// Manager functions
export {
    checkForUpdate,
    downloadModel,
    forceUpdateModel,
    getLocalModelPath,
    getModelChannel,
    initializeModel,
    type DownloadProgressCallback,
} from "@/lib/model/manager"

// Storage utilities
export {
    clearAll,
    clearModelMetadata,
    deleteModelFile,
    deleteSession,
    ensureModelDirectory,
    getModelDirectory,
    getModelFileSize,
    getModelPath,
    getSession,
    getSessions,
    loadModelMetadata,
    modelFileExists,
    saveModelMetadata,
    saveSession,
} from "@/lib/model/sessions"
export {
    clearModelMetadata,
    deleteModelFile,
    ensureModelDirectory,
    getModelDirectory,
    getModelFileSize,
    getModelPath,
    loadModelMetadata,
    modelFileExists,
    saveModelMetadata,
} from "@/lib/model/storage"

// React context
export { ModelProvider, useModel } from "@/lib/model/context"
