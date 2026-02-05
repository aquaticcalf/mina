/**
 * Inference service for running TFLite model and processing detections.
 *
 * Since the TFLite model is exported with nms=False, this service implements
 * Non-Maximum Suppression (NMS) to filter overlapping detections.
 */

import {
    loadTensorflowModel,
    type TensorflowModel,
} from "react-native-fast-tflite"
import * as ImageManipulator from "expo-image-manipulator"
import * as FileSystem from "expo-file-system/legacy"
import { Skia } from "@shopify/react-native-skia"
import { DISEASE_CLASSES } from "@/lib/model/types"
import type { BoundingBox, Detection, DiseaseClass } from "@/lib/model/types"
import { createComponentLogger, logTiming } from "@/lib/log"

const logger = createComponentLogger("model/inference")

const IMAGE_SIZE = 640
const CONFIDENCE_THRESHOLD = 0.3
const IOU_THRESHOLD = 0.45

// Cached model instance to avoid reloading
let cachedModel: TensorflowModel | null = null
let cachedModelPath: string | null = null

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

/**
 * Load TFLite model from file path.
 * Model is cached to avoid reloading on subsequent calls.
 *
 * @param modelPath - Local file path to the TFLite model
 * @returns Loaded TensorflowModel instance
 */
export async function loadModel(modelPath: string): Promise<TensorflowModel> {
    // Return cached model if already loaded for this path
    if (cachedModel && cachedModelPath === modelPath) {
        logger.debug("Using cached model", { modelPath })
        return cachedModel
    }

    logger.info("Loading TFLite model", { modelPath })

    // Load new model
    try {
        const model = await loadTensorflowModel({
            url: `file://${modelPath}`,
        })
        cachedModel = model
        cachedModelPath = modelPath
        logger.info("Model loaded successfully", { modelPath })
        return model
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        logger.error(
            "Failed to load TFLite model",
            error instanceof Error ? error : new Error(message),
            { modelPath },
        )
        throw new Error(`Failed to load TFLite model: ${message}`)
    }
}

/**
 * Check if a model is currently loaded and cached.
 *
 * @returns True if model is loaded
 */
export function isModelLoaded(): boolean {
    return cachedModel !== null
}

/**
 * Dispose of the cached model and free resources.
 */
export function disposeModel(): void {
    logger.info("Disposing cached model")
    cachedModel = null
    cachedModelPath = null
}

/**
 * Preprocess image for model input.
 *
 * Decodes the image to RGB pixel data using react-native-skia.
 * 1. Resize image to 640x640 using expo-image-manipulator
 * 2. Load image with Skia and decode to RGBA pixels
 * 3. Convert RGBA to RGB by dropping alpha channel
 *
 * @param imageUri - URI to the image file
 * @returns RGB tensor (640x640x3) as Uint8Array
 */
async function preprocessImage(imageUri: string): Promise<Uint8Array> {
    logger.debug("Preprocessing image", { imageUri })

    try {
        // Step 1: Resize image to 640x640
        const resized = await ImageManipulator.manipulateAsync(
            imageUri,
            [{ resize: { width: IMAGE_SIZE, height: IMAGE_SIZE } }],
            {
                compress: 1.0,
                format: ImageManipulator.SaveFormat.PNG,
            },
        )

        logger.debug("Image resized", {
            originalUri: imageUri,
            resizedUri: resized.uri,
        })

        // Step 2: Read image file as base64
        const base64 = await FileSystem.readAsStringAsync(resized.uri, {
            encoding: "base64",
        })

        // Step 3: Decode image with Skia
        const imageData = Skia.Data.fromBase64(base64)
        const image = Skia.Image.MakeImageFromEncoded(imageData)

        if (!image) {
            throw new Error("Failed to decode image with Skia")
        }

        logger.debug("Image decoded with Skia", {
            width: image.width(),
            height: image.height(),
        })

        // Step 4: Read pixels as RGBA (returns Uint8Array with 4 bytes per pixel)
        const pixels = image.readPixels()
        if (!pixels) {
            throw new Error("Failed to read pixels from image")
        }

        // Step 5: Convert RGBA to RGB by dropping alpha channel
        // YOLO expects RGB format (3 channels)
        const tensorSize = IMAGE_SIZE * IMAGE_SIZE * 3
        const rgbTensor = new Uint8Array(tensorSize)

        for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
            const rgbaIndex = i * 4
            const rgbIndex = i * 3

            rgbTensor[rgbIndex] = pixels[rgbaIndex] // R
            rgbTensor[rgbIndex + 1] = pixels[rgbaIndex + 1] // G
            rgbTensor[rgbIndex + 2] = pixels[rgbaIndex + 2] // B
            // Skip alpha channel (pixels[rgbaIndex + 3])
        }

        logger.debug("Image preprocessed successfully", { tensorSize })
        return rgbTensor
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        logger.error(
            "Image preprocessing failed",
            error instanceof Error ? error : new Error(message),
            { imageUri },
        )
        throw new Error(`Failed to preprocess image: ${message}`)
    }
}

/**
 * Run inference on an image using the TFLite model.
 *
 * @param imageUri - URI to the image file
 * @param modelPath - Local file path to the TFLite model
 * @returns Inference result with detections and timing
 */
export async function runInference(
    imageUri: string,
    modelPath: string,
): Promise<InferenceResult> {
    const startTime = Date.now()
    logger.info("Starting inference", { imageUri, modelPath })

    try {
        // Step 1: Load model
        const model = await loadModel(modelPath)

        // Step 2: Preprocess image
        const inputTensor = await preprocessImage(imageUri)

        // Step 3: Run model inference
        logger.debug("Running model inference")
        const outputs = model.runSync([inputTensor])

        // Step 4: Parse YOLO output
        // The model outputs raw detections in YOLO format
        const rawOutput = outputs[0] as Float32Array
        logger.debug("Raw output received", { outputLength: rawOutput.length })

        const rawDetections = parseYolov8Output(
            rawOutput,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
        logger.debug("YOLO output parsed", {
            rawDetectionsCount: rawDetections.length,
        })

        // Step 5: Apply NMS to remove overlapping boxes
        const nmsDetections = applyNMS(rawDetections, IOU_THRESHOLD)
        logger.debug("NMS applied", { afterNMSCount: nmsDetections.length })

        // Step 6: Filter by confidence threshold
        const filteredDetections = filterByConfidence(
            nmsDetections,
            CONFIDENCE_THRESHOLD,
        )
        logger.debug("Filtered by confidence", {
            afterFilterCount: filteredDetections.length,
        })

        // Step 7: Sort by confidence (descending)
        const sortedDetections = sortByConfidence(filteredDetections)

        // Step 8: Convert to Detection objects
        const detections = convertToDetections(
            sortedDetections,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        const inferenceTimeMs = Date.now() - startTime

        logger.info("Inference completed", {
            inferenceTimeMs,
            detectionsCount: detections.length,
            imageUri,
        })

        return {
            detections,
            inferenceTimeMs,
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        logger.error(
            "Inference failed",
            error instanceof Error ? error : new Error(message),
            { imageUri },
        )
        throw new Error(`Inference failed: ${message}`)
    }
}
