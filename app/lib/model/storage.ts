/**
 * Local storage utilities for model files and metadata.
 */

import * as FileSystem from "expo-file-system/legacy"
import { storage } from "@/lib/storage"
import type { ModelChannel, ModelMetadata } from "@/lib/model/types"
import { createComponentLogger } from "@/lib/log"

const logger = createComponentLogger("model/storage")

const STORAGE_KEYS = {
    METADATA: "mina_model_metadata",
} as const

export function getModelDirectory(): string {
    const fs = FileSystem as { documentDirectory?: string }
    return `${fs.documentDirectory ?? ""}models/`
}

export function getModelPath(channel: ModelChannel): string {
    return `${getModelDirectory()}${channel}_model.tflite`
}

export async function ensureModelDirectory(): Promise<void> {
    const dir = getModelDirectory()
    logger.debug("Ensuring model directory exists", { dir })
    const dirInfo = await FileSystem.getInfoAsync(dir)
    if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(dir, { intermediates: true })
        logger.info("Created model directory", { dir })
    }
}

export async function saveModelMetadata(
    metadata: ModelMetadata,
): Promise<void> {
    try {
        storage.set(STORAGE_KEYS.METADATA, JSON.stringify(metadata))
        logger.info("Model metadata saved", {
            channel: metadata.channel,
            updatedAt: metadata.updatedAt,
        })
    } catch (error) {
        logger.error(
            "Failed to save model metadata",
            error instanceof Error ? error : new Error(String(error)),
        )
        throw error
    }
}

export async function loadModelMetadata(): Promise<ModelMetadata | null> {
    try {
        const data = storage.getString(STORAGE_KEYS.METADATA)
        if (!data) {
            logger.debug("No model metadata found in storage")
            return null
        }

        const metadata = JSON.parse(data) as ModelMetadata
        logger.debug("Model metadata loaded", { channel: metadata.channel })
        return metadata
    } catch (error) {
        logger.error(
            "Failed to load model metadata",
            error instanceof Error ? error : new Error(String(error)),
        )
        return null
    }
}

export async function clearModelMetadata(): Promise<void> {
    try {
        storage.remove(STORAGE_KEYS.METADATA)
        logger.info("Model metadata cleared")
    } catch (error) {
        logger.error(
            "Failed to clear model metadata",
            error instanceof Error ? error : new Error(String(error)),
        )
        throw error
    }
}

export async function modelFileExists(channel: ModelChannel): Promise<boolean> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    const exists = info.exists
    logger.debug("Checked model file existence", { channel, path, exists })
    return exists
}

export async function deleteModelFile(channel: ModelChannel): Promise<void> {
    const path = getModelPath(channel)
    logger.info("Deleting model file", { channel, path })
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists) {
        await FileSystem.deleteAsync(path)
        logger.info("Model file deleted", { channel, path })
    } else {
        logger.warn("Model file not found for deletion", { channel, path })
    }
}

export async function getModelFileSize(
    channel: ModelChannel,
): Promise<number | null> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists && "size" in info) {
        logger.debug("Got model file size", { channel, size: info.size })
        return info.size
    }
    logger.debug("Model file size not available", {
        channel,
        exists: info.exists,
    })
    return null
}
