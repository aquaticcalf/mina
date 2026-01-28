/**
 * Local storage utilities for model files and metadata.
 */

import * as FileSystem from "expo-file-system/legacy"
import { storage } from "@/lib/storage"
import type { ModelChannel, ModelMetadata } from "@/lib/model/types"

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
    const dirInfo = await FileSystem.getInfoAsync(dir)
    if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(dir, { intermediates: true })
    }
}

export async function saveModelMetadata(
    metadata: ModelMetadata,
): Promise<void> {
    storage.set(STORAGE_KEYS.METADATA, JSON.stringify(metadata))
}

export async function loadModelMetadata(): Promise<ModelMetadata | null> {
    const data = storage.getString(STORAGE_KEYS.METADATA)
    if (!data) return null

    try {
        return JSON.parse(data) as ModelMetadata
    } catch {
        return null
    }
}

export async function clearModelMetadata(): Promise<void> {
    storage.delete(STORAGE_KEYS.METADATA)
}

export async function modelFileExists(channel: ModelChannel): Promise<boolean> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    return info.exists
}

export async function deleteModelFile(channel: ModelChannel): Promise<void> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists) {
        await FileSystem.deleteAsync(path)
    }
}

export async function getModelFileSize(
    channel: ModelChannel,
): Promise<number | null> {
    const path = getModelPath(channel)
    const info = await FileSystem.getInfoAsync(path)
    if (info.exists && "size" in info) {
        return info.size
    }
    return null
}
