import * as FileSystem from "expo-file-system/legacy"
import { storage } from "@/lib/storage"
import type { HistoryItem } from "./types"

const STORAGE_KEY = "mina_history_items"
const HISTORY_DIR = `${(FileSystem as { documentDirectory?: string }).documentDirectory ?? ""}history/`

// Ensure history directory exists
async function ensureHistoryDirectory() {
    const dirInfo = await FileSystem.getInfoAsync(HISTORY_DIR)
    if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(HISTORY_DIR, {
            intermediates: true,
        })
    }
}

export async function saveHistoryItem(
    item: Omit<HistoryItem, "id">,
): Promise<HistoryItem> {
    await ensureHistoryDirectory()

    const id = crypto.randomUUID()
    const itemDir = `${HISTORY_DIR}${id}/`

    await FileSystem.makeDirectoryAsync(itemDir, { intermediates: true })

    // Move images to permanent storage
    const originalFilename =
        item.originalImageUri.split("/").pop() ?? "original.jpg"
    const processedFilename =
        item.processedImageUri.split("/").pop() ?? "processed.png"

    const newOriginalUri = `${itemDir}${originalFilename}`
    const newProcessedUri = `${itemDir}${processedFilename}`

    // Copy instead of move to be safe with cache files
    await FileSystem.copyAsync({
        from: item.originalImageUri,
        to: newOriginalUri,
    })

    await FileSystem.copyAsync({
        from: item.processedImageUri,
        to: newProcessedUri,
    })

    const newItem: HistoryItem = {
        ...item,
        id,
        originalImageUri: newOriginalUri,
        processedImageUri: newProcessedUri,
    }

    // Update storage
    const currentItems = await getHistoryItems()
    const newItems = [newItem, ...currentItems]

    storage.set(STORAGE_KEY, JSON.stringify(newItems))

    return newItem
}

export async function getHistoryItems(): Promise<HistoryItem[]> {
    try {
        const json = storage.getString(STORAGE_KEY)
        return json ? JSON.parse(json) : []
    } catch (e) {
        console.error("Failed to load history items", e)
        return []
    }
}

export async function getHistoryItem(id: string): Promise<HistoryItem | null> {
    const items = await getHistoryItems()
    return items.find((item) => item.id === id) ?? null
}

export async function deleteHistoryItem(id: string): Promise<void> {
    const items = await getHistoryItems()
    const itemToDelete = items.find((item) => item.id === id)

    if (itemToDelete) {
        // Delete files
        try {
            // The directory containing the images
            const itemDir = `${HISTORY_DIR}${id}/`
            await FileSystem.deleteAsync(itemDir, { idempotent: true })
        } catch (e) {
            console.warn("Failed to delete history files", e)
        }
    }

    const newItems = items.filter((item) => item.id !== id)
    storage.set(STORAGE_KEY, JSON.stringify(newItems))
}
