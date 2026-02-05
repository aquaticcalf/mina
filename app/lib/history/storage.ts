import * as FileSystem from "expo-file-system/legacy"
import { storage } from "@/lib/storage"
import type { HistoryItem } from "./types"
import { createComponentLogger } from "@/lib/log"

const logger = createComponentLogger("history/storage")

const STORAGE_KEY = "mina_history_items"
const documentDirectory = (FileSystem as { documentDirectory?: string })
    .documentDirectory

if (!documentDirectory) {
    throw new Error(
        "Document directory not available - cannot initialize history storage",
    )
}

const HISTORY_DIR = `${documentDirectory}history/`

// Ensure history directory exists
async function ensureHistoryDirectory() {
    const dirInfo = await FileSystem.getInfoAsync(HISTORY_DIR)
    if (!dirInfo.exists) {
        await FileSystem.makeDirectoryAsync(HISTORY_DIR, {
            intermediates: true,
        })
        logger.info("Created history directory", { dir: HISTORY_DIR })
    }
}

export async function saveHistoryItem(
    item: Omit<HistoryItem, "id">,
): Promise<HistoryItem> {
    logger.info("Saving history item", { timestamp: item.timestamp })

    try {
        await ensureHistoryDirectory()

        const id = crypto.randomUUID()
        const itemDir = `${HISTORY_DIR}${id}/`

        await FileSystem.makeDirectoryAsync(itemDir, { intermediates: true })
        logger.debug("Created item directory", { id, dir: itemDir })

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

        logger.debug("Images copied to permanent storage", {
            id,
            newOriginalUri,
            newProcessedUri,
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

        logger.info("History item saved successfully", {
            id,
            timestamp: item.timestamp,
            detectionsCount: item.results.detections.length,
        })

        return newItem
    } catch (error) {
        logger.error(
            "Failed to save history item",
            error instanceof Error ? error : new Error(String(error)),
            {
                timestamp: item.timestamp,
            },
        )
        throw error
    }
}

export async function getHistoryItems(): Promise<HistoryItem[]> {
    try {
        const json = storage.getString(STORAGE_KEY)
        if (!json) {
            logger.debug("No history items found in storage")
            return []
        }

        const items = JSON.parse(json) as HistoryItem[]
        logger.debug("Loaded history items", { count: items.length })
        return items
    } catch (error) {
        logger.error(
            "Failed to load history items",
            error instanceof Error ? error : new Error(String(error)),
        )
        return []
    }
}

export async function getHistoryItem(id: string): Promise<HistoryItem | null> {
    try {
        const items = await getHistoryItems()
        const item = items.find((item) => item.id === id) ?? null
        logger.debug("Retrieved history item", { id, found: !!item })
        return item
    } catch (error) {
        logger.error(
            "Failed to get history item",
            error instanceof Error ? error : new Error(String(error)),
            { id },
        )
        return null
    }
}

export async function deleteHistoryItem(id: string): Promise<void> {
    logger.info("Deleting history item", { id })

    try {
        const items = await getHistoryItems()
        const itemToDelete = items.find((item) => item.id === id)

        if (itemToDelete) {
            // Delete files
            try {
                // The directory containing the images
                const itemDir = `${HISTORY_DIR}${id}/`
                await FileSystem.deleteAsync(itemDir, { idempotent: true })
                logger.debug("Deleted history item directory", {
                    id,
                    dir: itemDir,
                })
            } catch (e) {
                logger.warn("Failed to delete history files", { id, error: e })
            }
        }

        const newItems = items.filter((item) => item.id !== id)
        storage.set(STORAGE_KEY, JSON.stringify(newItems))

        logger.info("History item deleted", {
            id,
            remainingCount: newItems.length,
        })
    } catch (error) {
        logger.error(
            "Failed to delete history item",
            error instanceof Error ? error : new Error(String(error)),
            { id },
        )
        throw error
    }
}
