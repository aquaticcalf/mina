/**
 * Model download and update management.
 *
 * Downloads TFLite models from GitHub releases based on the configured channel
 * (dev or prod). Checks for updates by comparing release dates.
 */

import * as FileSystem from "expo-file-system/legacy"
import type {
    GitHubAsset,
    GitHubRelease,
    ModelChannel,
    ModelMetadata,
    ModelState,
    UpdateCheckResult,
} from "@/lib/model/types"
import {
    clearModelMetadata,
    ensureModelDirectory,
    getModelPath,
    loadModelMetadata,
    modelFileExists,
    saveModelMetadata,
} from "@/lib/model/storage"

/** GitHub repository for model releases */
const GITHUB_REPO = "fishcareyolo/fishcareyolo"

/** Model file name in the release assets */
const MODEL_FILENAME = "best_full_integer_quant.tflite"

/** Get the model channel from environment variable */
export function getModelChannel(): ModelChannel {
    const channel = process.env.EXPO_PUBLIC_MODEL_CHANNEL
    if (channel === "dev" || channel === "prod") {
        return channel
    }
    // Default to prod if not set or invalid
    return "prod"
}

/** Get the GitHub releases API URL for a channel */
function getReleaseApiUrl(channel: ModelChannel): string {
    return `https://api.github.com/repos/${GITHUB_REPO}/releases/tags/${channel}`
}

/** Get the direct download URL for the model */
function getModelDownloadUrl(channel: ModelChannel): string {
    return `https://github.com/${GITHUB_REPO}/releases/download/${channel}/${MODEL_FILENAME}`
}

/** Parse the "Updated" date from release body */
function parseReleaseDateFromBody(body: string): string | null {
    // Release body contains: **Updated:** 2024-01-15 10:30 UTC
    const match = body.match(
        /\*\*Updated:\*\*\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+UTC)/,
    )
    if (match) {
        return match[1]
    }
    return null
}

/** Fetch release info from GitHub API */
async function fetchReleaseInfo(channel: ModelChannel): Promise<GitHubRelease> {
    const url = getReleaseApiUrl(channel)
    const response = await fetch(url, {
        headers: {
            Accept: "application/vnd.github.v3+json",
            // User-Agent required by GitHub API
            "User-Agent": "Mina-App",
        },
    })

    if (!response.ok) {
        throw new Error(
            `Failed to fetch release info: ${response.status} ${response.statusText}`,
        )
    }

    return response.json()
}

/** Check if a model update is available */
export async function checkForUpdate(
    channel: ModelChannel,
): Promise<UpdateCheckResult> {
    const localMetadata = await loadModelMetadata()

    // If no local model or different channel, we need to download
    if (!localMetadata || localMetadata.channel !== channel) {
        const release = await fetchReleaseInfo(channel)
        const newDate = parseReleaseDateFromBody(release.body)
        return {
            hasUpdate: true,
            newDate,
            currentDate: localMetadata?.updatedAt ?? null,
        }
    }

    // Fetch remote release info
    const release = await fetchReleaseInfo(channel)
    const remoteDate = parseReleaseDateFromBody(release.body)

    // Compare dates
    const hasUpdate = remoteDate !== localMetadata.updatedAt

    return {
        hasUpdate,
        newDate: remoteDate,
        currentDate: localMetadata.updatedAt,
    }
}

/** Download progress callback type */
export type DownloadProgressCallback = (progress: number) => void

/** Download the model from GitHub releases */
export async function downloadModel(
    channel: ModelChannel,
    onProgress?: DownloadProgressCallback,
): Promise<ModelMetadata> {
    await ensureModelDirectory()

    const downloadUrl = getModelDownloadUrl(channel)
    const localPath = getModelPath(channel)

    // Fetch release info to get the updated date
    const release = await fetchReleaseInfo(channel)
    const updatedAt =
        parseReleaseDateFromBody(release.body) ?? release.published_at

    // Find the model asset to get size
    const modelAsset = release.assets.find(
        (asset: GitHubAsset) => asset.name === MODEL_FILENAME,
    )
    const expectedSize = modelAsset?.size ?? 0

    // Create download resumable for progress tracking
    const downloadResumable = FileSystem.createDownloadResumable(
        downloadUrl,
        localPath,
        {},
        (downloadProgress) => {
            if (onProgress && downloadProgress.totalBytesExpectedToWrite > 0) {
                const progress =
                    downloadProgress.totalBytesWritten /
                    downloadProgress.totalBytesExpectedToWrite
                onProgress(progress)
            }
        },
    )

    const result = await downloadResumable.downloadAsync()

    if (!result || result.status !== 200) {
        throw new Error(`Download failed with status: ${result?.status}`)
    }

    // Create and save metadata
    const metadata: ModelMetadata = {
        channel,
        updatedAt,
        sizeBytes: expectedSize,
        downloadUrl,
    }

    await saveModelMetadata(metadata)

    return metadata
}

/** Initialize model - download if needed, check for updates */
export async function initializeModel(
    onProgress?: DownloadProgressCallback,
): Promise<ModelState> {
    const channel = getModelChannel()

    try {
        // Check if we have a local model
        const hasLocalModel = await modelFileExists(channel)
        const localMetadata = await loadModelMetadata()

        if (
            !hasLocalModel ||
            !localMetadata ||
            localMetadata.channel !== channel
        ) {
            // No local model or wrong channel - download
            const metadata = await downloadModel(channel, onProgress)
            return {
                isReady: true,
                isLoading: false,
                progress: 1,
                error: null,
                metadata,
            }
        }

        // Check for updates in background
        try {
            const updateCheck = await checkForUpdate(channel)
            if (updateCheck.hasUpdate) {
                // Download update
                const metadata = await downloadModel(channel, onProgress)
                return {
                    isReady: true,
                    isLoading: false,
                    progress: 1,
                    error: null,
                    metadata,
                }
            }
        } catch (updateError) {
            // Update check failed, but we have a local model - continue with it
            console.warn("Update check failed, using local model:", updateError)
        }

        // Return existing model state
        return {
            isReady: true,
            isLoading: false,
            progress: 1,
            error: null,
            metadata: localMetadata,
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        return {
            isReady: false,
            isLoading: false,
            progress: 0,
            error: `Failed to initialize model: ${message}`,
            metadata: null,
        }
    }
}

/** Force re-download the model */
export async function forceUpdateModel(
    onProgress?: DownloadProgressCallback,
): Promise<ModelState> {
    const channel = getModelChannel()

    try {
        await clearModelMetadata()
        const metadata = await downloadModel(channel, onProgress)
        return {
            isReady: true,
            isLoading: false,
            progress: 1,
            error: null,
            metadata,
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        return {
            isReady: false,
            isLoading: false,
            progress: 0,
            error: `Failed to update model: ${message}`,
            metadata: null,
        }
    }
}

/** Get the local model file path (for use with TFLite inference) */
export async function getLocalModelPath(): Promise<string | null> {
    const channel = getModelChannel()
    const exists = await modelFileExists(channel)
    if (!exists) return null
    return getModelPath(channel)
}
