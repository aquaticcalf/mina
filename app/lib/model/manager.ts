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
import { createComponentLogger, logTiming } from "@/lib/log"

const logger = createComponentLogger("model/manager")

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
    logger.info("Fetching release info", { channel, url })

    const response = await fetch(url, {
        headers: {
            Accept: "application/vnd.github.v3+json",
            // User-Agent required by GitHub API
            "User-Agent": "Mina-App",
        },
    })

    if (!response.ok) {
        const error = `Failed to fetch release info: ${response.status} ${response.statusText}`
        logger.error("GitHub API request failed", new Error(error), {
            channel,
            status: response.status,
        })
        throw new Error(error)
    }

    logger.info("Release info fetched successfully", { channel })
    return response.json()
}

/** Check if a model update is available */
export async function checkForUpdate(
    channel: ModelChannel,
): Promise<UpdateCheckResult> {
    logger.info("Checking for model update", { channel })

    const localMetadata = await loadModelMetadata()

    // If no local model or different channel, we need to download
    if (!localMetadata || localMetadata.channel !== channel) {
        logger.info("No local model or channel mismatch", {
            hasLocalModel: !!localMetadata,
            localChannel: localMetadata?.channel,
            requestedChannel: channel,
        })
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

    logger.info("Update check completed", {
        hasUpdate,
        remoteDate,
        localDate: localMetadata.updatedAt,
    })

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
    logger.info("Starting model download", { channel })

    await ensureModelDirectory()

    const downloadUrl = getModelDownloadUrl(channel)
    const localPath = getModelPath(channel)

    logger.info("Download configuration", { downloadUrl, localPath })

    // Fetch release info to get the updated date
    const release = await fetchReleaseInfo(channel)
    const updatedAt =
        parseReleaseDateFromBody(release.body) ?? release.published_at

    // Find the model asset to get size
    const modelAsset = release.assets.find(
        (asset: GitHubAsset) => asset.name === MODEL_FILENAME,
    )
    const expectedSize = modelAsset?.size ?? 0

    logger.info("Model asset info", {
        filename: MODEL_FILENAME,
        expectedSize,
        found: !!modelAsset,
    })

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

    const result = await logTiming(
        "model_download",
        () => downloadResumable.downloadAsync(),
        logger,
    )

    if (!result || result.status !== 200) {
        const error = `Download failed with status: ${result?.status}`
        logger.error("Model download failed", new Error(error), {
            status: result?.status,
            channel,
        })
        throw new Error(error)
    }

    logger.info("Model downloaded successfully", {
        channel,
        uri: result.uri,
        status: result.status,
    })

    // Create and save metadata
    const metadata: ModelMetadata = {
        channel,
        updatedAt,
        sizeBytes: expectedSize,
        downloadUrl,
    }

    await saveModelMetadata(metadata)
    logger.info("Model metadata saved", { channel, updatedAt })

    return metadata
}

/** Initialize model - download if needed, check for updates */
export async function initializeModel(
    onProgress?: DownloadProgressCallback,
): Promise<ModelState> {
    const channel = getModelChannel()
    logger.info("Initializing model", { channel })

    try {
        // Check if we have a local model
        const hasLocalModel = await modelFileExists(channel)
        const localMetadata = await loadModelMetadata()

        logger.info("Local model state", {
            hasLocalModel,
            hasMetadata: !!localMetadata,
            localChannel: localMetadata?.channel,
        })

        if (
            !hasLocalModel ||
            !localMetadata ||
            localMetadata.channel !== channel
        ) {
            // No local model or wrong channel - download
            logger.info(
                "Downloading model - no local model or channel mismatch",
            )
            const metadata = await downloadModel(channel, onProgress)
            logger.info("Model initialized with fresh download", { channel })
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
            logger.info("Checking for updates")
            const updateCheck = await checkForUpdate(channel)
            if (updateCheck.hasUpdate) {
                logger.info("Update available, downloading", {
                    newDate: updateCheck.newDate,
                    currentDate: updateCheck.currentDate,
                })
                // Download update
                const metadata = await downloadModel(channel, onProgress)
                logger.info("Model updated successfully", { channel })
                return {
                    isReady: true,
                    isLoading: false,
                    progress: 1,
                    error: null,
                    metadata,
                }
            }
            logger.info("No update available, using local model")
        } catch (updateError) {
            // Update check failed, but we have a local model - continue with it
            logger.warn("Update check failed, using local model", {
                error:
                    updateError instanceof Error
                        ? updateError.message
                        : String(updateError),
            })
        }

        // Return existing model state
        logger.info("Model initialized from local storage", { channel })
        return {
            isReady: true,
            isLoading: false,
            progress: 1,
            error: null,
            metadata: localMetadata,
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        logger.error(
            "Model initialization failed",
            error instanceof Error ? error : new Error(message),
        )
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
    logger.info("Force updating model", { channel })

    try {
        await clearModelMetadata()
        logger.info("Model metadata cleared")
        const metadata = await downloadModel(channel, onProgress)
        logger.info("Model force updated successfully", { channel })
        return {
            isReady: true,
            isLoading: false,
            progress: 1,
            error: null,
            metadata,
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        logger.error(
            "Model force update failed",
            error instanceof Error ? error : new Error(message),
        )
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
    if (!exists) {
        logger.warn("Local model file not found", { channel })
        return null
    }
    const path = getModelPath(channel)
    logger.debug("Got local model path", { channel, path })
    return path
}
