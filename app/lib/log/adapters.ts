import * as FileSystem from "expo-file-system"
import { createConsoleAdapter, type LoggerAdapter } from "afterlog"
import type { FileAdapterConfig } from "./types"

// Default log directory within the app's document storage.
const LOG_DIR = new FileSystem.Directory(FileSystem.Paths.document, "logs")

// Maximum log file size in bytes before rotation should occur.
// TODO: Implement log rotation when this limit is reached.
const _MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 // 10MB

// Reuse TextEncoder instance to avoid repeated allocations.
const textEncoder = new TextEncoder()

/**
 * Ensure the log directory exists, creating it if necessary.
 * Uses intermediates: true to create parent directories.
 */
async function ensureLogDirectory(): Promise<void> {
    if (!LOG_DIR.exists) {
        LOG_DIR.create({ intermediates: true })
    }
}

/**
 * Create a console log adapter with pretty printing enabled.
 * Suitable for development environments.
 */
export function createConsoleLogAdapter() {
    return createConsoleAdapter({ pretty: true })
}

/**
 * Create a file-based log adapter.
 * Writes JSON lines to a file for structured log analysis.
 */
export async function createFileAdapter(
    config: FileAdapterConfig = {},
): Promise<LoggerAdapter> {
    await ensureLogDirectory()

    const directory = config.directory
        ? new FileSystem.Directory(config.directory)
        : LOG_DIR
    const filename = config.filename ?? "app.jsonl"
    const filepath = new FileSystem.File(directory, filename)
    // maxFiles is reserved for future log rotation implementation.
    void config.maxFiles

    let fileHandle: FileSystem.FileHandle | null = null
    let currentFile: FileSystem.File | null = null

    /**
     * Get or create the file handle for writing.
     * Lazily opens the file on first write.
     */
    async function getFileHandle(): Promise<{
        file: FileSystem.File
        handle: FileSystem.FileHandle
    }> {
        if (!currentFile || !fileHandle) {
            currentFile = filepath
            if (!currentFile.exists) {
                currentFile.create({ overwrite: true })
            }
            fileHandle = currentFile.open()
        }
        return { file: currentFile, handle: fileHandle }
    }

    const fileAdapter: LoggerAdapter = {
        async emit(event: Record<string, unknown>) {
            try {
                const line = JSON.stringify(event) + "\n"
                const { handle } = await getFileHandle()
                handle.writeBytes(textEncoder.encode(line))
            } catch {
                // Fail silently to avoid log errors causing app crashes.
                console.error("[Logger] Failed to write log to file.")
            }
        },

        async flush() {
            if (fileHandle) {
                fileHandle.close()
                fileHandle = null
                currentFile = null
            }
        },

        async destroy() {
            if (fileHandle) {
                fileHandle.close()
                fileHandle = null
                currentFile = null
            }
        },

        isHealthy() {
            return true
        },
    }

    return fileAdapter
}

/**
 * Create a hybrid adapter that writes to both console and file.
 * Useful for development (console) with production logging (file).
 */
export async function createHybridAdapter(config?: {
    console?: boolean
    file?: FileAdapterConfig
}): Promise<LoggerAdapter> {
    const adapters: LoggerAdapter[] = []

    // Console adapter enabled by default.
    if (config?.console ?? true) {
        adapters.push(createConsoleLogAdapter())
    }

    // File adapter enabled by default.
    if (config?.file ?? true) {
        const fileAdapter = await createFileAdapter(config?.file)
        adapters.push(fileAdapter)
    }

    return {
        async emit(event: Record<string, unknown>) {
            // Write to all adapters in parallel for performance.
            await Promise.all(adapters.map((adapter) => adapter.emit(event)))
        },
        async flush() {
            await Promise.all(adapters.map((adapter) => adapter.flush?.()))
        },
        async destroy() {
            await Promise.all(adapters.map((adapter) => adapter.destroy?.()))
        },
        isHealthy() {
            // Healthy only if all adapters are healthy.
            return adapters.every((adapter) => adapter.isHealthy?.() ?? true)
        },
    }
}
