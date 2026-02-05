import type { LoggerAdapter, SamplingConfig, SamplingRule } from "afterlog"

export type { LoggerAdapter, SamplingConfig, SamplingRule }

export interface LoggerConfig {
    adapter: LoggerAdapter
    sampling?: SamplingConfig
    service?: string
    version?: string
}

export interface FileAdapterConfig {
    directory?: string
    filename?: string
    maxFiles?: number
}

export interface UseLoggerReturn {
    debug: (message: string, data?: Record<string, unknown>) => void
    info: (message: string, data?: Record<string, unknown>) => void
    warn: (message: string, data?: Record<string, unknown>) => void
    error: (
        message: string,
        error?: Error,
        data?: Record<string, unknown>,
    ) => void
}

export type LoggerHookReturn = {
    debug: (message: string, data?: Record<string, unknown>) => void
    info: (message: string, data?: Record<string, unknown>) => void
    warn: (message: string, data?: Record<string, unknown>) => void
    error: (
        message: string,
        error?: Error,
        data?: Record<string, unknown>,
    ) => void
}
