import { afterlog, type Builder } from "afterlog"

interface CreateLoggerOptions {
    component?: string
    subsystem?: string
    requestId?: string
}

// Check if afterlog is configured by checking if it has an adapter
function isAfterlogConfigured(): boolean {
    try {
        // Try to create a builder - this will throw if not configured
        afterlog.createBuilder({})
        return true
    } catch {
        return false
    }
}

// Fallback logger that logs to console when afterlog is not configured
function createFallbackLogger(component?: string): Builder {
    const prefix = component ? `[${component}]` : "[log]"

    return {
        data: () => createFallbackLogger(component),
        debug: (message: string, data?: Record<string, unknown>) => {
            console.debug(prefix, message, data)
        },
        error: (error: Error, data?: Record<string, unknown>) => {
            console.error(prefix, error, data)
        },
        info: (message: string, data?: Record<string, unknown>) => {
            console.info(prefix, message, data)
        },
        timing: async <T>(
            label: string,
            fn: () => T | Promise<T>,
        ): Promise<T> => {
            const start = Date.now()
            const result = await fn()
            console.info(prefix, `${label} took ${Date.now() - start}ms`)
            return result
        },
        warn: (message: string, data?: Record<string, unknown>) => {
            console.warn(prefix, message, data)
        },
    } as unknown as Builder
}

function createLoggerBase(init?: Record<string, unknown>): Builder {
    if (!isAfterlogConfigured()) {
        return createFallbackLogger(init?.component as string)
    }
    return afterlog.createBuilder(init)
}

export function createLogger(options: CreateLoggerOptions = {}): Builder {
    return createLoggerBase({
        component: options.component,
        subsystem: options.subsystem,
        request_id: options.requestId,
    })
}

export function createComponentLogger(componentName: string): Builder {
    return createLoggerBase({
        component: componentName,
    })
}

export function createActionLogger(
    action: string,
    context?: Record<string, unknown>,
): Builder {
    return createLoggerBase({
        action,
        ...context,
    })
}

export function logTiming<T>(
    label: string,
    fn: () => T | Promise<T>,
    logger?: Builder,
): Promise<T> | T {
    if (logger) {
        return logger.timing(label, fn)
    }
    return fn()
}

export async function logAsync<T>(
    label: string,
    fn: () => Promise<T>,
    logger?: Builder,
): Promise<T> {
    if (logger) {
        return logger.timing(label, fn)
    }
    return fn()
}

export function logSync<T>(label: string, fn: () => T, logger?: Builder): T {
    if (logger) {
        return logger.timing(label, fn)
    }
    return fn()
}
