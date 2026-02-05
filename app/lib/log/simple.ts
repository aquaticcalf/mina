import { afterlog, type Builder } from "afterlog"

interface CreateLoggerOptions {
    component?: string
    subsystem?: string
    requestId?: string
}

function createLoggerBase(init?: Record<string, unknown>): Builder {
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
