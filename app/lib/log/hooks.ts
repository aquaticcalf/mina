import { useContext, useMemo } from "react"
import { afterlog, type Builder } from "afterlog"
import { LoggerContext } from "./context"
import type { UseLoggerReturn } from "./types"

export function useLogger(): UseLoggerReturn {
    const context = useContext(LoggerContext)
    if (!context) {
        throw new Error("useLogger must be used within a LoggerProvider")
    }
    return context.log
}

export function useRequestLogger(
    requestId: string,
    initialData?: Record<string, unknown>,
): Builder {
    useContext(LoggerContext)

    return useMemo(() => {
        return afterlog.createBuilder({
            request_id: requestId,
            ...initialData,
        })
    }, [requestId, initialData])
}
