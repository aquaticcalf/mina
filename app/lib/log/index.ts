/**
 * @/lib/log - Structured logging module built on afterlog
 *
 * @example
 * ```tsx
 * import { LoggerProvider, createHybridAdapter, useLogger } from "@/lib/log"
 *
 * // In app entry
 * export default function App() {
 *   return (
 *     <LoggerProvider adapter={await createHybridAdapter()}>
 *       <YourApp />
 *     </LoggerProvider>
 *   )
 * }
 *
 * // In components
 * function MyComponent() {
 *   const { info, error } = useLogger()
 *   info("Component mounted")
 * }
 * ```
 */

// Context & Provider
export { LoggerProvider, LoggerContext } from "./context"
export type { LoggerProviderProps } from "./context"

// Hooks
export { useLogger, useRequestLogger } from "./hooks"

// Adapters
export {
    createConsoleLogAdapter,
    createFileAdapter,
    createHybridAdapter,
} from "./adapters"

// Simple logger utilities
export {
    createLogger,
    createComponentLogger,
    createActionLogger,
    logTiming,
    logAsync,
    logSync,
} from "./simple"

// Types
export type {
    LoggerConfig,
    FileAdapterConfig,
    UseLoggerReturn,
    LoggerAdapter,
    SamplingConfig,
    SamplingRule,
} from "./types"

// Afterlog exports
export {
    afterlog,
    errorRule,
    createLatencyRule,
    createRandomRule,
    createConsistentRule,
} from "afterlog"
