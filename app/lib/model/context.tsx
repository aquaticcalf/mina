/**
 * React context for model state management.
 *
 * Provides model state to the entire app and auto-initializes on mount.
 */

import {
    createContext,
    useCallback,
    useContext,
    useEffect,
    useState,
    type ReactNode,
} from "react"
import {
    forceUpdateModel,
    getLocalModelPath,
    initializeModel,
} from "@/lib/model/manager"
import {
    runInference as runModelInference,
    type InferenceResult,
} from "@/lib/model/inference"
import type { ModelState } from "@/lib/model/types"
import { createComponentLogger } from "@/lib/log"

const logger = createComponentLogger("model/context")

interface ModelContextValue extends ModelState {
    /** Re-download the model from scratch */
    forceUpdate: () => Promise<void>
    /** Get the local file path to the model (for TFLite) */
    getModelPath: () => Promise<string | null>
    /** Run inference on an image */
    runInference: (imageUri: string) => Promise<InferenceResult | null>
}

const initialState: ModelState = {
    isReady: false,
    isLoading: true,
    progress: 0,
    error: null,
    metadata: null,
}

const ModelContext = createContext<ModelContextValue | null>(null)

export function ModelProvider({ children }: { children: ReactNode }) {
    const [state, setState] = useState<ModelState>(initialState)

    // Initialize model on mount
    useEffect(() => {
        let cancelled = false

        logger.info("ModelProvider mounted, starting initialization")

        async function init() {
            setState((prev) => ({ ...prev, isLoading: true, progress: 0 }))

            const result = await initializeModel((progress) => {
                if (!cancelled) {
                    setState((prev) => ({ ...prev, progress }))
                }
            })

            if (!cancelled) {
                setState(result)
                logger.info("Model initialization completed", {
                    isReady: result.isReady,
                    hasError: !!result.error,
                    metadata: result.metadata,
                })
            }
        }

        init()

        return () => {
            cancelled = true
            logger.debug("ModelProvider unmounted")
        }
    }, [])

    const forceUpdate = useCallback(async () => {
        logger.info("Force update requested")
        setState((prev) => ({
            ...prev,
            isLoading: true,
            progress: 0,
            error: null,
        }))

        const result = await forceUpdateModel((progress) => {
            setState((prev) => ({ ...prev, progress }))
        })

        setState(result)
        logger.info("Force update completed", {
            isReady: result.isReady,
            hasError: !!result.error,
        })
    }, [])

    const getModelPath = useCallback(async () => {
        const path = await getLocalModelPath()
        logger.debug("getModelPath called", { hasPath: !!path })
        return path
    }, [])

    const runInference = useCallback(
        async (imageUri: string): Promise<InferenceResult | null> => {
            logger.info("runInference called", {
                imageUri,
                isReady: state.isReady,
            })

            try {
                // Check if model is ready
                if (!state.isReady || state.error) {
                    logger.warn("Model not ready for inference", {
                        isReady: state.isReady,
                        error: state.error,
                    })
                    return null
                }

                // Get model path
                const modelPath = await getLocalModelPath()
                if (!modelPath) {
                    logger.error(
                        "Model file not found",
                        new Error("Model path is null"),
                    )
                    return null
                }

                logger.info("Running inference", { imageUri, modelPath })

                // Run inference
                const result = await runModelInference(imageUri, modelPath)
                logger.info("Inference completed successfully", {
                    imageUri,
                    detectionsCount: result.detections.length,
                    inferenceTimeMs: result.inferenceTimeMs,
                })
                return result
            } catch (error) {
                const message =
                    error instanceof Error ? error.message : String(error)
                logger.error(
                    "Inference error",
                    error instanceof Error ? error : new Error(message),
                    { imageUri },
                )
                return null
            }
        },
        [state.isReady, state.error],
    )

    return (
        <ModelContext.Provider
            value={{ ...state, forceUpdate, getModelPath, runInference }}
        >
            {children}
        </ModelContext.Provider>
    )
}

/** Hook to access model state and actions */
export function useModel(): ModelContextValue {
    const context = useContext(ModelContext)
    if (!context) {
        logger.error(
            "useModel called outside ModelProvider",
            new Error("useModel must be used within a ModelProvider"),
        )
        throw new Error("useModel must be used within a ModelProvider")
    }
    return context
}
