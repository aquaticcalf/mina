import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from "react"
import { inferenceService, gateService } from "./index"
import type { InferenceState, GateState } from "./index"

interface ModelLoaderState {
  modelsReady: boolean
  diseaseState: InferenceState
  gateState: GateState
  retryDownload: () => void
}

const ModelLoaderContext = createContext<ModelLoaderState | null>(null)

export function ModelLoaderProvider({ children }: { children: React.ReactNode }) {
  const [diseaseState, setDiseaseState] = useState<InferenceState>(inferenceService.getStatus())
  const [gateState, setGateState] = useState<GateState>(gateService.getStatus())
  const initRef = useRef(false)

  const startDownloads = useCallback(() => {
    // Only trigger if idle or error
    if (inferenceService.getStatus().status === "idle" || inferenceService.getStatus().status === "error") {
      inferenceService.serve().catch(console.error)
    }
    if (gateService.getStatus().status === "idle" || gateService.getStatus().status === "error") {
      gateService.serve().catch(console.error)
    }
  }, [])

  useEffect(() => {
    if (!initRef.current) {
      initRef.current = true
      startDownloads()
    }

    const unsubDisease = inferenceService.onStatusChange(setDiseaseState)
    const unsubGate = gateService.onStatusChange(setGateState)

    return () => {
      unsubDisease()
      unsubGate()
    }
  }, [startDownloads])

  const modelsReady = diseaseState.status === "ready" && gateState.status === "ready"

  return (
    <ModelLoaderContext.Provider
      value={{
        modelsReady,
        diseaseState,
        gateState,
        retryDownload: startDownloads,
      }}
    >
      {children}
    </ModelLoaderContext.Provider>
  )
}

export function useModelLoader() {
  const ctx = useContext(ModelLoaderContext)
  if (!ctx) throw new Error("useModelLoader must be used within ModelLoaderProvider")
  return ctx
}
