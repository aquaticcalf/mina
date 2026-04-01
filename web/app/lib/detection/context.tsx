import React, { createContext, useContext, useState } from "react"
import type { InferenceResult } from "@/lib/model/types"

interface DetectionState {
  currentResult: InferenceResult | null
  setCurrentResult: (result: InferenceResult | null) => void
}

const DetectionContext = createContext<DetectionState | null>(null)

export function DetectionProvider({ children }: { children: React.ReactNode }) {
  const [currentResult, setCurrentResult] = useState<InferenceResult | null>(null)

  return (
    <DetectionContext.Provider
      value={{
        currentResult,
        setCurrentResult,
      }}
    >
      {children}
    </DetectionContext.Provider>
  )
}

export function useDetectionContext() {
  const ctx = useContext(DetectionContext)
  if (!ctx) throw new Error("useDetectionContext must be inside DetectionProvider")
  return ctx
}
