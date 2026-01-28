/**
 * React context for camera state management.
 *
 * Provides camera settings state to the app and persists user preferences.
 */

import { storage } from "@/lib/storage"
import type React from "react"
import { createContext, useContext, useState } from "react"
import type { CameraFacing, FlashMode } from "@/lib/camera/types"

interface CameraContextType {
    cameraFacing: CameraFacing
    flashMode: FlashMode
    setCameraFacing: (facing: CameraFacing) => void
    setFlashMode: (mode: FlashMode) => void
}

const CameraContext = createContext<CameraContextType | undefined>(undefined)

const CAMERA_STORAGE_KEY = "@app/camera"

export function CameraProvider({ children }: { children: React.ReactNode }) {
    // Initialize state synchronously from MMKV
    const [cameraFacing, setCameraFacing] = useState<CameraFacing>(() => {
        try {
            const saved = storage.getString(CAMERA_STORAGE_KEY)
            if (saved) {
                const parsed = JSON.parse(saved)
                return parsed.cameraFacing ?? "back"
            }
        } catch {}
        return "back"
    })

    const [flashMode, setFlashMode] = useState<FlashMode>(() => {
        try {
            const saved = storage.getString(CAMERA_STORAGE_KEY)
            if (saved) {
                const parsed = JSON.parse(saved)
                return parsed.flashMode ?? "off"
            }
        } catch {}
        return "off"
    })

    const handleSetCameraFacing = (facing: CameraFacing) => {
        setCameraFacing(facing)
        persistSettings({ cameraFacing: facing })
    }

    const handleSetFlashMode = (mode: FlashMode) => {
        setFlashMode(mode)
        persistSettings({ flashMode: mode })
    }

    const persistSettings = (updates: Partial<CameraContextType>) => {
        const current = {
            cameraFacing,
            flashMode,
            ...updates,
        }
        storage.set(CAMERA_STORAGE_KEY, JSON.stringify(current))
    }

    return (
        <CameraContext.Provider
            value={{
                cameraFacing,
                flashMode,
                setCameraFacing: handleSetCameraFacing,
                setFlashMode: handleSetFlashMode,
            }}
        >
            {children}
        </CameraContext.Provider>
    )
}

export function useCamera() {
    const context = useContext(CameraContext)
    if (!context) {
        throw new Error("useCamera must be used within CameraProvider")
    }
    return context
}
