/**
 * React context for camera state management.
 *
 * Provides camera settings state to the app and persists user preferences.
 */

import { storage } from "@/lib/storage"
import type React from "react"
import { createContext, useContext, useState } from "react"
import type { CameraFacing, FlashMode } from "@/lib/camera/types"
import { createComponentLogger } from "@/lib/log"

const logger = createComponentLogger("camera/context")

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
                logger.debug("Loaded camera facing from storage", {
                    facing: parsed.cameraFacing,
                })
                return parsed.cameraFacing ?? "back"
            }
        } catch (error) {
            logger.warn("Failed to load camera facing from storage", { error })
        }
        return "back"
    })

    const [flashMode, setFlashMode] = useState<FlashMode>(() => {
        try {
            const saved = storage.getString(CAMERA_STORAGE_KEY)
            if (saved) {
                const parsed = JSON.parse(saved)
                logger.debug("Loaded flash mode from storage", {
                    mode: parsed.flashMode,
                })
                return parsed.flashMode ?? "off"
            }
        } catch (error) {
            logger.warn("Failed to load flash mode from storage", { error })
        }
        return "off"
    })

    const handleSetCameraFacing = (facing: CameraFacing) => {
        logger.info("Setting camera facing", { facing })
        setCameraFacing(facing)
        persistSettings({ cameraFacing: facing })
    }

    const handleSetFlashMode = (mode: FlashMode) => {
        logger.info("Setting flash mode", { mode })
        setFlashMode(mode)
        persistSettings({ flashMode: mode })
    }

    const persistSettings = (updates: Partial<CameraContextType>) => {
        const current = {
            cameraFacing,
            flashMode,
            ...updates,
        }
        try {
            storage.set(CAMERA_STORAGE_KEY, JSON.stringify(current))
            logger.debug("Camera settings persisted")
        } catch (error) {
            logger.error(
                "Failed to persist camera settings",
                error instanceof Error ? error : new Error(String(error)),
            )
        }
    }

    logger.debug("CameraProvider rendered", { cameraFacing, flashMode })

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
        logger.error(
            "useCamera called outside CameraProvider",
            new Error("useCamera must be used within CameraProvider"),
        )
        throw new Error("useCamera must be used within CameraProvider")
    }
    return context
}
