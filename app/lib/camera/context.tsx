/**
 * React context for camera state management.
 *
 * Provides camera settings state to the app and persists user preferences.
 */

import AsyncStorage from "@react-native-async-storage/async-storage"
import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"
import type { CameraFacing, FlashMode } from "@/lib/camera/types"

interface CameraContextType {
    cameraFacing: CameraFacing
    flashMode: FlashMode
    torchEnabled: boolean
    setCameraFacing: (facing: CameraFacing) => void
    setFlashMode: (mode: FlashMode) => void
    setTorchEnabled: (enabled: boolean) => void
}

const CameraContext = createContext<CameraContextType | undefined>(undefined)

const CAMERA_STORAGE_KEY = "@app/camera"

export function CameraProvider({ children }: { children: React.ReactNode }) {
    const [cameraFacing, setCameraFacing] = useState<CameraFacing>("back")
    const [flashMode, setFlashMode] = useState<FlashMode>("off")
    const [torchEnabled, setTorchEnabled] = useState(false)

    useEffect(() => {
        const loadCameraSettings = async () => {
            try {
                const saved = await AsyncStorage.getItem(CAMERA_STORAGE_KEY)
                if (saved) {
                    const parsed = JSON.parse(saved)
                    if (parsed.cameraFacing)
                        setCameraFacing(parsed.cameraFacing)
                    if (parsed.flashMode) setFlashMode(parsed.flashMode)
                    if (typeof parsed.torchEnabled === "boolean") {
                        setTorchEnabled(parsed.torchEnabled)
                    }
                }
            } catch (e) {
                console.error("Failed to load camera settings:", e)
            }
        }
        loadCameraSettings()
    }, [])

    const handleSetCameraFacing = (facing: CameraFacing) => {
        setCameraFacing(facing)
        persistSettings({ cameraFacing: facing })
    }

    const handleSetFlashMode = (mode: FlashMode) => {
        setFlashMode(mode)
        persistSettings({ flashMode: mode })
    }

    const handleSetTorchEnabled = (enabled: boolean) => {
        setTorchEnabled(enabled)
        persistSettings({ torchEnabled: enabled })
    }

    const persistSettings = async (updates: Partial<CameraContextType>) => {
        try {
            const current = {
                cameraFacing,
                flashMode,
                torchEnabled,
                ...updates,
            }
            await AsyncStorage.setItem(
                CAMERA_STORAGE_KEY,
                JSON.stringify(current),
            )
        } catch (e) {
            console.error("Failed to save camera settings:", e)
        }
    }

    return (
        <CameraContext.Provider
            value={{
                cameraFacing,
                flashMode,
                torchEnabled,
                setCameraFacing: handleSetCameraFacing,
                setFlashMode: handleSetFlashMode,
                setTorchEnabled: handleSetTorchEnabled,
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
