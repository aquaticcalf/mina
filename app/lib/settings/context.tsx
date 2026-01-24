/**
 * React context for app settings state management.
 *
 * Provides settings state to the entire app and persists user preferences.
 */

import AsyncStorage from "@react-native-async-storage/async-storage"
import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"

interface SettingsContextType {
    saveCapturesToPhotos: boolean
    toggleSaveCapturesToPhotos: () => Promise<void>
}

const SettingsContext = createContext<SettingsContextType | undefined>(
    undefined,
)

const SETTINGS_STORAGE_KEY = "@app/settings"

export function SettingsProvider({ children }: { children: React.ReactNode }) {
    const [saveCapturesToPhotos, setSaveCapturesToPhotos] = useState(false)
    const [isLoaded, setIsLoaded] = useState(false)

    useEffect(() => {
        loadSettings()
    }, [])

    const loadSettings = async () => {
        try {
            const saved = await AsyncStorage.getItem(SETTINGS_STORAGE_KEY)
            if (saved) {
                const settings = JSON.parse(saved)
                setSaveCapturesToPhotos(settings.saveCapturesToPhotos ?? false)
            }
        } catch (e) {
            console.error("Failed to load settings:", e)
        } finally {
            setIsLoaded(true)
        }
    }

    const handleToggleSaveCapturesToPhotos = async () => {
        const newValue = !saveCapturesToPhotos
        setSaveCapturesToPhotos(newValue)
        try {
            await AsyncStorage.setItem(
                SETTINGS_STORAGE_KEY,
                JSON.stringify({ saveCapturesToPhotos: newValue }),
            )
        } catch (e) {
            console.error("Failed to save settings:", e)
        }
    }

    if (!isLoaded) {
        return null
    }

    return (
        <SettingsContext.Provider
            value={{
                saveCapturesToPhotos,
                toggleSaveCapturesToPhotos: handleToggleSaveCapturesToPhotos,
            }}
        >
            {children}
        </SettingsContext.Provider>
    )
}

export function useSettings() {
    const context = useContext(SettingsContext)
    if (!context) {
        throw new Error("useSettings must be used within SettingsProvider")
    }
    return context
}
