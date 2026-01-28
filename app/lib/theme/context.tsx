/**
 * React context for theme state management.
 *
 * Provides theme state to the entire app and persists user preference.
 */

import { useColorScheme as useNativeWindColorScheme } from "nativewind"
import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"
import { storage } from "@/lib/storage"
import type { ColorScheme } from "@/lib/theme/types"

interface ThemeContextType {
    colorScheme: ColorScheme
    toggleColorScheme: () => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

const THEME_STORAGE_KEY = "@app/theme"

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const nativeWindColorScheme = useNativeWindColorScheme()
    const [isLoaded, setIsLoaded] = useState(false)

    useEffect(() => {
        const saved = storage.getString(THEME_STORAGE_KEY)
        if (saved === "light" || saved === "dark") {
            nativeWindColorScheme.setColorScheme(saved)
        }
        setIsLoaded(true)
    }, [])

    const handleToggleColorScheme = () => {
        const newScheme =
            nativeWindColorScheme.colorScheme === "light" ? "dark" : "light"
        nativeWindColorScheme.setColorScheme(newScheme)
        storage.set(THEME_STORAGE_KEY, newScheme)
    }

    if (!isLoaded) {
        return null
    }

    return (
        <ThemeContext.Provider
            value={{
                colorScheme: nativeWindColorScheme.colorScheme ?? "light",
                toggleColorScheme: handleToggleColorScheme,
            }}
        >
            {children}
        </ThemeContext.Provider>
    )
}

export function useTheme() {
    const context = useContext(ThemeContext)
    if (!context) {
        throw new Error("useTheme must be used within ThemeProvider")
    }
    return context
}
