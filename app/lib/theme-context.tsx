import AsyncStorage from "@react-native-async-storage/async-storage"
import { useColorScheme as useNativeWindColorScheme } from "nativewind"
import React, { createContext, useContext, useEffect, useState } from "react"

type ColorScheme = "light" | "dark"

interface ThemeContextType {
    colorScheme: ColorScheme
    toggleColorScheme: () => Promise<void>
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

const THEME_STORAGE_KEY = "@app/theme"

export function ThemeProvider({ children }: { children: React.ReactNode }) {
    const nativeWindColorScheme = useNativeWindColorScheme()
    const [isLoaded, setIsLoaded] = useState(false)

    const loadTheme = React.useCallback(async () => {
        try {
            const saved = await AsyncStorage.getItem(THEME_STORAGE_KEY)
            if (saved === "light" || saved === "dark") {
                nativeWindColorScheme.setColorScheme(saved)
            }
        } catch (e) {
            console.error("Failed to load theme:", e)
        } finally {
            setIsLoaded(true)
        }
    }, [nativeWindColorScheme])

    useEffect(() => {
        loadTheme()
    }, [loadTheme])

    const handleToggleColorScheme = async () => {
        const newScheme =
            nativeWindColorScheme.colorScheme === "light" ? "dark" : "light"
        nativeWindColorScheme.setColorScheme(newScheme)
        try {
            await AsyncStorage.setItem(THEME_STORAGE_KEY, newScheme)
        } catch (e) {
            console.error("Failed to save theme:", e)
        }
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
