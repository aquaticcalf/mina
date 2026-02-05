import "@/global.css"

import { ThemeProvider as NavThemeProvider } from "@react-navigation/native"
import { PortalHost } from "@rn-primitives/portal"
import { Tabs } from "expo-router"
import { StatusBar } from "expo-status-bar"
import React, { useEffect, useState } from "react"
import { FlatTabBar } from "@/components/ui/flat-tab-bar"
import { CameraProvider } from "@/lib/camera"
import { ModelProvider } from "@/lib/model"
import { useDefaultTab } from "@/lib/navigation/default"
import { NavigationProvider } from "@/lib/navigation"
import { NAV_THEME, ThemeProvider, useTheme } from "@/lib/theme"
import {
    LoggerProvider,
    createHybridAdapter,
    type LoggerAdapter,
} from "@/lib/log"

export { ErrorBoundary } from "expo-router"

function RootLayoutContent() {
    const { colorScheme } = useTheme()
    useDefaultTab()

    return (
        <NavThemeProvider value={NAV_THEME[colorScheme]}>
            <StatusBar style={colorScheme === "dark" ? "light" : "dark"} />
            <Tabs
                screenOptions={{ headerShown: false }}
                tabBar={(props) => <FlatTabBar {...props} />}
            >
                <Tabs.Screen name="history" options={{ title: "History" }} />
                <Tabs.Screen name="index" options={{ title: "Home" }} />
                <Tabs.Screen name="settings" options={{ title: "Settings" }} />
                <Tabs.Screen name="crop" options={{ href: null }} />
                <Tabs.Screen name="preview" options={{ href: null }} />
                <Tabs.Screen name="+not-found" options={{ href: null }} />
            </Tabs>
            <PortalHost />
        </NavThemeProvider>
    )
}

export default function RootLayout() {
    const [adapter, setAdapter] = useState<LoggerAdapter | null>(null)
    const [isLoggerReady, setIsLoggerReady] = useState(false)

    useEffect(() => {
        async function initLogger() {
            try {
                const hybridAdapter = await createHybridAdapter({
                    console: true,
                    file: { filename: "app.jsonl" },
                })
                setAdapter(hybridAdapter)
                setIsLoggerReady(true)
            } catch (error) {
                console.error("Failed to initialize logger:", error)
                // Continue without file logging if it fails
                setIsLoggerReady(true)
            }
        }

        initLogger()
    }, [])

    if (!isLoggerReady) {
        return null // Or a splash screen
    }

    return (
        <LoggerProvider adapter={adapter ?? undefined}>
            <ThemeProvider>
                <NavigationProvider>
                    <ModelProvider>
                        <CameraProvider>
                            <RootLayoutContent />
                        </CameraProvider>
                    </ModelProvider>
                </NavigationProvider>
            </ThemeProvider>
        </LoggerProvider>
    )
}
