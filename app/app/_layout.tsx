import "@/global.css"

import { ThemeProvider as NavThemeProvider } from "@react-navigation/native"
import { PortalHost } from "@rn-primitives/portal"
import { Stack } from "expo-router"
import { StatusBar } from "expo-status-bar"
import { CameraProvider } from "@/lib/camera"
import { ModelProvider } from "@/lib/model"
import { NavigationProvider } from "@/lib/navigation"
import { NAV_THEME, ThemeProvider, useTheme } from "@/lib/theme"

export { ErrorBoundary } from "expo-router"

function RootLayoutContent() {
    const { colorScheme } = useTheme()

    return (
        <NavThemeProvider value={NAV_THEME[colorScheme]}>
            <StatusBar style={colorScheme === "dark" ? "light" : "dark"} />
            <Stack screenOptions={{ headerShown: false }}>
                <Stack.Screen name="index" />
                <Stack.Screen
                    name="history"
                    options={{
                        presentation: "transparentModal",
                    }}
                />
                <Stack.Screen
                    name="settings"
                    options={{
                        presentation: "transparentModal",
                    }}
                />
                <Stack.Screen name="crop" />
                <Stack.Screen name="preview" />
                <Stack.Screen name="+not-found" />
            </Stack>
            <PortalHost />
        </NavThemeProvider>
    )
}

export default function RootLayout() {
    return (
        <ThemeProvider>
            <NavigationProvider>
                <ModelProvider>
                    <CameraProvider>
                        <RootLayoutContent />
                    </CameraProvider>
                </ModelProvider>
            </NavigationProvider>
        </ThemeProvider>
    )
}
