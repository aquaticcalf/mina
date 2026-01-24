import "@/global.css"

import { ThemeProvider as NavThemeProvider } from "@react-navigation/native"
import { PortalHost } from "@rn-primitives/portal"
import { Tabs } from "expo-router"
import { FlatTabBar } from "@/components/ui/flat-tab-bar"
import { StatusBar } from "expo-status-bar"
import { CameraProvider } from "@/lib/camera"
import { ModelProvider } from "@/lib/model"
import { NAV_THEME, ThemeProvider, useTheme } from "@/lib/theme"

export { ErrorBoundary } from "expo-router"

function RootLayoutContent() {
    const { colorScheme } = useTheme()

    return (
        <NavThemeProvider value={NAV_THEME[colorScheme]}>
            <StatusBar style={colorScheme === "dark" ? "light" : "dark"} />
            <Tabs
                screenOptions={{ headerShown: false }}
                tabBar={(props) => <FlatTabBar {...props} />}
            >
                <Tabs.Screen name="index" options={{ title: "Home" }} />
                <Tabs.Screen name="history" options={{ title: "History" }} />
                <Tabs.Screen name="settings" options={{ title: "Settings" }} />
                <Tabs.Screen name="+not-found" options={{ href: null }} />
                <Tabs.Screen name="+html" options={{ href: null }} />
            </Tabs>
            <PortalHost />
        </NavThemeProvider>
    )
}

export default function RootLayout() {
    return (
        <ThemeProvider>
            <ModelProvider>
                <CameraProvider>
                    <RootLayoutContent />
                </CameraProvider>
            </ModelProvider>
        </ThemeProvider>
    )
}
