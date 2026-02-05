import {
    MoonStarIcon,
    SunIcon,
    CameraIcon,
    ImageIcon,
    InfoIcon,
} from "lucide-react-native"
import { View, Pressable } from "react-native"
import { Icon } from "@/components/ui/icon"
import { Switch } from "@/components/ui/switch"
import { Text } from "@/components/ui/text"
import { useTheme } from "@/lib/theme/context"
import { useLogger } from "@/lib/log"

export default function SettingsScreen() {
    const { colorScheme, toggleColorScheme } = useTheme()
    const { info, debug } = useLogger()

    debug("SettingsScreen rendered", { colorScheme })

    const handleThemeToggle = () => {
        const newScheme = colorScheme === "dark" ? "light" : "dark"
        info("Theme changed", { from: colorScheme, to: newScheme })
        toggleColorScheme()
    }

    return (
        <View className="flex-1 bg-background">
            <View className="pt-14 pb-4 px-5 bg-card border-b border-border">
                <Text className="text-2xl font-bold text-foreground">
                    Settings
                </Text>
                <Text className="text-muted-foreground mt-1 text-base">
                    App preferences
                </Text>
            </View>

            <View className="p-5">
                {/* Theme Setting */}
                <View className="mb-6">
                    <Text className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                        Appearance
                    </Text>
                    <Pressable
                        className="flex-row items-center justify-between bg-card rounded-xl p-4 border border-border active:border-primary"
                        onPress={handleThemeToggle}
                    >
                        <View className="flex-row items-center gap-3">
                            <View className="w-12 h-12 rounded-xl bg-primary/10 items-center justify-center">
                                <Icon
                                    as={
                                        colorScheme === "dark"
                                            ? MoonStarIcon
                                            : SunIcon
                                    }
                                    size={24}
                                    className="text-primary"
                                />
                            </View>
                            <View>
                                <Text className="font-bold text-base text-foreground">
                                    {colorScheme === "dark"
                                        ? "Dark Mode"
                                        : "Light Mode"}
                                </Text>
                                <Text className="text-sm text-muted-foreground">
                                    Tap to change
                                </Text>
                            </View>
                        </View>
                        <Icon
                            as={colorScheme === "dark" ? MoonStarIcon : SunIcon}
                            size={20}
                            className="text-primary"
                        />
                    </Pressable>
                </View>

                {/* Photo Settings */}
                <View className="mb-6">
                    <Text className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                        Photos
                    </Text>
                    <View className="bg-card rounded-xl border border-border overflow-hidden">
                        <View className="flex-row items-center justify-between p-4 border-b border-border">
                            <View className="flex-row items-center gap-3">
                                <View className="w-12 h-12 rounded-xl bg-primary/10 items-center justify-center">
                                    <Icon
                                        as={CameraIcon}
                                        size={24}
                                        className="text-primary"
                                    />
                                </View>
                                <View>
                                    <Text className="font-bold text-base text-foreground">
                                        Save Photos
                                    </Text>
                                    <Text className="text-sm text-muted-foreground">
                                        Keep pictures in gallery
                                    </Text>
                                </View>
                            </View>
                            <Switch
                                checked={false}
                                onCheckedChange={() => {}}
                            />
                        </View>

                        <View className="flex-row items-center justify-between p-4">
                            <View className="flex-row items-center gap-3">
                                <View className="w-12 h-12 rounded-xl bg-primary/10 items-center justify-center">
                                    <Icon
                                        as={ImageIcon}
                                        size={24}
                                        className="text-primary"
                                    />
                                </View>
                                <View>
                                    <Text className="font-bold text-base text-foreground">
                                        Photo Quality
                                    </Text>
                                    <Text className="text-sm text-muted-foreground">
                                        High quality for clear results
                                    </Text>
                                </View>
                            </View>
                            <Text className="text-base font-bold text-primary">
                                High
                            </Text>
                        </View>
                    </View>
                </View>

                {/* About */}
                <View className="mb-6">
                    <Text className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                        About
                    </Text>
                    <View className="bg-card rounded-xl p-4 border border-border">
                        <View className="flex-row items-center gap-3 mb-4">
                            <View className="w-12 h-12 rounded-xl bg-primary/10 items-center justify-center">
                                <Icon
                                    as={InfoIcon}
                                    size={24}
                                    className="text-primary"
                                />
                            </View>
                            <View>
                                <Text className="font-bold text-lg text-foreground">
                                    Mina
                                </Text>
                                <Text className="text-sm text-muted-foreground">
                                    Fish Health Checker
                                </Text>
                            </View>
                        </View>
                        <Text className="text-sm text-muted-foreground leading-relaxed">
                            Check your fish for diseases using your phone
                            camera. All checks are done on your device - no
                            internet needed.
                        </Text>
                        <Text className="text-xs text-muted-foreground/60 mt-4 text-center">
                            Version 1.0.0
                        </Text>
                    </View>
                </View>
            </View>
        </View>
    )
}
