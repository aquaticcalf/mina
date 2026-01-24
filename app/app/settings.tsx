import { MoonStarIcon, SunIcon } from "lucide-react-native"
import { View } from "react-native"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { Switch } from "@/components/ui/switch"
import { Text } from "@/components/ui/text"
import { useTheme } from "@/lib/theme/context"

export default function SettingsScreen() {
    const { colorScheme, toggleColorScheme } = useTheme()
    const themeIcon = colorScheme === "dark" ? MoonStarIcon : SunIcon

    return (
        <View className="flex-1 bg-background px-5 pt-10">
            <Text className="text-2xl font-semibold text-foreground text-center py-5">
                Settings
            </Text>

            <View className="mt-8 rounded-2xl border border-border bg-muted/30 p-4">
                <View className="flex-row items-center justify-between">
                    <View className="flex-row items-center gap-3">
                        <Icon
                            as={themeIcon}
                            size={18}
                            className="text-foreground"
                        />
                        <Text className="text-base font-medium text-foreground">
                            Theme
                        </Text>
                    </View>

                    <Button
                        variant="ghost"
                        className="h-10 px-3"
                        onPress={toggleColorScheme}
                    >
                        <Text className="text-sm text-muted-foreground">
                            {colorScheme === "dark" ? "Dark" : "Light"}
                        </Text>
                    </Button>
                </View>

                <View className="mt-4 h-px bg-border" />

                <View className="mt-4 flex-row items-center justify-between">
                    <Text className="text-base font-medium text-foreground">
                        Save captures to Photos
                    </Text>
                    <Switch
                        checked={false}
                        onCheckedChange={() => {
                            /* TODO: implement save to photos */
                        }}
                    />
                </View>

                <View className="mt-4 h-px bg-border" />

                <View className="mt-4 flex-row items-center justify-between">
                    <Text className="text-base font-medium text-foreground">
                        Photo quality
                    </Text>
                    <Text className="text-sm text-muted-foreground">High</Text>
                </View>
            </View>
        </View>
    )
}
