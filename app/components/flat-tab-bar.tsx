import type { BottomTabBarProps } from "@react-navigation/bottom-tabs"
import {
    CameraIcon,
    CameraOffIcon,
    ClockIcon,
    HistoryIcon,
    Settings2Icon,
    SettingsIcon,
} from "lucide-react-native"
import { Platform, Pressable, Text, View } from "react-native"
import { useSafeAreaInsets } from "react-native-safe-area-context"
import { Icon } from "@/components/ui/icon"
import { cn } from "@/lib/utils"

function TabItem({
    focused,
    label,
    icon,
    onPress,
}: {
    focused: boolean
    label: string
    icon: "camera" | "history" | "settings"
    onPress: () => void
}) {
    const IconCmp =
        icon === "camera"
            ? focused
                ? CameraIcon
                : CameraOffIcon
            : icon === "history"
              ? focused
                  ? HistoryIcon
                  : ClockIcon
              : focused
                ? Settings2Icon
                : SettingsIcon

    return (
        <Pressable
            onPress={onPress}
            accessibilityRole={Platform.select({
                ios: "button",
                default: "tab",
            })}
            accessibilityState={focused ? { selected: true } : {}}
            className={cn("flex-1 items-center justify-center", "min-h-11")}
        >
            <View className="relative items-center justify-center px-2 py-2">
                <Icon
                    as={IconCmp}
                    size={22}
                    className={cn(
                        focused ? "text-foreground" : "text-muted-foreground",
                    )}
                />
                <Text
                    className={cn(
                        "mt-1 text-[11px] font-medium",
                        focused ? "text-foreground" : "text-muted-foreground",
                    )}
                >
                    {label}
                </Text>
                <View
                    className={cn(
                        "absolute bottom-0 h-[2px] w-7 rounded-full",
                        focused ? "bg-foreground" : "bg-transparent",
                    )}
                />
            </View>
        </Pressable>
    )
}

export function FlatTabBar({
    state,
    descriptors,
    navigation,
}: BottomTabBarProps) {
    const insets = useSafeAreaInsets()

    return (
        <View
            className={cn("border-t border-border bg-background", "flex-row")}
            style={{ paddingBottom: Math.max(insets.bottom, 8) }}
        >
            {state.routes
                .filter(
                    (route) =>
                        route.name === "index" ||
                        route.name === "history" ||
                        route.name === "settings",
                )
                .map((route, index) => {
                    const { options } = descriptors[route.key]
                    const label =
                        options.tabBarLabel !== undefined
                            ? String(options.tabBarLabel)
                            : options.title !== undefined
                              ? options.title
                              : route.name

                    const isFocused =
                        state.index ===
                        state.routes.findIndex((r) => r.key === route.key)

                    const onPress = () => {
                        const event = navigation.emit({
                            type: "tabPress",
                            target: route.key,
                            canPreventDefault: true,
                        })

                        if (!isFocused && !event.defaultPrevented) {
                            navigation.navigate(route.name)
                        }
                    }

                    const icon: "camera" | "history" | "settings" =
                        route.name === "index"
                            ? "camera"
                            : route.name === "history"
                              ? "history"
                              : "settings"

                    return (
                        <TabItem
                            key={route.key}
                            focused={isFocused}
                            label={label}
                            icon={icon}
                            onPress={onPress}
                        />
                    )
                })}
        </View>
    )
}
