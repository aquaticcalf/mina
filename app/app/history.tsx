import { ClockIcon } from "lucide-react-native"
import { View } from "react-native"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"

export default function HistoryScreen() {
    return (
        <View className="flex-1 bg-background px-5 pt-10">
            <Text className="text-2xl font-semibold text-foreground">
                History
            </Text>

            <View className="mt-8 items-center justify-center rounded-2xl border border-border bg-muted/30 p-8">
                <Icon
                    as={ClockIcon}
                    size={28}
                    className="text-muted-foreground"
                />
                <Text className="mt-3 text-base font-medium text-foreground">
                    No scans yet
                </Text>
                <Text className="mt-1 text-center text-sm text-muted-foreground">
                    Your recent detections will show up here.
                </Text>
            </View>
        </View>
    )
}
