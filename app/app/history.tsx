import { useFocusEffect, useRouter } from "expo-router"
import React, { useCallback, useState } from "react"
import { FlatList, Pressable, View, ActivityIndicator } from "react-native"
import { Image } from "expo-image"
import { ClockIcon, ChevronRightIcon } from "lucide-react-native"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { getHistoryItems } from "@/lib/history/storage"
import type { HistoryItem } from "@/lib/history/types"
import { getBoundingBoxColor, getDiseaseLabel } from "@/lib/model/disease/info"

export default function HistoryScreen() {
    const router = useRouter()
    const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
    const [isLoading, setIsLoading] = useState(true)

    useFocusEffect(
        useCallback(() => {
            loadHistory()
        }, []),
    )

    const loadHistory = async () => {
        try {
            setIsLoading(true)
            const items = await getHistoryItems()
            // Sort by newest first
            const sorted = items.sort((a, b) => b.timestamp - a.timestamp)
            setHistoryItems(sorted)
        } catch (e) {
            console.error("Failed to load history:", e)
        } finally {
            setIsLoading(false)
        }
    }

    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        })
    }

    const getPrimaryDetection = (item: HistoryItem) => {
        if (item.results.detections.length === 0) {
            return { label: "Healthy", color: getBoundingBoxColor("healthy") }
        }
        const topDetection = item.results.detections[0]
        const label = getDiseaseLabel(topDetection.diseaseClass)
        const color = getBoundingBoxColor(topDetection.diseaseClass)

        return { label, color, confidence: topDetection.confidence }
    }

    const renderItem = ({ item }: { item: HistoryItem }) => {
        const primary = getPrimaryDetection(item)

        return (
            <Pressable
                className="flex-row items-center bg-card border border-border rounded-xl p-3 mb-3"
                onPress={() =>
                    router.push({
                        pathname: "/results",
                        params: { historyId: item.id },
                    })
                }
            >
                <Image
                    source={{ uri: item.processedImageUri }}
                    className="w-16 h-16 rounded-lg bg-muted"
                    contentFit="cover"
                    transition={200}
                />
                <View className="flex-1 ml-3 justify-center">
                    <View className="flex-row justify-between items-center mb-1">
                        <Text
                            className="font-semibold text-base"
                            style={{ color: primary.color }}
                        >
                            {primary.label}
                        </Text>
                        {primary.confidence && (
                            <Text className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                                {(primary.confidence * 100).toFixed(0)}%
                            </Text>
                        )}
                    </View>
                    <Text className="text-sm text-muted-foreground">
                        {formatDate(item.timestamp)}
                    </Text>
                    <Text className="text-xs text-muted-foreground mt-0.5">
                        {item.results.detections.length}{" "}
                        {item.results.detections.length === 1
                            ? "issue"
                            : "issues"}{" "}
                        found
                    </Text>
                </View>
                <Icon
                    as={ChevronRightIcon}
                    size={20}
                    className="text-muted-foreground ml-2"
                />
            </Pressable>
        )
    }

    return (
        <View className="flex-1 bg-background px-5 pt-10">
            <Text className="text-2xl font-semibold text-foreground text-center py-5">
                History
            </Text>

            {isLoading ? (
                <View className="flex-1 items-center justify-center">
                    <ActivityIndicator size="large" />
                </View>
            ) : historyItems.length === 0 ? (
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
            ) : (
                <FlatList
                    data={historyItems}
                    renderItem={renderItem}
                    keyExtractor={(item) => item.id}
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={{ paddingBottom: 20 }}
                />
            )}
        </View>
    )
}
