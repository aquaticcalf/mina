import { useFocusEffect, useRouter } from "expo-router"
import React, { useCallback, useState } from "react"
import { FlatList, Pressable, View } from "react-native"
import { Image } from "expo-image"
import { ClockIcon, ChevronRightIcon } from "lucide-react-native"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { getHistoryItems } from "@/lib/history/storage"
import type { HistoryItem } from "@/lib/history/types"
import { getBoundingBoxColor, getDiseaseLabel } from "@/lib/model/disease/info"
import { useLogger } from "@/lib/log"

export default function HistoryScreen() {
    const router = useRouter()
    const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])
    const [isLoading, setIsLoading] = useState(true)
    const { info, error: logError, debug } = useLogger()

    useFocusEffect(
        useCallback(() => {
            info("HistoryScreen focused, loading history")
            loadHistory()
        }, []),
    )

    const loadHistory = async () => {
        try {
            setIsLoading(true)
            const items = await getHistoryItems()
            const sorted = items.sort((a, b) => b.timestamp - a.timestamp)
            setHistoryItems(sorted)
            info("History loaded", { count: sorted.length })
        } catch (e) {
            const message = e instanceof Error ? e.message : String(e)
            logError(
                "Failed to load history",
                e instanceof Error ? e : new Error(message),
            )
        } finally {
            setIsLoading(false)
        }
    }

    const handleHistoryItemPress = (item: HistoryItem) => {
        info("History item pressed", {
            id: item.id,
            timestamp: item.timestamp,
            detectionsCount: item.results.detections.length,
        })
        router.push({
            pathname: "/results",
            params: { historyId: item.id },
        })
    }

    const formatDate = (timestamp: number) => {
        const date = new Date(timestamp)
        const today = new Date()
        const yesterday = new Date(today)
        yesterday.setDate(yesterday.getDate() - 1)

        if (date.toDateString() === today.toDateString()) {
            return `Today at ${date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`
        } else if (date.toDateString() === yesterday.toDateString()) {
            return `Yesterday at ${date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`
        }

        return date.toLocaleDateString(undefined, {
            month: "short",
            day: "numeric",
            year: "numeric",
        })
    }

    const getStatusText = (item: HistoryItem) => {
        if (item.results.detections.length === 0) {
            return {
                label: "Healthy Fish",
                subtitle: "No problems detected",
                color: getBoundingBoxColor("healthy"),
            }
        }
        const topDetection = item.results.detections[0]
        const label = getDiseaseLabel(topDetection.diseaseClass)
        const color = getBoundingBoxColor(topDetection.diseaseClass)

        return {
            label,
            subtitle: `${item.results.detections.length} problem${item.results.detections.length > 1 ? "s" : ""} found`,
            color,
            confidence: topDetection.confidence,
        }
    }

    const renderItem = ({ item }: { item: HistoryItem }) => {
        const status = getStatusText(item)
        debug("Rendering history item", { id: item.id, label: status.label })

        return (
            <Pressable
                className="flex-row items-center bg-card rounded-xl p-4 mb-3 border-2 border-border active:border-primary"
                onPress={() => handleHistoryItemPress(item)}
            >
                <View className="relative">
                    <Image
                        source={{ uri: item.processedImageUri }}
                        className="w-16 h-16 rounded-lg bg-secondary"
                        contentFit="cover"
                        transition={300}
                    />
                    <View
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-card"
                        style={{ backgroundColor: status.color }}
                    />
                </View>
                <View className="flex-1 ml-4 justify-center">
                    <Text className="font-bold text-base text-foreground">
                        {status.label}
                    </Text>
                    <Text className="text-sm text-muted-foreground mt-0.5">
                        {status.subtitle}
                    </Text>
                    <Text className="text-xs text-muted-foreground/70 mt-1">
                        {formatDate(item.timestamp)}
                    </Text>
                </View>
                <Icon
                    as={ChevronRightIcon}
                    size={24}
                    className="text-primary/60"
                />
            </Pressable>
        )
    }

    return (
        <View className="flex-1 bg-background">
            <View className="pt-14 pb-4 px-5 bg-card border-b border-border">
                <Text className="text-2xl font-bold text-foreground">
                    My Fish Checks
                </Text>
                <Text className="text-muted-foreground mt-1 text-base">
                    View your past health scans
                </Text>
            </View>

            {isLoading ? (
                <View className="flex-1 items-center justify-center">
                    <View className="w-12 h-12 rounded-full border-4 border-primary/20 border-t-primary animate-spin" />
                </View>
            ) : historyItems.length === 0 ? (
                <View className="flex-1 items-center justify-center px-6">
                    <View className="w-20 h-20 rounded-full bg-secondary items-center justify-center mb-4">
                        <Icon
                            as={ClockIcon}
                            size={32}
                            className="text-primary/40"
                        />
                    </View>
                    <Text className="text-xl font-bold text-foreground text-center">
                        No fish checked yet
                    </Text>
                    <Text className="text-center text-muted-foreground mt-2 text-base">
                        Take a photo of your fish to check their health
                    </Text>
                </View>
            ) : (
                <FlatList
                    data={historyItems}
                    renderItem={renderItem}
                    keyExtractor={(item) => item.id}
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={{ padding: 16, paddingBottom: 120 }}
                    initialNumToRender={5}
                    maxToRenderPerBatch={10}
                />
            )}
        </View>
    )
}
