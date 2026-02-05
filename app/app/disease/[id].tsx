import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect } from "react"
import { ScrollView, View } from "react-native"
import { useNavigation } from "@react-navigation/native"
import {
    AlertTriangleIcon,
    CheckCircleIcon,
    InfoIcon,
} from "lucide-react-native"
import { Text } from "@/components/ui/text"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { getDiseaseInfo } from "@/lib/model/disease/info"
import type { DiseaseClass } from "@/lib/model/types"
import { useLogger } from "@/lib/log"

export default function DiseaseInfoScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { id } = useLocalSearchParams<{ id: string }>()
    const diseaseClass = id as DiseaseClass
    const { info, error: logError, debug } = useLogger()

    useEffect(() => {
        info("DiseaseInfoScreen mounted", { diseaseClass })

        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            info("DiseaseInfoScreen unmounted")
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation, diseaseClass])

    if (!diseaseClass) {
        logError("No disease class provided")
        return (
            <View className="flex-1 bg-background items-center justify-center p-4">
                <Text className="text-xl font-bold text-foreground text-center mb-4">
                    Could not load disease information
                </Text>
                <Button onPress={() => router.back()}>
                    <Text>Go Back</Text>
                </Button>
            </View>
        )
    }

    const diseaseInfo = getDiseaseInfo(diseaseClass)
    debug("Loaded disease info", {
        diseaseClass,
        displayName: diseaseInfo.displayName,
    })

    const getSeverityInfo = () => {
        switch (diseaseInfo.severity) {
            case "low":
                return {
                    title: "Low Concern",
                    description: "This condition is usually easy to treat",
                    color: "text-yellow-600",
                    bg: "bg-yellow-500/10",
                    border: "border-yellow-500/30",
                    icon: InfoIcon,
                }
            case "medium":
                return {
                    title: "Medium Concern",
                    description: "This condition needs attention",
                    color: "text-orange-600",
                    bg: "bg-orange-500/10",
                    border: "border-orange-500/30",
                    icon: AlertTriangleIcon,
                }
            case "high":
                return {
                    title: "High Concern",
                    description: "This condition needs immediate attention",
                    color: "text-red-600",
                    bg: "bg-red-500/10",
                    border: "border-red-500/30",
                    icon: AlertTriangleIcon,
                }
            default:
                return {
                    title: "Good News",
                    description: "Your fish appears healthy",
                    color: "text-green-600",
                    bg: "bg-green-500/10",
                    border: "border-green-500/30",
                    icon: CheckCircleIcon,
                }
        }
    }

    const severity = getSeverityInfo()

    const handleBack = () => {
        info("Back to results pressed")
        router.back()
    }

    return (
        <View className="flex-1 bg-background">
            <View
                className={`px-5 pt-14 pb-6 ${severity.bg} border-b ${severity.border}`}
            >
                <View className="flex-row items-center gap-3 mb-3">
                    <View
                        className={`w-12 h-12 rounded-xl ${severity.bg} items-center justify-center border ${severity.border}`}
                    >
                        <Icon
                            as={severity.icon}
                            size={24}
                            className={severity.color}
                        />
                    </View>
                    <View>
                        <Text
                            className={`text-sm font-semibold ${severity.color} uppercase tracking-wide`}
                        >
                            {severity.title}
                        </Text>
                        <Text className="text-xs text-muted-foreground">
                            {severity.description}
                        </Text>
                    </View>
                </View>
                <Text className="text-3xl font-bold text-foreground">
                    {diseaseInfo.displayName}
                </Text>
            </View>

            <ScrollView className="flex-1">
                <View className="p-5">
                    {/* What is it? */}
                    <View className="mb-6">
                        <Text className="text-lg font-bold text-foreground mb-3">
                            What is this?
                        </Text>
                        <View className="bg-card rounded-xl p-5 border border-border">
                            <Text className="text-base text-foreground leading-relaxed">
                                {diseaseInfo.description}
                            </Text>
                        </View>
                    </View>

                    {/* Symptoms */}
                    <View className="mb-6">
                        <Text className="text-lg font-bold text-foreground mb-3">
                            Signs to look for:
                        </Text>
                        <View className="bg-card rounded-xl p-5 border border-border">
                            {diseaseInfo.symptoms.map((symptom, index) => (
                                <View
                                    key={`symptom-${index}`}
                                    className="flex-row items-start mb-3 last:mb-0"
                                >
                                    <Text className="text-primary font-bold mr-3 text-lg">
                                        •
                                    </Text>
                                    <Text className="flex-1 text-base text-foreground leading-relaxed">
                                        {symptom}
                                    </Text>
                                </View>
                            ))}
                        </View>
                    </View>

                    {/* Treatments */}
                    <View className="mb-6">
                        <Text className="text-lg font-bold text-foreground mb-3">
                            How to treat it:
                        </Text>
                        <View className="bg-card rounded-xl p-5 border border-border">
                            {diseaseInfo.treatments.map((treatment, index) => (
                                <View
                                    key={`treatment-${index}`}
                                    className="flex-row items-start mb-4 last:mb-0"
                                >
                                    <View className="w-8 h-8 rounded-lg bg-primary/10 items-center justify-center mr-3">
                                        <Text className="text-primary font-bold">
                                            {index + 1}
                                        </Text>
                                    </View>
                                    <Text className="flex-1 text-base text-foreground leading-relaxed pt-1">
                                        {treatment}
                                    </Text>
                                </View>
                            ))}
                        </View>
                    </View>

                    {/* Disclaimer */}
                    <View className="bg-muted/50 rounded-xl p-4 border border-border">
                        <Text className="text-sm text-muted-foreground text-center leading-relaxed">
                            ⚠️ This information is for guidance only. For serious
                            conditions, consult a veterinarian or fish expert.
                        </Text>
                    </View>
                </View>
            </ScrollView>

            <View className="px-5 pb-8 pt-4 border-t border-border bg-card">
                <Button onPress={handleBack} className="w-full h-14 rounded-xl">
                    <Text className="text-base font-bold">Back to Results</Text>
                </Button>
            </View>
        </View>
    )
}
