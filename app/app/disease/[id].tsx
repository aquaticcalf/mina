import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect } from "react"
import { ScrollView, View } from "react-native"
import { useNavigation } from "@react-navigation/native"
import { Text } from "@/components/ui/text"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getDiseaseInfo } from "@/lib/model/disease/info"
import type { DiseaseClass } from "@/lib/model/types"

export default function DiseaseInfoScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { id } = useLocalSearchParams<{ id: string }>()
    const diseaseClass = id as DiseaseClass

    // Hide tab bar on this screen
    useEffect(() => {
        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation])

    if (!diseaseClass) {
        return (
            <View className="flex-1 bg-background items-center justify-center p-4">
                <Text className="text-destructive text-center mb-4">
                    No disease specified
                </Text>
                <Button onPress={() => router.back()}>
                    <Text>Go Back</Text>
                </Button>
            </View>
        )
    }

    const diseaseInfo = getDiseaseInfo(diseaseClass)

    // Severity-based color classes
    const getSeverityColors = () => {
        switch (diseaseInfo.severity) {
            case "low":
                return {
                    text: "text-yellow-600",
                    bg: "bg-yellow-100",
                }
            case "medium":
                return {
                    text: "text-orange-600",
                    bg: "bg-orange-100",
                }
            case "high":
                return {
                    text: "text-red-600",
                    bg: "bg-red-100",
                }
            default:
                return {
                    text: "text-green-600",
                    bg: "bg-green-100",
                }
        }
    }

    const colors = getSeverityColors()

    return (
        <View className="flex-1 bg-background">
            {/* Header */}
            <View className={`px-4 pt-12 pb-6 ${colors.bg}`}>
                <Text className={`text-3xl font-bold ${colors.text}`}>
                    {diseaseInfo.displayName}
                </Text>
                <Text className="text-sm text-muted-foreground mt-1 capitalize">
                    Severity: {diseaseInfo.severity}
                </Text>
            </View>

            <ScrollView className="flex-1">
                <View className="p-4">
                    {/* Description */}
                    <Card className="mb-4">
                        <CardHeader>
                            <CardTitle>About</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <Text className="text-muted-foreground">
                                {diseaseInfo.description}
                            </Text>
                        </CardContent>
                    </Card>

                    {/* Symptoms */}
                    <Card className="mb-4">
                        <CardHeader>
                            <CardTitle>Symptoms</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {diseaseInfo.symptoms.map((symptom, index) => (
                                <View
                                    key={`symptom-${index}-${symptom.substring(0, 10)}`}
                                    className="flex-row mb-2"
                                >
                                    <Text className="mr-2">•</Text>
                                    <Text className="flex-1 text-muted-foreground">
                                        {symptom}
                                    </Text>
                                </View>
                            ))}
                        </CardContent>
                    </Card>

                    {/* Treatments */}
                    <Card className="mb-4">
                        <CardHeader>
                            <CardTitle>Recommended Treatments</CardTitle>
                        </CardHeader>
                        <CardContent>
                            {diseaseInfo.treatments.map((treatment, index) => (
                                <View
                                    key={`treatment-${index}-${treatment.substring(0, 10)}`}
                                    className="flex-row mb-2"
                                >
                                    <Text className="mr-2">{index + 1}.</Text>
                                    <Text className="flex-1 text-muted-foreground">
                                        {treatment}
                                    </Text>
                                </View>
                            ))}
                        </CardContent>
                    </Card>
                </View>
            </ScrollView>

            {/* Bottom actions */}
            <View className="px-4 pb-8 pt-4 border-t border-border">
                <Button onPress={() => router.back()} className="w-full">
                    <Text>Back to Results</Text>
                </Button>
            </View>
        </View>
    )
}
