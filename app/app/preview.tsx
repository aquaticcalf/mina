import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect } from "react"
import { View } from "react-native"
import { Image } from "expo-image"
import { useNavigation } from "@react-navigation/native"
import { Button } from "@/components/ui/button"
import { Text } from "@/components/ui/text"
import { useNavigationState } from "@/lib/navigation"
import { useLogger } from "@/lib/log"

export default function PreviewScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { originalImageUri } = useNavigationState()
    const { imageUri, from } = useLocalSearchParams<{
        imageUri: string
        from?: string
    }>()
    const cameFromCrop = from === "crop"
    const { info, debug } = useLogger()

    useEffect(() => {
        info("PreviewScreen mounted", { imageUri, from, cameFromCrop })

        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            info("PreviewScreen unmounted")
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation, imageUri, from, cameFromCrop])

    const handleBack = () => {
        info("Preview back pressed", {
            cameFromCrop,
            hasOriginalImage: !!originalImageUri,
        })
        if (cameFromCrop && originalImageUri) {
            router.replace({
                pathname: "/crop",
                params: { imageUri: originalImageUri },
            })
        } else {
            router.push("/")
        }
    }

    const handleAnalyze = () => {
        info("Analyze fish pressed", { imageUri })
        router.push({
            pathname: "/results",
            params: { imageUri },
        })
    }

    debug("Rendering PreviewScreen", { imageUri: imageUri?.substring(0, 50) })

    return (
        <View className="flex-1 bg-black">
            <View className="absolute top-0 left-0 right-0 z-10 flex-row justify-center items-center pt-12 px-4">
                <Text className="text-white font-medium">Preview</Text>
            </View>

            <View className="flex-1 items-center justify-center">
                <Image
                    source={{ uri: imageUri }}
                    className="w-full h-full"
                    contentFit="contain"
                    transition={200}
                />
            </View>

            <View className="absolute bottom-0 left-0 right-0 flex-row gap-3 px-4 pb-8">
                <Button
                    variant="outline"
                    className="flex-1 h-12"
                    onPress={handleBack}
                >
                    <Text>Back</Text>
                </Button>
                <Button className="flex-1 h-12" onPress={handleAnalyze}>
                    <Text>Analyze Fish</Text>
                </Button>
            </View>
        </View>
    )
}
