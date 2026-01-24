import { useLocalSearchParams, useRouter } from "expo-router"
import React from "react"
import { Image, View } from "react-native"
import { useNavigation } from "@react-navigation/native"
import { Button } from "@/components/ui/button"
import { Text } from "@/components/ui/text"
import { useNavigationState } from "@/lib/navigation"

export default function PreviewScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { originalImageUri } = useNavigationState()
    const { imageUri, from } = useLocalSearchParams<{
        imageUri: string
        from?: string
    }>()
    const cameFromCrop = from === "crop"

    React.useEffect(() => {
        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation])

    return (
        <View className="flex-1 bg-black">
            <View className="absolute top-0 left-0 right-0 z-10 flex-row justify-center items-center pt-12 px-4">
                <Text className="text-white font-medium">Preview</Text>
            </View>

            <View className="flex-1 items-center justify-center">
                <Image
                    source={{ uri: imageUri }}
                    className="w-full h-full"
                    resizeMode="contain"
                />
            </View>

            <View className="absolute bottom-0 left-0 right-0 flex-row gap-3 px-4 pb-8">
                <Button
                    variant="outline"
                    className="flex-1 h-12"
                    onPress={() =>
                        cameFromCrop && originalImageUri
                            ? router.replace({
                                  pathname: "/crop",
                                  params: { imageUri: originalImageUri },
                              })
                            : router.push("/")
                    }
                >
                    <Text>Back</Text>
                </Button>
                <Button
                    className="flex-1 h-12"
                    onPress={() => {
                        console.log("Analyze image:", imageUri)
                    }}
                >
                    <Text>Analyze Fish</Text>
                </Button>
            </View>
        </View>
    )
}
