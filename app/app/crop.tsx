import { useLocalSearchParams, useRouter } from "expo-router"
import { CheckIcon } from "lucide-react-native"
import React, { useState, useEffect } from "react"
import { Image, Pressable, View, Dimensions } from "react-native"
import * as ImageManipulator from "expo-image-manipulator"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { useNavigationState } from "@/lib/navigation"

export default function CropScreen() {
    const router = useRouter()
    const { setOriginalImageUri } = useNavigationState()
    const { imageUri: originalImageUri } = useLocalSearchParams<{
        imageUri: string
    }>()
    const [croppedImageUri, setCroppedImageUri] = useState<string | null>(null)
    const [isProcessing, setIsProcessing] = useState(true)

    const screenWidth = Dimensions.get("window").width
    const imageSize = screenWidth - 32

    // Store original image URI and crop to square immediately on mount
    useEffect(() => {
        if (!originalImageUri) return

        setOriginalImageUri(originalImageUri)

        const processCropImage = async () => {
            try {
                setIsProcessing(true)

                // Get dimensions
                const { width, height } = await new Promise<{
                    width: number
                    height: number
                }>((resolve, reject) => {
                    Image.getSize(
                        originalImageUri,
                        (w, h) => resolve({ width: w, height: h }),
                        reject,
                    )
                })

                const size = Math.min(width, height)
                const cropX = (width - size) / 2
                const cropY = (height - size) / 2

                // Crop immediately
                const result = await ImageManipulator.manipulateAsync(
                    originalImageUri,
                    [
                        {
                            crop: {
                                originX: Math.round(cropX),
                                originY: Math.round(cropY),
                                width: size,
                                height: size,
                            },
                        },
                    ],
                    {
                        compress: 0.85,
                        format: ImageManipulator.SaveFormat.JPEG,
                    },
                )

                setCroppedImageUri(result.uri)
                setIsProcessing(false)
            } catch (err) {
                console.error("Initial crop failed:", err)
                setIsProcessing(false)
            }
        }

        processCropImage()
    }, [originalImageUri, setOriginalImageUri])

    const handleCrop = () => {
        if (!croppedImageUri) return
        router.push({
            pathname: "/preview",
            params: { imageUri: croppedImageUri, from: "crop" },
        })
    }

    if (!croppedImageUri || isProcessing) {
        return (
            <View className="flex-1 items-center justify-center bg-background">
                <Text className="text-muted-foreground">
                    Processing image...
                </Text>
            </View>
        )
    }

    return (
        <View className="flex-1 bg-background">
            {/* Header */}
            <View className="flex-row justify-center items-center pt-12 px-4 pb-4">
                <Text className="text-lg font-semibold text-foreground">
                    Crop to Square
                </Text>
            </View>

            {/* Image preview - shows the cropped square */}
            <View className="flex-1 items-center justify-center px-4">
                <View
                    style={{
                        width: imageSize,
                        height: imageSize,
                        backgroundColor: "#000",
                        borderRadius: 12,
                        overflow: "hidden",
                        borderWidth: 2,
                        borderColor: "#fff",
                    }}
                >
                    <Image
                        source={{ uri: croppedImageUri }}
                        style={{
                            width: imageSize,
                            height: imageSize,
                        }}
                        resizeMode="contain"
                    />

                    {/* Grid overlay */}
                    <View
                        style={{
                            position: "absolute",
                            top: 0,
                            left: imageSize / 3,
                            width: 1,
                            height: imageSize,
                            backgroundColor: "rgba(255, 255, 255, 0.2)",
                        }}
                    />
                    <View
                        style={{
                            position: "absolute",
                            top: 0,
                            left: (imageSize * 2) / 3,
                            width: 1,
                            height: imageSize,
                            backgroundColor: "rgba(255, 255, 255, 0.2)",
                        }}
                    />
                    <View
                        style={{
                            position: "absolute",
                            top: imageSize / 3,
                            left: 0,
                            width: imageSize,
                            height: 1,
                            backgroundColor: "rgba(255, 255, 255, 0.2)",
                        }}
                    />
                    <View
                        style={{
                            position: "absolute",
                            top: (imageSize * 2) / 3,
                            left: 0,
                            width: imageSize,
                            height: 1,
                            backgroundColor: "rgba(255, 255, 255, 0.2)",
                        }}
                    />
                </View>
            </View>

            {/* Info text */}
            <View className="px-6 py-4">
                <Text className="text-center text-sm text-muted-foreground">
                    Image cropped to square from center
                </Text>
            </View>

            {/* Controls */}
            <View className="flex-row gap-3 px-4 pb-8">
                <Button
                    variant="outline"
                    className="flex-1 h-12"
                    onPress={() => router.push("/")}
                    disabled={isProcessing}
                >
                    <Text>Cancel</Text>
                </Button>
                <Button className="flex-1 h-12" onPress={handleCrop}>
                    <Text>Continue</Text>
                </Button>
            </View>
        </View>
    )
}
