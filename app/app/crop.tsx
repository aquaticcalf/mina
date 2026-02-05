import { useLocalSearchParams, useRouter } from "expo-router"
import { CheckIcon } from "lucide-react-native"
import React, { useState, useEffect } from "react"
import { Image as RNImage, Pressable, View, Dimensions } from "react-native"
import { Image } from "expo-image"
import * as ImageManipulator from "expo-image-manipulator"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { useNavigationState } from "@/lib/navigation"
import { useLogger } from "@/lib/log"

export default function CropScreen() {
    const router = useRouter()
    const { setOriginalImageUri } = useNavigationState()
    const { imageUri: originalImageUri } = useLocalSearchParams<{
        imageUri: string
    }>()
    const [croppedImageUri, setCroppedImageUri] = useState<string | null>(null)
    const [isProcessing, setIsProcessing] = useState(true)
    const { info, error: logError, debug } = useLogger()

    const screenWidth = Dimensions.get("window").width
    const imageSize = screenWidth - 32

    // Store original image URI and crop to square immediately on mount
    useEffect(() => {
        info("CropScreen mounted", {
            originalImageUri: originalImageUri?.substring(0, 50),
        })

        if (!originalImageUri) {
            logError("No original image URI provided")
            setIsProcessing(false)
            return
        }

        setOriginalImageUri(originalImageUri)

        const processCropImage = async () => {
            try {
                setIsProcessing(true)
                info("Starting image crop processing")

                // Get actual image dimensions first
                const { width, height } = await new Promise<{
                    width: number
                    height: number
                }>((resolve, reject) => {
                    RNImage.getSize(
                        originalImageUri,
                        (w, h) => resolve({ width: w, height: h }),
                        reject,
                    )
                })

                debug("Got image dimensions", { width, height })

                const size = Math.min(width, height)
                const cropX = (width - size) / 2
                const cropY = (height - size) / 2

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

                info("Image cropped successfully", {
                    croppedUri: result.uri.substring(0, 50),
                })
                setCroppedImageUri(result.uri)
                setIsProcessing(false)
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err)
                logError(
                    "Initial crop failed",
                    err instanceof Error ? err : new Error(message),
                )
                setIsProcessing(false)
            }
        }

        processCropImage()

        return () => {
            info("CropScreen unmounted")
        }
    }, [originalImageUri, setOriginalImageUri])

    const handleCrop = () => {
        if (!croppedImageUri) {
            logError("Cannot navigate - no cropped image URI")
            return
        }
        info("Continue to preview pressed", {
            croppedImageUri: croppedImageUri.substring(0, 50),
        })
        router.push({
            pathname: "/preview",
            params: { imageUri: croppedImageUri, from: "crop" },
        })
    }

    const handleCancel = () => {
        info("Crop cancelled")
        router.push("/")
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
                        contentFit="contain"
                        transition={200}
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
                    onPress={handleCancel}
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
