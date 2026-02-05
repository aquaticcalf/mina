import { useIsFocused } from "@react-navigation/native"
import { CameraView, useCameraPermissions } from "expo-camera"
import * as ImageManipulator from "expo-image-manipulator"
import * as ImagePicker from "expo-image-picker"
import { useRouter } from "expo-router"
import {
    ImageIcon,
    RefreshCcwIcon,
    ZapIcon,
    ZapOffIcon,
} from "lucide-react-native"
import * as React from "react"
import {
    Linking,
    Platform,
    Pressable,
    useWindowDimensions,
    View,
} from "react-native"
import { Button } from "@/components/ui/button"
import { Icon } from "@/components/ui/icon"
import { Text } from "@/components/ui/text"
import { useCamera } from "@/lib/camera"
import { useLogger } from "@/lib/log"

export default function HomeCameraScreen() {
    const router = useRouter()
    const cameraRef = React.useRef<CameraView>(null)
    const { width: screenW, height: screenH } = useWindowDimensions()
    const camSize = Math.min(screenW, screenH) - 40
    const isFocused = useIsFocused()
    const [permission, requestPermission] = useCameraPermissions()
    const [lastCaptureUri, setLastCaptureUri] = React.useState<string | null>(
        null,
    )
    const [isGalleryLoading, setIsGalleryLoading] = React.useState(false)
    const { cameraFacing, setCameraFacing, flashMode, setFlashMode } =
        useCamera()
    const { info, error: logError } = useLogger()

    const canShowCamera = permission?.granted && isFocused

    React.useEffect(() => {
        info("HomeCameraScreen mounted")
        return () => {
            info("HomeCameraScreen unmounted")
        }
    }, [])

    const toggleFlash = () => {
        const modes: ("off" | "on")[] = ["off", "on"]
        const currentIndex = modes.indexOf(flashMode)
        const nextIndex = (currentIndex + 1) % modes.length
        const newMode = modes[nextIndex]
        setFlashMode(newMode)
        info("Flash mode toggled", { from: flashMode, to: newMode })
    }

    const getFlashIcon = () => {
        switch (flashMode) {
            case "on":
                return ZapIcon
            case "off":
            default:
                return ZapOffIcon
        }
    }

    const getFlashIconProps = () => {
        switch (flashMode) {
            case "on":
                return { className: "text-yellow-500" }
            case "off":
            default:
                return { className: "text-foreground/50" }
        }
    }

    React.useEffect(() => {
        if (!canShowCamera) return
        if (Platform.OS === "web") return
    }, [canShowCamera])

    const capture = async () => {
        info("Capture initiated")
        try {
            const photo = await cameraRef.current?.takePictureAsync({
                quality: 0.85,
                base64: false,
                exif: false,
                skipProcessing: false,
            })

            if (photo?.uri) {
                info("Photo captured", {
                    uri: photo.uri,
                    width: photo.width,
                    height: photo.height,
                })

                // Crop to square to match camera view
                const { width, height } = photo
                const size = Math.min(width, height)
                const cropX = (width - size) / 2
                const cropY = (height - size) / 2

                const manipulatedImage = await ImageManipulator.manipulateAsync(
                    photo.uri,
                    [
                        ...(cameraFacing === "front"
                            ? [{ flip: ImageManipulator.FlipType.Horizontal }]
                            : []),
                        {
                            crop: {
                                originX: cropX,
                                originY: cropY,
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

                info("Image manipulated", {
                    manipulatedUri: manipulatedImage.uri,
                })
                setLastCaptureUri(manipulatedImage.uri)
                router.push({
                    pathname: "/preview",
                    params: { imageUri: manipulatedImage.uri },
                })
            } else {
                logError("Photo capture returned no URI")
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err)
            logError(
                "Capture failed",
                err instanceof Error ? err : new Error(message),
            )
        }
    }

    const handleGalleryPick = async () => {
        if (isGalleryLoading) return
        setIsGalleryLoading(true)
        info("Gallery picker initiated")

        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ["images"],
                quality: 0.85,
                base64: false,
            })
            if (!result.canceled && result.assets[0]) {
                const uri = result.assets[0].uri
                info("Image selected from gallery", { uri })
                setLastCaptureUri(uri)
                router.push({
                    pathname: "/crop",
                    params: { imageUri: uri },
                })
            } else {
                info("Gallery picker cancelled")
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err)
            logError(
                "Gallery launch failed",
                err instanceof Error ? err : new Error(message),
            )
        } finally {
            setIsGalleryLoading(false)
        }
    }

    const handleCameraFacingChange = () => {
        const newFacing = cameraFacing === "back" ? "front" : "back"
        setCameraFacing(newFacing)
        info("Camera facing changed", { from: cameraFacing, to: newFacing })
    }

    if (!permission) {
        return <View className="flex-1 bg-background" />
    }

    if (!canShowCamera) {
        return (
            <View className="flex-1 items-center justify-center bg-background px-6">
                <Text className="text-xl font-semibold text-foreground">
                    Camera access needed
                </Text>
                <Text className="mt-2 text-center text-sm text-muted-foreground">
                    Mina needs camera access to scan your fish.
                </Text>

                <View className="mt-6 w-full gap-3">
                    <Button className="h-12" onPress={requestPermission}>
                        <Text className="text-base">Allow camera</Text>
                    </Button>

                    <Button
                        variant="outline"
                        className="h-12"
                        onPress={() => {
                            if (Platform.OS !== "web") Linking.openSettings()
                        }}
                    >
                        <Text className="text-base">Open Settings</Text>
                    </Button>
                </View>
            </View>
        )
    }

    return (
        <View className="flex-1 items-center justify-center bg-background">
            <View className="absolute top-0 left-0 right-0 flex-row justify-between pt-12 px-4 z-10">
                <View className="w-10" />
                <View className="items-center">
                    <Pressable
                        onPress={toggleFlash}
                        className="h-10 w-10 items-center justify-center rounded-full bg-background/80"
                    >
                        <Icon
                            as={getFlashIcon()}
                            size={18}
                            {...getFlashIconProps()}
                        />
                    </Pressable>
                </View>
            </View>
            <View
                style={{
                    width: camSize,
                    height: camSize,
                    overflow: "hidden",
                    borderRadius: 12,
                }}
            >
                <CameraView
                    ref={cameraRef}
                    style={{
                        width: camSize,
                        height: camSize,
                    }}
                    facing={cameraFacing}
                    mode="picture"
                    flash={flashMode}
                    onCameraReady={() =>
                        info("Camera ready with flash mode", { flashMode })
                    }
                />
            </View>

            <View className="absolute inset-x-0 bottom-0 items-center bg-transparent px-10 pb-6 pt-6">
                <View className="flex-row items-center justify-between self-stretch">
                    <Pressable
                        className="h-12 w-12 items-center justify-center rounded-full bg-background/20"
                        onPress={handleGalleryPick}
                        disabled={isGalleryLoading}
                    >
                        <Icon
                            as={ImageIcon}
                            size={20}
                            className={`${isGalleryLoading ? "text-foreground/50" : "text-foreground"}`}
                        />
                    </Pressable>

                    <Pressable
                        onPress={capture}
                        className="h-20 w-20 items-center justify-center rounded-full border-2 border-foreground/70 bg-background/15"
                    >
                        <View className="h-[68px] w-[68px] rounded-full bg-foreground/90" />
                    </Pressable>

                    <Pressable
                        className="h-12 w-12 items-center justify-center rounded-full bg-background/20"
                        onPress={handleCameraFacingChange}
                    >
                        <Icon
                            as={RefreshCcwIcon}
                            size={20}
                            className="text-foreground"
                        />
                    </Pressable>
                </View>
            </View>
        </View>
    )
}
