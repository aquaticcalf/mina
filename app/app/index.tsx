import { useNavigation, useFocusEffect } from "@react-navigation/native"
import { CameraView, useCameraPermissions } from "expo-camera"
import * as ImageManipulator from "expo-image-manipulator"
import * as ImagePicker from "expo-image-picker"
import * as MediaLibrary from "expo-media-library"
import { useRouter } from "expo-router"
import {
    FlashlightIcon,
    FlashlightOffIcon,
    HistoryIcon,
    ImageIcon,
    Minus,
    Plus,
    RefreshCcwIcon,
    SettingsIcon,
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
import { SliderComponent } from "@/components/ui/slider"
import { Text } from "@/components/ui/text"
import { useCamera } from "@/lib/camera"
import { useSettings } from "@/lib/settings"

export default function HomeCameraScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const cameraRef = React.useRef<CameraView>(null)
    const { width: screenW, height: screenH } = useWindowDimensions()
    const camSize = Math.min(screenW, screenH) - 40
    const [permission, requestPermission] = useCameraPermissions()
    const [lastCaptureUri, setLastCaptureUri] = React.useState<string | null>(
        null,
    )
    const [isGalleryLoading, setIsGalleryLoading] = React.useState(false)
    const [zoom, setZoom] = React.useState(0)
    const [showCarousel, setShowCarousel] = React.useState(false)
    const [recentPhotos, setRecentPhotos] = React.useState<
        MediaLibrary.Asset[]
    >([])
    const {
        cameraFacing,
        setCameraFacing,
        flashMode,
        setFlashMode,
        isCameraActive,
        setCameraActive,
    } = useCamera()
    const { saveCapturesToPhotos } = useSettings()

    const [touchStart, setTouchStart] = React.useState<number | null>(null)

    const loadRecentPhotos = async () => {
        try {
            const { status } = await MediaLibrary.requestPermissionsAsync()
            if (status !== "granted") {
                console.warn("media library permission denied")
                return
            }

            const photos = await MediaLibrary.getAssetsAsync({
                mediaType: "photo",
                first: 20,
                sortBy: [["creationTime", false]],
            })
            setRecentPhotos(photos.assets)
        } catch (err) {
            console.warn("failed to load recent photos", err)
        }
    }

    React.useEffect(() => {
        loadRecentPhotos()
    }, [])

    const selectPhotoFromCarousel = async (asset: MediaLibrary.Asset) => {
        try {
            const assetInfo = await MediaLibrary.getAssetInfoAsync(asset)
            const uri = assetInfo.localUri || assetInfo.uri
            if (uri) {
                setLastCaptureUri(uri)
                setShowCarousel(false)
                router.push({
                    pathname: "/crop",
                    params: { imageUri: uri },
                })
            }
        } catch (err) {
            console.warn("failed to load photo", err)
        }
    }

    const openGallery = async () => {
        if (isGalleryLoading) return
        setIsGalleryLoading(true)
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ["images"],
                quality: 0.85,
                base64: false,
            })
            if (!result.canceled && result.assets[0]) {
                const uri = result.assets[0].uri
                setLastCaptureUri(uri)
                router.push({
                    pathname: "/crop",
                    params: { imageUri: uri },
                })
            }
        } catch (err) {
            console.warn("gallery launch failed", err)
        } finally {
            setIsGalleryLoading(false)
        }
    }

    const handleTouchStart = (e: any) => {
        setTouchStart(e.nativeEvent.pageY)
    }

    const handleTouchEnd = (e: any) => {
        if (touchStart === null) return
        const touchEnd = e.nativeEvent.pageY
        const diff = touchStart - touchEnd
        if (diff > 50) {
            openGallery()
        }
        setTouchStart(null)
    }

    const canShowCamera = permission?.granted && isCameraActive

    // Keep camera active when index screen is focused
    useFocusEffect(
        React.useCallback(() => {
            setCameraActive(true)
        }, [setCameraActive]),
    )

    const toggleFlash = () => {
        const modes: ("off" | "on")[] = ["off", "on"]
        const currentIndex = modes.indexOf(flashMode)
        const nextIndex = (currentIndex + 1) % modes.length
        const newMode = modes[nextIndex]
        setFlashMode(newMode)
    }

    const getFlashIcon = () => {
        switch (flashMode) {
            case "on":
                return FlashlightIcon
            case "off":
            default:
                return FlashlightOffIcon
        }
    }

    const getFlashIconProps = () => {
        switch (flashMode) {
            case "on":
                return { className: "text-foreground" }
            case "off":
            default:
                return { className: "text-foreground/50" }
        }
    }

    const capture = async () => {
        try {
            const photo = await cameraRef.current?.takePictureAsync({
                quality: 0.85,
                base64: false,
                exif: false,
                skipProcessing: false,
            })

            if (photo?.uri) {
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

                // Save to media library if enabled
                if (saveCapturesToPhotos) {
                    try {
                        const { status } =
                            await MediaLibrary.requestPermissionsAsync()
                        if (status === "granted") {
                            const album =
                                await MediaLibrary.getAlbumAsync("mina")
                            if (album) {
                                await MediaLibrary.addAssetsToAlbumAsync(
                                    [manipulatedImage.uri],
                                    album,
                                    false,
                                )
                            } else {
                                const createdAlbum =
                                    await MediaLibrary.createAlbumAsync(
                                        "mina",
                                        manipulatedImage.uri,
                                        false,
                                    )
                                if (!createdAlbum) {
                                    console.warn("failed to create mina album")
                                }
                            }
                        }
                    } catch (err) {
                        console.warn(
                            "failed to save photo to media library",
                            err,
                        )
                    }
                }

                setLastCaptureUri(manipulatedImage.uri)
                router.push({
                    pathname: "/preview",
                    params: { imageUri: manipulatedImage.uri },
                })
            }
        } catch (err) {
            console.warn("capture failed", err)
        }
    }

    if (!permission) {
        return <View className="flex-1 bg-background" />
    }

    if (!permission.granted) {
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

    if (!isCameraActive) {
        return <View className="flex-1 bg-background" />
    }

    return (
        <View className="flex-1 items-center justify-center bg-background">
            <View className="absolute top-0 left-0 right-0 flex-row justify-between pt-20 px-4 z-10">
                <View className="items-center">
                    <Pressable
                        onPress={toggleFlash}
                        className="h-16 w-16 items-center justify-center rounded-full bg-background/80"
                    >
                        <Icon
                            as={getFlashIcon()}
                            size={32}
                            {...getFlashIconProps()}
                        />
                    </Pressable>
                </View>
                <View className="flex-row gap-3">
                    <Pressable
                        onPress={() => router.push("/history")}
                        className="h-16 w-16 items-center justify-center rounded-full bg-background/80"
                    >
                        <Icon
                            as={HistoryIcon}
                            size={32}
                            className="text-foreground"
                        />
                    </Pressable>
                    <Pressable
                        onPress={() => router.push("/settings")}
                        className="h-16 w-16 items-center justify-center rounded-full bg-background/80"
                    >
                        <Icon
                            as={SettingsIcon}
                            size={32}
                            className="text-foreground"
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
                    marginTop: -40,
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
                    zoom={zoom}
                    onCameraReady={() =>
                        console.log("Camera ready with flash mode :", flashMode)
                    }
                />
            </View>

            <View className="w-full px-6 py-4">
                <View className="flex-row items-center gap-4">
                    <Pressable onPress={() => setZoom(Math.max(0, zoom - 0.1))}>
                        <Icon
                            as={Minus}
                            size={20}
                            className="text-foreground"
                        />
                    </Pressable>
                    <View className="flex-1">
                        <SliderComponent
                            value={zoom}
                            onValueChange={setZoom}
                            minimumValue={0}
                            maximumValue={1}
                            step={0.01}
                        />
                    </View>
                    <Pressable onPress={() => setZoom(Math.min(1, zoom + 0.1))}>
                        <Icon as={Plus} size={20} className="text-foreground" />
                    </Pressable>
                </View>
            </View>

            <View
                className="absolute inset-x-0 bottom-0 items-center bg-transparent px-10 pb-16 pt-6"
                onTouchStart={handleTouchStart}
                onTouchEnd={handleTouchEnd}
            >
                <View className="flex-row items-center justify-between self-stretch">
                    <Pressable
                        className="h-16 w-16 items-center justify-center rounded-full bg-background/20"
                        onPress={openGallery}
                        disabled={isGalleryLoading}
                    >
                        <Icon
                            as={ImageIcon}
                            size={28}
                            className={`${isGalleryLoading ? "text-foreground/50" : "text-foreground"}`}
                        />
                    </Pressable>

                    <Pressable
                        onPress={capture}
                        className="h-24 w-24 items-center justify-center rounded-full border-2 border-foreground/70 bg-background/15"
                    >
                        <View className="h-20 w-20 rounded-full bg-foreground/90" />
                    </Pressable>

                    <Pressable
                        className="h-16 w-16 items-center justify-center rounded-full bg-background/20"
                        onPress={() =>
                            setCameraFacing(
                                cameraFacing === "back" ? "front" : "back",
                            )
                        }
                    >
                        <Icon
                            as={RefreshCcwIcon}
                            size={28}
                            className="text-foreground"
                        />
                    </Pressable>
                </View>
            </View>
        </View>
    )
}
