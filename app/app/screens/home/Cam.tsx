import { MaterialIcons } from "@expo/vector-icons"
import { CameraView, FlashMode, useCameraPermissions } from "expo-camera"
import * as ImagePicker from "expo-image-picker"
import { LinearGradient } from "expo-linear-gradient"
import * as MediaLibrary from "expo-media-library"
import { Wheat } from "lucide-react-native"
import { useEffect, useRef, useState } from "react"
import { Alert, Image, Text, TouchableOpacity, View } from "react-native"
import { useCamera } from "@/lib/camera"

type CameraType = "front" | "back"

export default function CameraScreen() {
    const cameraRef = useRef<CameraView | null>(null)
    const [permission, requestPermission] = useCameraPermissions()
    const [mediaPermission, requestMediaPermission] =
        MediaLibrary.usePermissions()
    const {
        cameraFacing,
        flashMode,
        torchEnabled,
        setCameraFacing,
        setFlashMode,
        setTorchEnabled,
    } = useCamera()
    const [capturedPhoto, setCapturedPhoto] = useState<string | null>(null)

    useEffect(() => {
        if (!permission?.granted) {
            requestPermission()
        }
    }, [permission, requestPermission])

    const takePicture = async () => {
        if (!cameraRef.current) return

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 1,
                base64: false,
            })

            if (photo) {
                // Show photo preview
                setCapturedPhoto(photo.uri)
            }
        } catch (error) {
            console.error("Error taking picture:", error)
            Alert.alert("Error", "Failed to take picture")
        }
    }

    const savePhoto = async () => {
        if (!capturedPhoto) return

        try {
            //media perms
            if (!mediaPermission?.granted) {
                const { granted } = await requestMediaPermission()
                if (!granted) {
                    Alert.alert(
                        "Permission needed",
                        "Please grant media library permission to save photos",
                    )
                    return
                }
            }

            // Sabe to media library
            await MediaLibrary.saveToLibraryAsync(capturedPhoto)
            Alert.alert("Success", "Photo saved to gallery!")
            setCapturedPhoto(null)
        } catch (error) {
            console.error("Error saving picture:", error)
            Alert.alert("Error", "Failed to save picture")
        }
    }

    const retakePhoto = () => {
        setCapturedPhoto(null)
    }

    const pickImage = async () => {
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.Images,
                allowsEditing: false,
                quality: 1,
            })

            if (!result.canceled && result.assets[0]) {
                setCapturedPhoto(result.assets[0].uri)
            }
        } catch (error) {
            console.error("Error picking image:", error)
            Alert.alert("Error", "Failed to pick image")
        }
    }

    if (!permission?.granted) {
        return (
            <View className="flex-1 items-center justify-center bg-black">
                <Text className="text-white">Camera permission required</Text>
            </View>
        )
    }

    // Show captured photo preview
    if (capturedPhoto) {
        return (
            <View className="flex-1 bg-black">
                <Image
                    source={{ uri: capturedPhoto }}
                    style={{ flex: 1 }}
                    resizeMode="contain"
                />

                {/* Preview Controls */}
                <View className="absolute bottom-0 left-0 right-0 pb-10 pt-6 bg-black/80">
                    <View className="flex-row justify-around items-center px-10">
                        <TouchableOpacity
                            className="bg-red-500 px-8 py-4 rounded-full"
                            onPress={retakePhoto}
                        >
                            <Text className="text-white font-bold text-lg">
                                Retake
                            </Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            className="bg-green-500 px-8 py-4 rounded-full"
                            onPress={savePhoto}
                        >
                            <Text className="text-white font-bold text-lg">
                                Save
                            </Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        )
    }

    return (
        <View className="flex-1 bg-black">
            {/* Camera */}
            <CameraView
                ref={cameraRef}
                facing={cameraFacing}
                flash={flashMode}
                enableTorch={torchEnabled}
                style={{ flex: 1 }}
                //donot use tailwind styles here T_T ( 1hour it took for my stupid ass to realise) - UwU
            />

            {/* Top Bar */}
            <View className="absolute top-12 left-0 right-0 flex-row justify-between px-6">
                <Icon name="close" />
                <View className="flex-row gap-4">
                    <TouchableOpacity
                        className="w-12 h-12 rounded-full bg-black/40 items-center justify-center"
                        onPress={() => {
                            const next =
                                flashMode === "off"
                                    ? "on"
                                    : flashMode === "on"
                                      ? "auto"
                                      : "off"
                            setFlashMode(next)
                        }}
                    >
                        <MaterialIcons
                            name={
                                flashMode === "off"
                                    ? "flash-off"
                                    : flashMode === "on"
                                      ? "flash-on"
                                      : "flash-auto"
                            }
                            size={24}
                            color="white"
                        />
                    </TouchableOpacity>
                    <TouchableOpacity
                        className="w-12 h-12 rounded-full bg-black/40 items-center justify-center"
                        onPress={() => {
                            setTorchEnabled(!torchEnabled)
                        }}
                    >
                        <MaterialIcons
                            name={
                                torchEnabled
                                    ? "flashlight-on"
                                    : "flashlight-off"
                            }
                            size={24}
                            color="white"
                        />
                    </TouchableOpacity>
                    <TouchableOpacity
                        className="w-12 h-12 rounded-full bg-black/40 items-center justify-center"
                        onPress={() => {
                            setCameraFacing(
                                cameraFacing === "back" ? "front" : "back",
                            )
                        }}
                    >
                        <MaterialIcons
                            name="cameraswitch"
                            size={24}
                            color="white"
                        />
                    </TouchableOpacity>
                </View>
            </View>

            {/* Instruction */}
            <View className="absolute top-1/4 w-full items-center">
                <View className="bg-black/50 px-6 py-2 rounded-full">
                    <Text className="text-white text-base">
                        Point at a fish to detect
                    </Text>
                </View>
            </View>

            {/* Scan Overlay */}
            <View className="absolute inset-0 items-center justify-center">
                <View className="w-64 h-64 border border-primary/50 rounded-xl">
                    <Corner position="tl" />
                    <Corner position="tr" />
                    <Corner position="bl" />
                    <Corner position="br" />

                    <View className="absolute top-0 left-0 right-0 h-[2px] bg-primary/60" />
                </View>
            </View>

            {/* AI Tag
      <View className="absolute top-1/2 left-2/3">
        <View className="bg-primary px-3 py-1 rounded-lg flex-row gap-2">
          <Text className="text-bg-dark text-xs font-bold">
            BETTA FISH
          </Text>
          <Text className="text-bg-dark/70 text-[10px] font-bold">
            98%
          </Text>
        </View>
        <View className="w-px h-8 bg-primary ml-4" />
      </View> */}

            {/* Bottom Controls */}
            <LinearGradient
                colors={["transparent", "rgba(0,0,0,0.85)"]}
                style={{ position: "absolute", bottom: 0, left: 0, right: 0 }}
            >
                <View className="flex-row justify-between items-center px-10 pb-10 pt-6">
                    <TouchableOpacity
                        className="w-12 h-12 rounded-full bg-black/40 items-center justify-center"
                        onPress={pickImage}
                    >
                        <MaterialIcons
                            name="photo-library"
                            size={24}
                            color="white"
                        />
                    </TouchableOpacity>

                    {/* Shutter */}
                    <View className="w-24 h-24 rounded-full border-4 border-primary/30 items-center justify-center">
                        <TouchableOpacity
                            className="w-16 h-16 bg-white rounded-full active:scale-90"
                            onPress={takePicture}
                        />
                    </View>

                    <Icon name="history" />
                </View>
            </LinearGradient>
        </View>
    )
}

/* ---------- Components ---------- */

function Icon({ name }: { name: any }) {
    return (
        <TouchableOpacity className="w-12 h-12 rounded-full bg-black/40 items-center justify-center">
            <MaterialIcons name={name} size={24} color="white" />
        </TouchableOpacity>
    )
}

function Corner({ position }: { position: "tl" | "tr" | "bl" | "br" }) {
    const base = "absolute w-8 h-8 border-primary"
    const map = {
        tl: "top-[-4px] left-[-4px] border-t-4 border-l-4 rounded-tl-xl",
        tr: "top-[-4px] right-[-4px] border-t-4 border-r-4 rounded-tr-xl",
        bl: "bottom-[-4px] left-[-4px] border-b-4 border-l-4 rounded-bl-xl",
        br: "bottom-[-4px] right-[-4px] border-b-4 border-r-4 rounded-br-xl",
    }

    return <View className={`${base} ${map[position]}`} />
}
