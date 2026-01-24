import { useIsFocused } from "@react-navigation/native"
import { CameraView, useCameraPermissions } from "expo-camera"
import { ImageIcon, RefreshCcwIcon } from "lucide-react-native"
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

export default function HomeCameraScreen() {
    const cameraRef = React.useRef<CameraView>(null)
    const { width: screenW, height: screenH } = useWindowDimensions()
    const [ratio, setRatio] = React.useState<"4:3" | "16:9">("4:3")
    const isFocused = useIsFocused()
    const [permission, requestPermission] = useCameraPermissions()
    const [lastCaptureUri, setLastCaptureUri] = React.useState<string | null>(
        null,
    )
    const [facing, setFacing] = React.useState<"back" | "front">("back")

    const canShowCamera = permission?.granted && isFocused

    React.useEffect(() => {
        if (!canShowCamera) return
        if (Platform.OS === "web") return

        let cancelled = false
        const pickBestRatio = async () => {
            try {
                const ratios = await (
                    cameraRef.current as unknown as {
                        getSupportedRatiosAsync?: () => Promise<string[]>
                    }
                )?.getSupportedRatiosAsync?.()

                if (!ratios || ratios.length === 0) return

                const desired = screenW / screenH
                let best: "4:3" | "16:9" = "4:3"
                let bestDelta = Number.POSITIVE_INFINITY

                for (const r of ratios) {
                    if (r !== "4:3" && r !== "16:9") continue
                    const [w, h] = r.split(":").map(Number)
                    const value = w / h
                    const delta = Math.abs(value - desired)
                    if (delta < bestDelta) {
                        bestDelta = delta
                        best = r
                    }
                }

                if (!cancelled) setRatio(best)
            } catch {
                // ignore: ratio discovery is best-effort
            }
        }

        pickBestRatio()
        return () => {
            cancelled = true
        }
    }, [canShowCamera, screenH, screenW])

    const capture = async () => {
        try {
            const photo = await cameraRef.current?.takePictureAsync({
                quality: 0.85,
                base64: false,
                exif: false,
            })

            if (photo?.uri) setLastCaptureUri(photo.uri)
        } catch (err) {
            console.warn("capture failed", err)
        }
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
        <View className="flex-1 bg-black">
            <CameraView
                ref={cameraRef}
                className="absolute inset-0"
                facing={facing}
                mode="picture"
                ratio={ratio}
            />

            <View className="absolute inset-x-0 bottom-0 items-center bg-transparent px-6 pb-24 pt-6">
                <View className="flex-row items-center justify-between self-stretch">
                    <Pressable className="h-12 w-12 items-center justify-center rounded-full bg-black/20">
                        <Icon as={ImageIcon} size={20} className="text-white" />
                    </Pressable>

                    <Pressable
                        onPress={capture}
                        className="h-20 w-20 items-center justify-center rounded-full border-2 border-white/70 bg-black/15"
                    >
                        <View className="h-[68px] w-[68px] rounded-full bg-white/90" />
                    </Pressable>

                    <Pressable
                        className="h-12 w-12 items-center justify-center rounded-full bg-black/20"
                        onPress={() =>
                            setFacing((v) => (v === "back" ? "front" : "back"))
                        }
                    >
                        <Icon
                            as={RefreshCcwIcon}
                            size={20}
                            className="text-white"
                        />
                    </Pressable>
                </View>
            </View>
        </View>
    )
}
