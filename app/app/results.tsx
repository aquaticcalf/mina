import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect, useState, useRef } from "react"
import { ScrollView, View } from "react-native"
import { useSafeAreaInsets } from "react-native-safe-area-context"
import { Image } from "expo-image"
import { useNavigation } from "@react-navigation/native"
import { Skia, matchFont, ImageFormat } from "@shopify/react-native-skia"
import * as FileSystem from "expo-file-system/legacy"
import { Button } from "@/components/ui/button"
import { Text } from "@/components/ui/text"
import { useModel } from "@/lib/model"
import type { InferenceResult } from "@/lib/model/inference"
import { saveHistoryItem, getHistoryItem } from "@/lib/history/storage"
import { getBoundingBoxColor, getDiseaseLabel } from "@/lib/model/disease/info"
import { useLogger } from "@/lib/log"

export default function ResultsScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const insets = useSafeAreaInsets()
    const { imageUri, historyId } = useLocalSearchParams<{
        imageUri?: string
        historyId?: string
    }>()
    const { runInference, isReady } = useModel()
    const { info, error: logError, debug } = useLogger()

    const [isAnalyzing, setIsAnalyzing] = useState(true)
    const [results, setResults] = useState<InferenceResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [processedImageUri, setProcessedImageUri] = useState<string | null>(
        null,
    )
    const savedRef = useRef(false)

    useEffect(() => {
        info("ResultsScreen mounted", {
            hasImageUri: !!imageUri,
            hasHistoryId: !!historyId,
            isReady,
        })

        navigation.setOptions({
            tabBarStyle: { display: "none" },
        })

        return () => {
            info("ResultsScreen unmounted")
            navigation.setOptions({
                tabBarStyle: undefined,
            })
        }
    }, [navigation, imageUri, historyId, isReady])

    useEffect(() => {
        async function load() {
            if (historyId) {
                info("Loading from history", { historyId })
                try {
                    setIsAnalyzing(true)
                    const item = await getHistoryItem(historyId)
                    if (item) {
                        info("History item loaded", {
                            historyId,
                            detectionsCount: item.results.detections.length,
                        })
                        setResults(item.results)
                        setProcessedImageUri(item.processedImageUri)
                    } else {
                        logError(
                            "History item not found",
                            new Error(`No item with id: ${historyId}`),
                        )
                        setError("Could not find this scan")
                    }
                } catch (e) {
                    const message = e instanceof Error ? e.message : String(e)
                    logError(
                        "Failed to load history item",
                        e instanceof Error ? e : new Error(message),
                        { historyId },
                    )
                    setError("Could not load this scan")
                } finally {
                    setIsAnalyzing(false)
                }
                return
            }

            if (!imageUri) {
                logError(
                    "No image URI provided",
                    new Error("imageUri is undefined"),
                )
                setError("No photo provided")
                setIsAnalyzing(false)
                return
            }

            if (!isReady) {
                logError(
                    "Model not ready for inference",
                    new Error("Model is not ready"),
                )
                setError("App is not ready. Please try again.")
                setIsAnalyzing(false)
                return
            }

            try {
                setIsAnalyzing(true)
                setError(null)

                info("Starting inference", { imageUri })
                const result = await runInference(imageUri)

                if (!result) {
                    logError("Inference returned null")
                    setError("Could not analyze this photo")
                    setIsAnalyzing(false)
                    return
                }

                info("Inference completed", {
                    detectionsCount: result.detections.length,
                    inferenceTimeMs: result.inferenceTimeMs,
                })
                setResults(result)

                let finalProcessedUri = imageUri

                if (result.detections.length > 0) {
                    info("Processing image with overlays", {
                        detectionsCount: result.detections.length,
                    })
                    try {
                        const processedUri = await processImageWithOverlays(
                            imageUri,
                            result,
                        )
                        setProcessedImageUri(processedUri)
                        finalProcessedUri = processedUri
                        info("Image processed with overlays", { processedUri })
                    } catch (e) {
                        const message =
                            e instanceof Error ? e.message : String(e)
                        logError(
                            "Failed to process image",
                            e instanceof Error ? e : new Error(message),
                        )
                        setProcessedImageUri(imageUri)
                    }
                } else {
                    info("No detections, using original image")
                    setProcessedImageUri(imageUri)
                }

                if (!savedRef.current) {
                    savedRef.current = true
                    try {
                        info("Saving to history")
                        await saveHistoryItem({
                            timestamp: Date.now(),
                            originalImageUri: imageUri,
                            processedImageUri: finalProcessedUri,
                            results: result,
                        })
                        info("Saved to history successfully")
                    } catch (e) {
                        const message =
                            e instanceof Error ? e.message : String(e)
                        logError(
                            "Failed to save to history",
                            e instanceof Error ? e : new Error(message),
                        )
                    }
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err)
                logError(
                    "Analysis failed",
                    err instanceof Error ? err : new Error(message),
                    { imageUri },
                )
                setError(`Analysis failed: ${message}`)
            } finally {
                setIsAnalyzing(false)
            }
        }

        load()
    }, [imageUri, historyId, isReady, runInference])

    const processImageWithOverlays = async (
        uri: string,
        result: InferenceResult,
    ): Promise<string> => {
        debug("Processing image with overlays", {
            uri: uri.substring(0, 50),
            detectionsCount: result.detections.length,
        })

        const base64 = await FileSystem.readAsStringAsync(uri, {
            encoding: "base64",
        })
        const imageData = Skia.Data.fromBase64(base64)
        const image = Skia.Image.MakeImageFromEncoded(imageData)

        if (!image) throw new Error("Could not decode image")

        const width = image.width()
        const height = image.height()

        const surface = Skia.Surface.Make(width, height)
        if (!surface) throw new Error("Could not create surface")

        const canvas = surface.getCanvas()
        canvas.drawImage(image, 0, 0)

        const paint = Skia.Paint()
        paint.setStyle(1)
        paint.setStrokeWidth(4 * (width / 640))
        paint.setAntiAlias(true)

        const fontStyle = {
            fontFamily: "sans-serif",
            fontSize: 24 * (width / 640),
            fontWeight: "bold" as const,
        }
        const font = matchFont(fontStyle)

        const textPaint = Skia.Paint()
        textPaint.setColor(Skia.Color("white"))

        const bgPaint = Skia.Paint()
        bgPaint.setStyle(0)
        bgPaint.setAlphaf(0.7)

        for (const detection of result.detections) {
            const { boundingBox: bbox, diseaseClass, confidence } = detection

            const x = bbox.x * width
            const y = bbox.y * height
            const w = bbox.width * width
            const h = bbox.height * height

            const colorHex = getBoundingBoxColor(diseaseClass)
            paint.setColor(Skia.Color(colorHex))
            bgPaint.setColor(Skia.Color(colorHex))
            bgPaint.setAlphaf(0.7)

            canvas.drawRect({ x, y, width: w, height: h }, paint)

            const label = `${getDiseaseLabel(diseaseClass)} ${(confidence * 100).toFixed(0)}%`
            const textWidth = font.getTextWidth(label)
            const textHeight = font.getSize()

            const labelBgPadding = 8
            canvas.drawRect(
                {
                    x: x,
                    y: y - textHeight - labelBgPadding * 2,
                    width: textWidth + labelBgPadding * 2,
                    height: textHeight + labelBgPadding * 2,
                },
                bgPaint,
            )

            canvas.drawText(
                label,
                x + labelBgPadding,
                y - labelBgPadding,
                textPaint,
                font,
            )
        }

        const snapshot = surface.makeImageSnapshot()
        const resultBase64 = snapshot.encodeToBase64(ImageFormat.PNG, 100)

        const filename = `processed_${Date.now()}.png`
        const resultUri = `${FileSystem.cacheDirectory}${filename}`

        await FileSystem.writeAsStringAsync(resultUri, resultBase64, {
            encoding: "base64",
        })

        debug("Image with overlays saved", { resultUri })

        return resultUri
    }

    const handleTryAgain = () => {
        info("Try again pressed")
        router.push("/")
    }

    const handleCheckAnother = () => {
        info("Check another fish pressed")
        router.push("/")
    }

    const handleLearnMore = (diseaseClass: string) => {
        info("Learn more pressed", { diseaseClass })
        router.push({
            pathname: "/disease/[id]",
            params: { id: diseaseClass },
        })
    }

    return (
        <View className="flex-1 bg-background">
            <View className="pt-14 pb-4 px-5 bg-card border-b border-border">
                <Text className="text-2xl font-bold text-foreground text-center">
                    Health Check Results
                </Text>
            </View>

            <ScrollView
                className="flex-1"
                contentContainerStyle={{
                    paddingTop: 20,
                    paddingBottom: insets.bottom + 100,
                    paddingHorizontal: 16,
                }}
                showsVerticalScrollIndicator={false}
            >
                {(processedImageUri || imageUri) && (
                    <View className="items-center justify-center mb-6">
                        <Image
                            source={{ uri: processedImageUri || imageUri }}
                            className="w-full h-64 rounded-xl"
                            contentFit="contain"
                            transition={200}
                        />
                    </View>
                )}

                {isAnalyzing ? (
                    <View className="items-center justify-center py-12">
                        <View className="w-16 h-16 rounded-full border-4 border-primary/20 border-t-primary animate-spin mb-4" />
                        <Text className="text-xl font-bold text-foreground">
                            Checking your fish...
                        </Text>
                        <Text className="text-muted-foreground mt-2">
                            This will just take a moment
                        </Text>
                    </View>
                ) : error ? (
                    <View className="items-center justify-center px-6 py-12">
                        <View className="w-16 h-16 rounded-full bg-destructive/10 items-center justify-center mb-4">
                            <Text className="text-2xl">❌</Text>
                        </View>
                        <Text className="text-xl font-bold text-foreground text-center mb-2">
                            Something went wrong
                        </Text>
                        <Text className="text-muted-foreground text-center">
                            {error}
                        </Text>
                        <Button className="mt-6 px-6" onPress={handleTryAgain}>
                            <Text>Try Again</Text>
                        </Button>
                    </View>
                ) : results ? (
                    <View>
                        {results.detections.length === 0 ? (
                            <View className="bg-green-500/10 border-2 border-green-500/30 rounded-2xl p-6 items-center">
                                <View className="w-16 h-16 rounded-full bg-green-500/20 items-center justify-center mb-4">
                                    <Text className="text-3xl">✓</Text>
                                </View>
                                <Text className="text-2xl font-bold text-green-600 mb-2 text-center">
                                    Your Fish Looks Healthy!
                                </Text>
                                <Text className="text-muted-foreground text-center text-base">
                                    No diseases or problems were found in this
                                    photo
                                </Text>
                            </View>
                        ) : (
                            <View>
                                <View className="bg-destructive/10 border-2 border-destructive/30 rounded-2xl p-6 mb-6">
                                    <View className="w-16 h-16 rounded-full bg-destructive/20 items-center justify-center mb-4">
                                        <Text className="text-3xl">⚠️</Text>
                                    </View>
                                    <Text className="text-2xl font-bold text-foreground mb-2">
                                        Problems Found
                                    </Text>
                                    <Text className="text-muted-foreground text-base">
                                        {results.detections.length === 1
                                            ? "We found 1 potential issue with your fish"
                                            : `We found ${results.detections.length} potential issues with your fish`}
                                    </Text>
                                </View>

                                <Text className="text-lg font-bold text-foreground mb-4">
                                    What We Found:
                                </Text>

                                {results.detections.map((detection) => (
                                    <View
                                        key={detection.id}
                                        className="bg-card border-2 border-border rounded-xl p-5 mb-3"
                                    >
                                        <View className="flex-row items-start gap-3 mb-3">
                                            <View
                                                className="w-4 h-4 rounded-full mt-1"
                                                style={{
                                                    backgroundColor:
                                                        getBoundingBoxColor(
                                                            detection.diseaseClass,
                                                        ),
                                                }}
                                            />
                                            <View className="flex-1">
                                                <Text className="text-lg font-bold text-foreground">
                                                    {getDiseaseLabel(
                                                        detection.diseaseClass,
                                                    )}
                                                </Text>
                                                <Text className="text-base text-muted-foreground mt-1">
                                                    {(
                                                        detection.confidence *
                                                        100
                                                    ).toFixed(0)}
                                                    % confident
                                                </Text>
                                            </View>
                                        </View>

                                        <Button
                                            variant="outline"
                                            className="mt-2"
                                            onPress={() =>
                                                handleLearnMore(
                                                    detection.diseaseClass,
                                                )
                                            }
                                        >
                                            <Text>Learn More & Treatment</Text>
                                        </Button>
                                    </View>
                                ))}
                            </View>
                        )}
                    </View>
                ) : null}
            </ScrollView>

            <View className="absolute bottom-0 left-0 right-0 px-5 pb-8 pt-4 bg-gradient-to-t from-background to-transparent">
                <Button
                    className="w-full h-14 rounded-xl"
                    onPress={handleCheckAnother}
                >
                    <Text className="text-base font-bold">
                        Check Another Fish
                    </Text>
                </Button>
            </View>
        </View>
    )
}
