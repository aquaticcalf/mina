import { useLocalSearchParams, useRouter } from "expo-router"
import React, { useEffect, useState, useRef } from "react"
import { ActivityIndicator, Image, ScrollView, View } from "react-native"
import { useNavigation } from "@react-navigation/native"
import { Skia, matchFont, ImageFormat } from "@shopify/react-native-skia"
import * as FileSystem from "expo-file-system/legacy"
import { Button } from "@/components/ui/button"
import { Text } from "@/components/ui/text"
import { useModel } from "@/lib/model"
import type { InferenceResult } from "@/lib/model/inference"
import { saveHistoryItem, getHistoryItem } from "@/lib/history/storage"
import { getBoundingBoxColor, getDiseaseLabel } from "@/lib/model/disease/info"

export default function ResultsScreen() {
    const router = useRouter()
    const navigation = useNavigation()
    const { imageUri, historyId } = useLocalSearchParams<{
        imageUri?: string
        historyId?: string
    }>()
    const { runInference, isReady } = useModel()

    const [isAnalyzing, setIsAnalyzing] = useState(true)
    const [results, setResults] = useState<InferenceResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [processedImageUri, setProcessedImageUri] = useState<string | null>(
        null,
    )
    const savedRef = useRef(false)

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

    useEffect(() => {
        async function load() {
            // Case 1: Load existing history item
            if (historyId) {
                try {
                    setIsAnalyzing(true)
                    const item = await getHistoryItem(historyId)
                    if (item) {
                        setResults(item.results)
                        setProcessedImageUri(item.processedImageUri)
                        // We don't need to re-analyze or save
                    } else {
                        setError("History item not found")
                    }
                } catch (e) {
                    setError("Failed to load history item")
                } finally {
                    setIsAnalyzing(false)
                }
                return
            }

            // Case 2: New Analysis
            if (!imageUri) {
                setError("No image provided")
                setIsAnalyzing(false)
                return
            }

            if (!isReady) {
                setError("Model not ready")
                setIsAnalyzing(false)
                return
            }

            try {
                setIsAnalyzing(true)
                setError(null)

                // 1. Run Inference
                const result = await runInference(imageUri)

                if (!result) {
                    setError("Inference failed - no results returned")
                    setIsAnalyzing(false)
                    return
                }

                setResults(result)

                let finalProcessedUri = imageUri

                // 2. Process Image with Skia (Overlay Bounding Boxes)
                if (result.detections.length > 0) {
                    try {
                        const processedUri = await processImageWithOverlays(
                            imageUri,
                            result,
                        )
                        setProcessedImageUri(processedUri)
                        finalProcessedUri = processedUri
                    } catch (e) {
                        console.error("Failed to process image overlays:", e)
                        // Fallback to original image if overlay processing fails
                        setProcessedImageUri(imageUri)
                    }
                } else {
                    setProcessedImageUri(imageUri)
                }

                // 3. Save to History (if not already saved in this session)
                if (!savedRef.current) {
                    savedRef.current = true
                    try {
                        await saveHistoryItem({
                            timestamp: Date.now(),
                            originalImageUri: imageUri,
                            processedImageUri: finalProcessedUri,
                            results: result,
                        })
                    } catch (e) {
                        console.error("Failed to save history:", e)
                    }
                }
            } catch (err) {
                const message = err instanceof Error ? err.message : String(err)
                setError(`Analysis failed: ${message}`)
            } finally {
                setIsAnalyzing(false)
            }
        }

        load()
    }, [imageUri, historyId, isReady, runInference])

    // Helper to draw overlays and save image
    const processImageWithOverlays = async (
        uri: string,
        result: InferenceResult,
    ): Promise<string> => {
        // Read original image
        const base64 = await FileSystem.readAsStringAsync(uri, {
            encoding: "base64",
        })
        const imageData = Skia.Data.fromBase64(base64)
        const image = Skia.Image.MakeImageFromEncoded(imageData)

        if (!image)
            throw new Error("Could not decode image for overlay processing")

        const width = image.width()
        const height = image.height()

        // Create Surface/Canvas
        const surface = Skia.Surface.Make(width, height)
        if (!surface) throw new Error("Could not create Skia surface")

        const canvas = surface.getCanvas()

        // Draw original image
        canvas.drawImage(image, 0, 0)

        // Setup Paint for Bounding Boxes
        const paint = Skia.Paint()
        paint.setStyle(1) // Stroke
        paint.setStrokeWidth(4 * (width / 640)) // Scale stroke width relative to image size
        paint.setAntiAlias(true)

        // Setup Font for Labels
        const fontStyle = {
            fontFamily: "sans-serif",
            fontSize: 24 * (width / 640), // Scale font size
            fontWeight: "bold" as const,
        }
        const font = matchFont(fontStyle)

        const textPaint = Skia.Paint()
        textPaint.setColor(Skia.Color("white"))

        const bgPaint = Skia.Paint()
        bgPaint.setStyle(0) // Fill
        bgPaint.setAlphaf(0.7) // Semi-transparent

        // Draw Detections
        for (const detection of result.detections) {
            const { boundingBox: bbox, diseaseClass, confidence } = detection

            // Convert normalized coordinates to pixel coordinates
            const x = bbox.x * width
            const y = bbox.y * height
            const w = bbox.width * width
            const h = bbox.height * height

            // Get color based on disease class
            const colorHex = getBoundingBoxColor(diseaseClass)
            paint.setColor(Skia.Color(colorHex))
            bgPaint.setColor(Skia.Color(colorHex))
            bgPaint.setAlphaf(0.7) // Reset alpha as setColor might reset it? (Safety)

            // Draw Box
            canvas.drawRect({ x, y, width: w, height: h }, paint)

            // Draw Label
            const label = `${getDiseaseLabel(diseaseClass)} ${(confidence * 100).toFixed(0)}%`
            const textWidth = font.getTextWidth(label)
            const textHeight = font.getSize()

            // Draw Label Background
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

            // Draw Text
            canvas.drawText(
                label,
                x + labelBgPadding,
                y - labelBgPadding,
                textPaint,
                font,
            )
        }

        // Save Result
        const snapshot = surface.makeImageSnapshot()
        // Use ImageFormat.PNG directly (imported from Skia)
        const resultBase64 = snapshot.encodeToBase64(ImageFormat.PNG, 100)

        const filename = `processed_${Date.now()}.png`
        const resultUri = `${FileSystem.cacheDirectory}${filename}`

        await FileSystem.writeAsStringAsync(resultUri, resultBase64, {
            encoding: "base64",
        })

        return resultUri
    }

    return (
        <View className="flex-1 bg-black">
            <View className="absolute top-0 left-0 right-0 z-10 flex-row justify-center items-center pt-12 px-4">
                <Text className="text-white font-medium">Analysis Results</Text>
            </View>

            <ScrollView className="flex-1 pt-20">
                {(processedImageUri || imageUri) && (
                    <View className="items-center justify-center px-4 mb-6">
                        <Image
                            source={{ uri: processedImageUri || imageUri }}
                            className="w-full h-64 rounded-lg"
                            resizeMode="contain"
                        />
                    </View>
                )}

                {isAnalyzing ? (
                    <View className="items-center justify-center py-12">
                        <ActivityIndicator size="large" color="#ffffff" />
                        <Text className="text-white mt-4">
                            Analyzing fish...
                        </Text>
                    </View>
                ) : error ? (
                    <View className="items-center justify-center px-6 py-12">
                        <Text className="text-red-500 text-center text-lg mb-2">
                            Error
                        </Text>
                        <Text className="text-white text-center">{error}</Text>
                    </View>
                ) : results ? (
                    <View className="px-4">
                        <View className="bg-gray-900 rounded-lg p-4 mb-4">
                            <Text className="text-white text-sm mb-2">
                                Inference Time:{" "}
                                <Text className="text-gray-400">
                                    {results.inferenceTimeMs}ms
                                </Text>
                            </Text>
                            <Text className="text-white text-sm">
                                Detections Found:{" "}
                                <Text className="text-gray-400">
                                    {results.detections.length}
                                </Text>
                            </Text>
                        </View>

                        {results.detections.length === 0 ? (
                            <View className="bg-green-900/30 border border-green-500/30 rounded-lg p-6 items-center">
                                <Text className="text-green-400 text-lg font-semibold mb-2">
                                    Fish Appears Healthy
                                </Text>
                                <Text className="text-gray-300 text-center">
                                    No diseases detected in the image
                                </Text>
                            </View>
                        ) : (
                            <View>
                                <Text className="text-white text-lg font-semibold mb-3">
                                    Detected Issues
                                </Text>
                                {results.detections.map((detection, index) => (
                                    <View
                                        key={detection.id}
                                        className="bg-gray-900 rounded-lg p-4 mb-3"
                                    >
                                        <View className="flex-row justify-between items-start mb-2">
                                            <Text
                                                className="text-lg font-semibold"
                                                style={{
                                                    color: getBoundingBoxColor(
                                                        detection.diseaseClass,
                                                    ),
                                                }}
                                            >
                                                {getDiseaseLabel(
                                                    detection.diseaseClass,
                                                )}
                                            </Text>
                                            <Text className="text-white text-lg font-bold">
                                                {(
                                                    detection.confidence * 100
                                                ).toFixed(1)}
                                                %
                                            </Text>
                                        </View>
                                        <Text className="text-gray-400 text-xs">
                                            Detection ID: {detection.id}
                                        </Text>
                                        <Text className="text-gray-400 text-xs mt-1">
                                            Location: (
                                            {detection.boundingBox.x.toFixed(3)}
                                            ,{" "}
                                            {detection.boundingBox.y.toFixed(3)}
                                            ) | Size:{" "}
                                            {detection.boundingBox.width.toFixed(
                                                3,
                                            )}{" "}
                                            x{" "}
                                            {detection.boundingBox.height.toFixed(
                                                3,
                                            )}
                                        </Text>
                                    </View>
                                ))}
                            </View>
                        )}
                    </View>
                ) : null}
            </ScrollView>

            <View className="absolute bottom-0 left-0 right-0 flex-row gap-3 px-4 pb-8">
                <Button
                    variant="outline"
                    className="flex-1 h-12"
                    onPress={() => router.push("/")}
                    disabled={isAnalyzing}
                >
                    <Text>Done</Text>
                </Button>
                {results && results.detections.length > 0 && (
                    <Button
                        className="flex-1 h-12"
                        onPress={() => {
                            // Navigate to disease info for the top detection
                            const topDetection = results.detections[0]
                            router.push({
                                pathname: "/disease/[id]",
                                params: {
                                    id: topDetection.diseaseClass,
                                },
                            })
                        }}
                        disabled={isAnalyzing}
                    >
                        <Text>View Details</Text>
                    </Button>
                )}
            </View>
        </View>
    )
}
