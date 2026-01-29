import type { ExpoConfig, ConfigContext } from "expo/config"

export default ({ config }: ConfigContext): ExpoConfig => ({
    ...config,
    name: "mina",
    slug: "mina",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/images/icon.png",
    scheme: "mina",
    userInterfaceStyle: "automatic",
    newArchEnabled: true,
    splash: {
        image: "./assets/images/icon.png",
        resizeMode: "contain",
        backgroundColor: "#673ab7",
    },
    assetBundlePatterns: ["**/*"],
    ios: {
        supportsTablet: true,
        bundleIdentifier: "com.mina.app",
        infoPlist: {
            NSCameraUsageDescription:
                "This app uses the camera to detect and identify fish species.",
            NSMicrophoneUsageDescription:
                "This app may use the microphone for video recording.",
        },
    },
    android: {
        edgeToEdgeEnabled: true,
        adaptiveIcon: {
            foregroundImage: "./assets/images/adaptive-icon.png",
            backgroundColor: "#673ab7",
        },
        package: "com.mina.app",
        permissions: ["CAMERA", "RECORD_AUDIO"],
    },
    web: {
        bundler: "metro",
        output: "static",
        favicon: "./assets/images/favicon.png",
    },
    plugins: [
        "expo-router",
        [
            "expo-camera",
            {
                cameraPermission:
                    "Allow mina to access your camera to detect and identify fish species.",
            },
        ],
    ],
    experiments: {
        typedRoutes: true,
    },
    extra: {
        // Model channel: "dev" or "prod"
        // Set via EXPO_PUBLIC_MODEL_CHANNEL env var at build time
        modelChannel: process.env.EXPO_PUBLIC_MODEL_CHANNEL ?? "prod",
        eas: {
            projectId: "4ee8677c-b53f-405f-ae45-afb23a3502f8",
        },
    },
})
