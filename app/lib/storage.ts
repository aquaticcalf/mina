import { createMMKV } from "react-native-mmkv"

/**
 * Global MMKV storage instance.
 * Replaces AsyncStorage for synchronous, high-performance key-value storage.
 */
export const storage = createMMKV()
