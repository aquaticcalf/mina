import { MMKV } from "react-native-mmkv"

/**
 * Global MMKV storage instance.
 * Replaces AsyncStorage for synchronous, high-performance key-value storage.
 */
// @ts-ignore
export const storage = new MMKV()
