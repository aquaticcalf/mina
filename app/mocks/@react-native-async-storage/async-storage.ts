const mockStorage: Record<string, string> = {}

const AsyncStorage = {
    getItem: async (key: string): Promise<string | null> => {
        return mockStorage[key] ?? null
    },
    setItem: async (key: string, value: string): Promise<void> => {
        mockStorage[key] = value
    },
    removeItem: async (key: string): Promise<void> => {
        delete mockStorage[key]
    },
    clear: async (): Promise<void> => {
        for (const key of Object.keys(mockStorage)) {
            delete mockStorage[key]
        }
    },
    getAllKeys: async (): Promise<string[]> => {
        return Object.keys(mockStorage)
    },
    multiGet: async (keys: string[]): Promise<[string, string | null][]> => {
        return keys.map((key) => [key, mockStorage[key] ?? null])
    },
    multiSet: async (keyValuePairs: [string, string][]): Promise<void> => {
        for (const [key, value] of keyValuePairs) {
            mockStorage[key] = value
        }
    },
    multiRemove: async (keys: string[]): Promise<void> => {
        for (const key of keys) {
            delete mockStorage[key]
        }
    },
}

export default AsyncStorage
