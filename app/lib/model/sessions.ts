/**
 * Storage service for detection sessions.
 * Handles persisting and retrieving detection history using AsyncStorage.
 */

import AsyncStorage from "@react-native-async-storage/async-storage"
import type { DetectionSession } from "@/lib/model/types"
import { parseSession, serializeSession } from "@/lib/model/types"

const STORAGE_KEY = "mina_detection_sessions"

async function getAllSessionJsons(): Promise<string[]> {
    const data = await AsyncStorage.getItem(STORAGE_KEY)
    if (!data) return []
    try {
        const parsed = JSON.parse(data)
        if (Array.isArray(parsed)) {
            return parsed.filter(
                (item): item is string => typeof item === "string",
            )
        }
        return []
    } catch {
        return []
    }
}

async function saveAllSessionJsons(sessionJsons: string[]): Promise<void> {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(sessionJsons))
}

export async function saveSession(session: DetectionSession): Promise<void> {
    const sessionJson = serializeSession(session)
    const allSessions = await getAllSessionJsons()
    const existingIndex = allSessions.findIndex((json) => {
        const parsed = parseSession(json)
        return parsed?.id === session.id
    })
    if (existingIndex >= 0) {
        allSessions[existingIndex] = sessionJson
    } else {
        allSessions.push(sessionJson)
    }
    await saveAllSessionJsons(allSessions)
}

export async function getSessions(): Promise<DetectionSession[]> {
    const allSessions = await getAllSessionJsons()
    const sessions: DetectionSession[] = []
    for (const json of allSessions) {
        const session = parseSession(json)
        if (session) {
            sessions.push(session)
        }
    }
    sessions.sort((a, b) => b.timestamp - a.timestamp)
    return sessions
}

export async function getSession(id: string): Promise<DetectionSession | null> {
    const allSessions = await getAllSessionJsons()
    for (const json of allSessions) {
        const session = parseSession(json)
        if (session?.id === id) {
            return session
        }
    }
    return null
}

export async function deleteSession(id: string): Promise<void> {
    const allSessions = await getAllSessionJsons()
    const filtered = allSessions.filter((json) => {
        const session = parseSession(json)
        return session?.id !== id
    })
    await saveAllSessionJsons(filtered)
}

export async function clearAll(): Promise<void> {
    await AsyncStorage.removeItem(STORAGE_KEY)
}
