import { describe, it, expect, beforeEach, afterEach, mock } from "bun:test"
import type { DetectionSession } from "@/lib/model/types"

const mockStorage: Record<string, string> = {}

const mockAsyncStorage = {
    getItem: mock(async (key: string): Promise<string | null> => {
        return mockStorage[key] ?? null
    }),
    setItem: mock(async (key: string, value: string): Promise<void> => {
        mockStorage[key] = value
    }),
    removeItem: mock(async (key: string): Promise<void> => {
        delete mockStorage[key]
    }),
    clear: mock(async (): Promise<void> => {
        for (const key of Object.keys(mockStorage)) {
            delete mockStorage[key]
        }
    }),
}

mock.module("@react-native-async-storage/async-storage", () => ({
    default: mockAsyncStorage,
}))

const { clearAll, deleteSession, getSession, getSessions, saveSession } =
    await import("@/lib/model/sessions")

function createTestSession(id: string, timestamp: number): DetectionSession {
    return {
        id,
        imageUri: `file:///test/${id}.jpg`,
        detections: [
            {
                id: `det_${id}`,
                diseaseClass: "bacterial_infection",
                confidence: 0.85,
                boundingBox: { x: 0.1, y: 0.1, width: 0.3, height: 0.3 },
            },
        ],
        timestamp,
    }
}

describe("**Feature: fish-disease-detection, StorageService**", () => {
    beforeEach(() => {
        for (const key of Object.keys(mockStorage)) {
            delete mockStorage[key]
        }
        mockAsyncStorage.getItem.mockClear()
        mockAsyncStorage.setItem.mockClear()
        mockAsyncStorage.removeItem.mockClear()
        mockAsyncStorage.clear.mockClear()
    })

    describe("saveSession", () => {
        it("should save a session to storage", async () => {
            const session = createTestSession("session_1", Date.now())
            await saveSession(session)
            const retrieved = await getSession("session_1")
            expect(retrieved).not.toBeNull()
            expect(retrieved?.id).toBe("session_1")
        })

        it("should update an existing session with same id", async () => {
            const session1 = createTestSession("session_update", 1000)
            await saveSession(session1)

            const session2 = createTestSession("session_update", 2000)
            session2.detections[0].confidence = 0.92
            await saveSession(session2)

            const retrieved = await getSession("session_update")
            expect(retrieved).not.toBeNull()
            expect(retrieved?.timestamp).toBe(2000)
            expect(retrieved?.detections[0].confidence).toBe(0.92)
        })
    })

    describe("getSession", () => {
        it("should return null for non-existent session", async () => {
            const result = await getSession("non_existent")
            expect(result).toBeNull()
        })

        it("should retrieve exact session that was saved", async () => {
            const session = createTestSession("exact_test", 12345)
            session.detections = [
                {
                    id: "det_exact",
                    diseaseClass: "healthy",
                    confidence: 0.99,
                    boundingBox: { x: 0.5, y: 0.5, width: 0.2, height: 0.2 },
                },
            ]
            await saveSession(session)
            const retrieved = await getSession("exact_test")
            expect(retrieved).not.toBeNull()
            expect(retrieved?.id).toBe(session.id)
            expect(retrieved?.imageUri).toBe(session.imageUri)
            expect(retrieved?.timestamp).toBe(session.timestamp)
            expect(retrieved?.detections).toHaveLength(1)
            expect(retrieved?.detections[0].diseaseClass).toBe("healthy")
        })
    })

    describe("deleteSession", () => {
        it("should remove a session from storage", async () => {
            const session = createTestSession("delete_me", Date.now())
            await saveSession(session)
            expect(await getSession("delete_me")).not.toBeNull()

            await deleteSession("delete_me")
            expect(await getSession("delete_me")).toBeNull()
        })

        it("should not affect other sessions", async () => {
            await saveSession(createTestSession("keep_1", 1000))
            await saveSession(createTestSession("keep_2", 2000))
            await saveSession(createTestSession("delete_me", 1500))

            await deleteSession("delete_me")

            const sessions = await getSessions()
            expect(sessions).toHaveLength(2)
            const ids = sessions.map((s) => s.id).sort()
            expect(ids).toEqual(["keep_1", "keep_2"])
        })

        it("should handle deleting non-existent session gracefully", async () => {
            await deleteSession("non_existent")
            const sessions = await getSessions()
            expect(sessions).toHaveLength(0)
        })
    })

    describe("clearAll", () => {
        it("should remove all sessions from storage", async () => {
            await saveSession(createTestSession("test1", 1000))
            await saveSession(createTestSession("test2", 2000))
            await saveSession(createTestSession("test3", 3000))

            await clearAll()

            const sessions = await getSessions()
            expect(sessions).toHaveLength(0)
        })
    })

    describe("**Feature: fish-disease-detection, Property 4: History sorting by timestamp**", () => {
        describe("getSessions sorting", () => {
            it("should return sessions sorted by timestamp descending", async () => {
                await saveSession(createTestSession("oldest", 1000))
                await saveSession(createTestSession("middle", 2000))
                await saveSession(createTestSession("newest", 3000))

                const sessions = await getSessions()

                expect(sessions).toHaveLength(3)
                expect(sessions[0].id).toBe("newest")
                expect(sessions[1].id).toBe("middle")
                expect(sessions[2].id).toBe("oldest")
            })

            it("should handle unsaved timestamps correctly", async () => {
                const timestamps = [5000, 1000, 3000, 4000, 2000]
                for (const ts of timestamps) {
                    await saveSession(createTestSession(`ts_${ts}`, ts))
                }

                const sessions = await getSessions()

                expect(sessions).toHaveLength(5)
                for (let i = 0; i < sessions.length - 1; i++) {
                    expect(sessions[i].timestamp).toBeGreaterThanOrEqual(
                        sessions[i + 1].timestamp,
                    )
                }
            })

            it("should return empty array when no sessions exist", async () => {
                const sessions = await getSessions()
                expect(sessions).toEqual([])
            })

            it("should maintain sort order after deletions", async () => {
                await saveSession(createTestSession("first", 1000))
                await saveSession(createTestSession("second", 2000))
                await saveSession(createTestSession("third", 3000))
                await saveSession(createTestSession("fourth", 4000))

                await deleteSession("fourth")

                const sessions = await getSessions()
                expect(sessions).toHaveLength(3)
                expect(sessions[0].id).toBe("third")
                expect(sessions[2].id).toBe("first")
            })
        })
    })
})
