import { afterAll, describe, expect, it, beforeEach } from "bun:test"
import fakeIndexedDB, {
  IDBCursor,
  IDBCursorWithValue,
  IDBDatabase,
  IDBFactory,
  IDBIndex,
  IDBKeyRange,
  IDBObjectStore,
  IDBOpenDBRequest,
  IDBRequest,
  IDBTransaction,
  IDBVersionChangeEvent,
} from "fake-indexeddb"

globalThis.indexedDB = fakeIndexedDB
globalThis.IDBCursor = IDBCursor
globalThis.IDBCursorWithValue = IDBCursorWithValue
globalThis.IDBDatabase = IDBDatabase
globalThis.IDBFactory = IDBFactory
globalThis.IDBIndex = IDBIndex
globalThis.IDBKeyRange = IDBKeyRange
globalThis.IDBObjectStore = IDBObjectStore
globalThis.IDBOpenDBRequest = IDBOpenDBRequest
globalThis.IDBRequest = IDBRequest
globalThis.IDBTransaction = IDBTransaction
globalThis.IDBVersionChangeEvent = IDBVersionChangeEvent

import type { HistoryItemInput } from "@/lib/history/types"
import {
  clearHistory,
  closeHistoryDB,
  deleteHistoryItem,
  getHistoryItem,
  getHistoryItems,
  initHistoryDB,
  saveHistoryItem,
} from "@/lib/history/storage"

async function createTestInput(timestamp: number): Promise<HistoryItemInput> {
  return {
    timestamp,
    originalImage: new Blob([new Uint8Array([1, 2, 3, 4, 5])], {
      type: "image/png",
    }),
    processedImage: new Blob([new Uint8Array([6, 7, 8, 9, 10])], {
      type: "image/png",
    }),
    results: {
      detections: [
        {
          id: "det-1",
          diseaseClass: "bacterial_infection",
          confidence: 0.85,
          boundingBox: { x: 0.1, y: 0.1, width: 0.3, height: 0.3 },
        },
      ],
      inferenceTimeMs: 180,
    },
  }
}

describe("history storage", () => {
  beforeEach(async () => {
    await initHistoryDB()
    try {
      await clearHistory()
    } catch {
      // ignore
    }
  })

  afterAll(() => {
    closeHistoryDB()
  })

  it("should initialize database", async () => {
    await initHistoryDB()
  })

  it("should save and retrieve a history item", async () => {
    const input = await createTestInput(Date.now())
    const saved = await saveHistoryItem(input)

    expect(saved.id).toBeDefined()
    expect(typeof saved.id).toBe("string")
    expect(saved.timestamp).toBe(input.timestamp)
    expect(saved.results).toEqual(input.results)

    const items = await getHistoryItems()
    expect(items).toHaveLength(1)
    expect(items[0].id).toBe(saved.id)
  })

  it("should save multiple items", async () => {
    await saveHistoryItem(await createTestInput(1000))
    await saveHistoryItem(await createTestInput(2000))
    await saveHistoryItem(await createTestInput(3000))

    const items = await getHistoryItems()
    expect(items).toHaveLength(3)
  })

  it("should return empty array when no items", async () => {
    const items = await getHistoryItems()
    expect(items).toEqual([])
  })

  it("should sort by timestamp descending", async () => {
    await saveHistoryItem(await createTestInput(1000))
    await saveHistoryItem(await createTestInput(3000))
    await saveHistoryItem(await createTestInput(2000))

    const items = await getHistoryItems()

    expect(items).toHaveLength(3)
    expect(items[0].timestamp).toBe(3000)
    expect(items[1].timestamp).toBe(2000)
    expect(items[2].timestamp).toBe(1000)
  })

  it("should handle random order insertion", async () => {
    for (const ts of [5000, 1000, 3000, 4000, 2000]) {
      await saveHistoryItem(await createTestInput(ts))
    }

    const items = await getHistoryItems()
    expect(items).toHaveLength(5)
    for (let i = 0; i < items.length - 1; i++) {
      expect(items[i].timestamp).toBeGreaterThanOrEqual(items[i + 1].timestamp)
    }
  })

  it("should return null for non-existent item", async () => {
    const result = await getHistoryItem("non_existent")
    expect(result).toBeNull()
  })

  it("should retrieve exact saved item", async () => {
    const input = await createTestInput(12345)
    const saved = await saveHistoryItem(input)

    const retrieved = await getHistoryItem(saved.id)

    expect(retrieved).not.toBeNull()
    expect(retrieved?.id).toBe(saved.id)
    expect(retrieved?.timestamp).toBe(12345)
  })

  it("should delete an item", async () => {
    const saved = await saveHistoryItem(await createTestInput(Date.now()))
    expect(await getHistoryItem(saved.id)).not.toBeNull()

    await deleteHistoryItem(saved.id)

    expect(await getHistoryItem(saved.id)).toBeNull()
  })

  it("should not affect other items on delete", async () => {
    const item1 = await saveHistoryItem(await createTestInput(1000))
    const item2 = await saveHistoryItem(await createTestInput(2000))
    const item3 = await saveHistoryItem(await createTestInput(3000))

    await deleteHistoryItem(item2.id)

    const items = await getHistoryItems()
    expect(items).toHaveLength(2)

    const ids = items.map((i) => i.id).sort()
    expect(ids).toEqual([item1.id, item3.id].sort())
  })

  it("should clear all items", async () => {
    await saveHistoryItem(await createTestInput(1000))
    await saveHistoryItem(await createTestInput(2000))
    await saveHistoryItem(await createTestInput(3000))

    let items = await getHistoryItems()
    expect(items).toHaveLength(3)

    await clearHistory()

    items = await getHistoryItems()
    expect(items).toHaveLength(0)
  })
})
