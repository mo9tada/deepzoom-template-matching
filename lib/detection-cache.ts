import { randomUUID } from "crypto"
import { type Detection } from "./detections"

const TTL_MS = Number(process.env.DETECTION_CACHE_TTL ?? 10 * 60 * 1000)

interface CacheEntry {
  detections: Detection[]
  createdAt: number
}

const cache = new Map<string, CacheEntry>()

const prune = () => {
  const now = Date.now()
  for (const [key, entry] of cache.entries()) {
    if (now - entry.createdAt > TTL_MS) {
      cache.delete(key)
    }
  }
}

export const storeDetections = (detections: Detection[]) => {
  prune()
  const id = randomUUID()
  cache.set(id, { detections, createdAt: Date.now() })
  return id
}

export const getDetections = (id: string) => {
  prune()
  const entry = cache.get(id)
  if (!entry) return null
  if (Date.now() - entry.createdAt > TTL_MS) {
    cache.delete(id)
    return null
  }
  return entry.detections
}

export const clearDetections = (id: string) => {
  cache.delete(id)
}
