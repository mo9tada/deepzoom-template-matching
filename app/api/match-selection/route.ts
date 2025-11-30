import { NextResponse } from "next/server"
import { normalizeSelection, type SelectionRect } from "@/lib/selection"

interface TemplateMatchRequestBody {
  image: string
  coordinates?: {
    x?: number
    y?: number
    width?: number
    height?: number
  }
  options?: TemplateMatchOptionsPayload
}

interface TemplateMatchOptionsPayload {
  mode?: "auto" | "linework" | "texture"
  minSimilarity?: number
  searchPadding?: number
  label?: string
  scaleSteps?: number[]
}

interface ParsedImagePayload {
  dataUrl: string
  mime: string
}

interface MatcherDetection {
  label: string
  confidence: number
  boundingBox: SelectionRect
  templateSimilarity?: number
  templatePass?: string
}

interface MatcherStats {
  pass: string
  candidates: number
  kept: number
  durationMs: number
  bestSimilarity: number | null
}

interface MatcherResult {
  matches: MatcherDetection[]
  stats: MatcherStats[]
  bestSimilarity: number | null
}

const PYTHON_MATCHER_BASE_URL = process.env.PY_MATCHER_URL ?? "http://127.0.0.1:8000"
const PYTHON_MATCHER_TIMEOUT_MS = Number(process.env.PY_MATCHER_TIMEOUT_MS ?? 60000)

const PYTHON_MATCHER_ENDPOINT = (() => {
  try {
    return new URL("/match-selection", PYTHON_MATCHER_BASE_URL).toString()
  } catch (error) {
    console.warn("Invalid PY_MATCHER_URL, disabling matcher endpoint", error)
    return null
  }
})()

const parseIncomingImage = (raw: string): ParsedImagePayload => {
  if (!raw) throw new Error("No image provided for template match")
  const trimmed = raw.trim()
  if (trimmed.startsWith("data:image")) {
    const match = trimmed.match(/^data:(image\/[^;]+);base64,(.*)$/)
    if (!match) throw new Error("Invalid image data URL")
    const mime = match[1]
    return { dataUrl: trimmed, mime }
  }
  const mime = "image/png"
  return {
    dataUrl: `data:${mime};base64,${trimmed}`,
    mime,
  }
}

const serializeError = (error: unknown) => {
  if (error instanceof Error) {
    return { name: error.name, message: error.message, stack: error.stack }
  }
  return { message: "Unknown error", value: error }
}

type PythonMatcherOptions = {
  mode?: string
  min_similarity?: number
  search_padding?: number
  label?: string
  scales?: number[]
}

const serializePythonOptions = (options?: TemplateMatchOptionsPayload): PythonMatcherOptions | undefined => {
  if (!options) return undefined
  const payload: PythonMatcherOptions = {}
  if (options.mode) payload.mode = options.mode
  if (typeof options.minSimilarity === "number") payload.min_similarity = options.minSimilarity
  if (typeof options.searchPadding === "number") payload.search_padding = options.searchPadding
  if (options.label) payload.label = options.label
  if (Array.isArray(options.scaleSteps) && options.scaleSteps.length) payload.scales = options.scaleSteps
  return Object.keys(payload).length ? payload : undefined
}

const callPythonMatcher = async (
  imageDataUrl: string,
  selection: SelectionRect,
  options?: TemplateMatchOptionsPayload
): Promise<MatcherResult> => {
  if (!PYTHON_MATCHER_ENDPOINT) {
    throw new Error("Python matcher endpoint is not configured")
  }
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), PYTHON_MATCHER_TIMEOUT_MS)
  try {
    const pythonOptions = serializePythonOptions(options)
    const response = await fetch(PYTHON_MATCHER_ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        image: imageDataUrl,
        coordinates: selection,
        ...(pythonOptions ? { options: pythonOptions } : {}),
      }),
      signal: controller.signal,
    })
    if (!response.ok) {
      const detail = await response.text()
      throw new Error(`Python matcher responded with ${response.status}: ${detail}`)
    }
    return (await response.json()) as MatcherResult
  } finally {
    clearTimeout(timeout)
  }
}

export const runtime = "nodejs"

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as TemplateMatchRequestBody
    if (!body?.image) {
      return NextResponse.json({ success: false, error: "No image provided" }, { status: 400 })
    }

    const selection = normalizeSelection(body.coordinates)
    if (!selection) {
      return NextResponse.json({ success: false, error: "Selection coordinates required" }, { status: 400 })
    }

    const parsed = parseIncomingImage(body.image)
    if (!PYTHON_MATCHER_ENDPOINT) {
      throw new Error("Python matcher endpoint is not configured")
    }

    const result = await callPythonMatcher(parsed.dataUrl, selection, body.options)
    const matcherBackend: "python" = "python"

    return NextResponse.json({
      success: true,
      matches: result.matches,
      metadata: {
        selection,
        totalMatches: result.matches.length,
        passStats: result.stats ?? [],
        bestSimilarity: result.bestSimilarity,
        options: body.options ?? null,
        mime: parsed.mime,
        matcherBackend,
      },
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Template match request failed", error)
    return NextResponse.json(
      {
        success: false,
        error: "Failed to match selection",
        ...(process.env.NODE_ENV === "development" ? { details: serializeError(error) } : {}),
      },
      { status: 500 }
    )
  }
}
