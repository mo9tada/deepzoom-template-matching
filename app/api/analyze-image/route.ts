import path from "path"
import fs from "fs/promises"
import sharp from "sharp"
import { NextResponse } from "next/server"
import { analyzeImageWithRoboflow } from "@/lib/roboflow"
import { type Detection } from "@/lib/detections"
import { type SelectionRect, normalizeSelection } from "@/lib/selection"
import { matchSimilarDetections } from "@/lib/detection-similarity"
import { storeDetections } from "@/lib/detection-cache"

interface ParsedImagePayload {
  buffer: Buffer
  dataUrl: string
  mime: string
  extension: string
}

const MAX_ROBOFLOW_EDGE = Number(process.env.ROBOFLOW_MAX_EDGE ?? 2048)
const MAX_ROBOFLOW_BYTES = Number(process.env.ROBOFLOW_MAX_BYTES ?? 12 * 1024 * 1024)

const prepareRoboflowPayload = async (buffer: Buffer) => {
  if (!buffer) return buffer
  try {
    const metadata = await sharp(buffer).metadata()
    const width = metadata.width ?? 0
    const height = metadata.height ?? 0
    const needsResize =
      buffer.byteLength > MAX_ROBOFLOW_BYTES || width > MAX_ROBOFLOW_EDGE || height > MAX_ROBOFLOW_EDGE
    if (!needsResize) {
      return buffer
    }

    let nextEdge = MAX_ROBOFLOW_EDGE
    let quality = 90
    let bestBuffer = buffer

    for (let attempt = 0; attempt < 4; attempt++) {
      const resizedBuffer = await sharp(buffer)
        .resize({ width: nextEdge, height: nextEdge, fit: "inside" })
        .jpeg({ quality })
        .toBuffer()

      bestBuffer = resizedBuffer
      if (resizedBuffer.byteLength <= MAX_ROBOFLOW_BYTES) {
        return resizedBuffer
      }

      nextEdge = Math.max(512, Math.floor(nextEdge * 0.75))
      quality = Math.max(50, quality - 10)
    }

    return bestBuffer
  } catch (error) {
    console.warn("Roboflow payload prep failed, using original buffer", error)
    return buffer
  }
}

const parseIncomingImage = (raw: string): ParsedImagePayload => {
  if (!raw) throw new Error("No image data provided")
  const trimmed = raw.trim()
  if (trimmed.startsWith("data:image")) {
    const match = trimmed.match(/^data:(image\/[^;]+);base64,(.*)$/)
    if (!match) throw new Error("Invalid data URL provided for image")
    const mime = match[1]
    const base64 = match[2]
    const buffer = Buffer.from(base64, "base64")
    const extension = mime.split("/").pop() || "png"
    return { buffer, dataUrl: trimmed, mime, extension }
  }
  const mime = "image/png"
  const buffer = Buffer.from(trimmed, "base64")
  return {
    buffer,
    dataUrl: `data:${mime};base64,${trimmed}`,
    mime,
    extension: "png",
  }
}

const serializeError = (error: unknown) => {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack,
    }
  }
  return { message: "Unknown error", value: error }
}

const uploadsDir = path.join(process.cwd(), "uploads")

const resolveRoboflowModelName = () => {
  const modelId = process.env.ROBOFLOW_MODEL_ID
  if (!modelId) return "roboflow-model"
  const version = process.env.ROBOFLOW_MODEL_VERSION?.trim()
  return version ? `${modelId}/${version}` : modelId
}

const saveUploadedImage = async (payload: ParsedImagePayload) => {
  await fs.mkdir(uploadsDir, { recursive: true })
  const filename = `upload-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${payload.extension}`
  const filePath = path.join(uploadsDir, filename)
  await fs.writeFile(filePath, payload.buffer)
  return filePath
}

const renderAnnotations = async (imageBuffer: Buffer, detections: Detection[]) => {
  const baseImage = sharp(imageBuffer)
  const metadata = await baseImage.metadata()
  const width = metadata.width ?? 0
  const height = metadata.height ?? 0
  if (!width || !height) {
    return { buffer: imageBuffer, width: 0, height: 0, dataUrl: null as string | null }
  }

  if (detections.length === 0) {
    const raw = imageBuffer.toString("base64")
    return { buffer: imageBuffer, width, height, dataUrl: `data:image/png;base64,${raw}` }
  }

  const palette = ["#3b82f6", "#a855f7", "#f97316", "#10b981", "#ec4899", "#f59e0b"]

  const rects = detections
    .map((det, index) => {
      const color = palette[index % palette.length]
      const x = Math.round(det.boundingBox.x * width)
      const y = Math.round(det.boundingBox.y * height)
      const boxWidth = Math.round(det.boundingBox.width * width)
      const boxHeight = Math.round(det.boundingBox.height * height)
      const label = `${det.label} ${(det.confidence * 100).toFixed(1)}%`
      return `
        <g>
          <rect x="${x}" y="${y}" width="${boxWidth}" height="${boxHeight}" fill="none" stroke="${color}" stroke-width="3" />
          <rect x="${x}" y="${Math.max(0, y - 22)}" width="${Math.max(80, label.length * 7)}" height="20" rx="4" fill="${color}" opacity="0.9" />
          <text x="${x + 8}" y="${Math.max(12, y - 8)}" font-size="12" fill="#fff" font-family="Arial">${label}</text>
        </g>
      `
    })
    .join("\n")

  const svg = Buffer.from(
    `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${rects}</svg>`
  )

  const annotatedBuffer = await baseImage.composite([{ input: svg, top: 0, left: 0 }]).png().toBuffer()
  return {
    buffer: annotatedBuffer,
    width,
    height,
    dataUrl: `data:image/png;base64,${annotatedBuffer.toString("base64")}`,
  }
}

export const runtime = "nodejs"

export async function POST(request: Request) {
  try {
    const { image, coordinates } = await request.json()
    if (!image) {
      return NextResponse.json({ success: false, error: "No image provided" }, { status: 400 })
    }

    const parsedImage = parseIncomingImage(image)
    const savedPath = await saveUploadedImage(parsedImage)
    const selection = normalizeSelection(coordinates)
    const roboflowInputBuffer = await prepareRoboflowPayload(parsedImage.buffer)
    const roboflowResult = await analyzeImageWithRoboflow(roboflowInputBuffer)
    const similarity = matchSimilarDetections(selection, roboflowResult.detections)
    const detections = similarity.matches
    const detectionSetId = storeDetections(roboflowResult.detections)
    const summary: string | null = null
    const provider = "roboflow"
    const modelName = resolveRoboflowModelName()
    const annotated = await renderAnnotations(parsedImage.buffer, detections)

    return NextResponse.json({
      success: true,
      provider,
      model: modelName,
      uploadedPath: path.relative(process.cwd(), savedPath),
      annotations: detections,
      annotatedImage: annotated.dataUrl,
      detectionSetId,
      metadata: {
        width: annotated.width,
        height: annotated.height,
        summary,
        selection: {
          provided: Boolean(selection),
          normalized: selection,
          selectedLabel: similarity.selected?.label ?? null,
          similarityIoU: similarity.iou,
        },
      },
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Vision analysis failed", error)
    const responseBody = {
      success: false,
      error: "Failed to process image",
      message: error instanceof Error ? error.message : "Unknown error occurred",
      timestamp: new Date().toISOString(),
      ...(process.env.NODE_ENV === "development" ? { details: serializeError(error) } : {}),
    }

    return new Response(JSON.stringify(responseBody, null, 2), {
      status: 500,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store, max-age=0",
      },
    })
  }
}
