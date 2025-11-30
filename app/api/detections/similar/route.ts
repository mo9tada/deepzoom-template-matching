import { NextResponse } from "next/server"
import { normalizeSelection } from "@/lib/selection"
import { matchSimilarDetections } from "@/lib/detection-similarity"
import { getDetections } from "@/lib/detection-cache"

export const runtime = "nodejs"

export async function POST(request: Request) {
  try {
    const { detectionSetId, coordinates, sizeTolerance } = await request.json()
    if (!detectionSetId) {
      return NextResponse.json({ success: false, error: "Missing detectionSetId" }, { status: 400 })
    }

    const detections = getDetections(detectionSetId)
    if (!detections) {
      return NextResponse.json({ success: false, error: "Detection set expired or not found" }, { status: 404 })
    }

    const selection = normalizeSelection(coordinates)
    const similarity = matchSimilarDetections(selection, detections, sizeTolerance)

    return NextResponse.json({
      success: true,
      detectionSetId,
      matches: similarity.matches,
      metadata: {
        selection: {
          provided: Boolean(selection),
          normalized: selection,
          selectedLabel: similarity.selected?.label ?? null,
          similarityIoU: similarity.iou,
        },
        totalDetections: detections.length,
      },
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("Similarity lookup failed", error)
    return NextResponse.json({ success: false, error: "Failed to resolve similar detections" }, { status: 500 })
  }
}
