import { type Detection } from "./detections"
import { type SelectionRect } from "./selection"

export interface SimilarityResult {
  selected: Detection | null
  iou: number
  matches: Detection[]
}

const computeIoU = (a: SelectionRect, b: SelectionRect) => {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.width, b.x + b.width)
  const y2 = Math.min(a.y + a.height, b.y + b.height)
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  if (intersection <= 0) return 0
  const areaA = a.width * a.height
  const areaB = b.width * b.height
  const union = areaA + areaB - intersection
  return union > 0 ? intersection / union : 0
}

const detectionToRect = (det: Detection): SelectionRect => det.boundingBox

export const matchSimilarDetections = (
  selection: SelectionRect | null,
  detections: Detection[],
  sizeTolerance = 0.3
): SimilarityResult => {
  if (!selection || detections.length === 0) {
    return { selected: null, iou: 0, matches: detections }
  }

  let bestMatch: Detection | null = null
  let bestIoU = 0
  for (const det of detections) {
    const iou = computeIoU(selection, detectionToRect(det))
    if (iou > bestIoU) {
      bestIoU = iou
      bestMatch = det
    }
  }

  if (!bestMatch || bestIoU < 0.1) {
    return { selected: null, iou: bestIoU, matches: detections }
  }

  const selectedWidth = bestMatch.boundingBox.width
  const selectedHeight = bestMatch.boundingBox.height

  const matches = detections.filter((det) => {
    if (det.label !== bestMatch?.label) return false
    const widthDelta = Math.abs(det.boundingBox.width - selectedWidth) / (selectedWidth || 1)
    const heightDelta = Math.abs(det.boundingBox.height - selectedHeight) / (selectedHeight || 1)
    return widthDelta <= sizeTolerance && heightDelta <= sizeTolerance
  })

  return { selected: bestMatch, iou: bestIoU, matches }
}
