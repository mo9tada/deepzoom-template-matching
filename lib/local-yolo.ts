import { detectFloorplanObjects } from "./yolo"
import { type Detection } from "./detections"

export interface LocalYoloResult {
  detections: Detection[]
}

const mapLocalDetections = (results: Awaited<ReturnType<typeof detectFloorplanObjects>>): Detection[] => {
  return results.map((result) => ({
    label: result.name,
    confidence: result.score,
    boundingBox: {
      x: result.boundingBox.x,
      y: result.boundingBox.y,
      width: result.boundingBox.width,
      height: result.boundingBox.height,
    },
  }))
}

export const analyzeImageWithLocalYolo = async (imageBuffer: Buffer): Promise<LocalYoloResult> => {
  const yoloDetections = await detectFloorplanObjects(imageBuffer)
  return { detections: mapLocalDetections(yoloDetections) }
}
