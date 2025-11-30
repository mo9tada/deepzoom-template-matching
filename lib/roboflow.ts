import { type Detection } from "./detections"

interface RoboflowPrediction {
  x?: number
  y?: number
  width?: number
  height?: number
  confidence: number
  class: string
  image_path?: string
  image_width?: number
  image_height?: number
  x_min?: number
  x_max?: number
  y_min?: number
  y_max?: number
  bbox?: [number, number, number, number]
}

interface RoboflowImageMeta {
  width?: number
  height?: number
  identifier?: string
}

interface WorkflowOutput {
  type?: string
  name?: string
  json?: Record<string, unknown>
}

interface RoboflowResponseBody {
  predictions?: RoboflowPrediction[]
  time?: number
  image?: RoboflowImageMeta
  error?: string
  message?: string
  outputs?: WorkflowOutput[]
}

export interface RoboflowResult {
  detections: Detection[]
  image?: RoboflowImageMeta
  raw: RoboflowResponseBody
}

const clamp01 = (value: number) => Math.min(1, Math.max(0, value))

const resolveModelPath = () => {
  const workflowId = process.env.ROBOFLOW_WORKFLOW_ID?.trim()
  if (workflowId) {
    return `workflow/${workflowId.replace(/^workflow\//, "")}`
  }
  const modelId = process.env.ROBOFLOW_MODEL_ID
  if (!modelId) {
    throw new Error("ROBOFLOW_MODEL_ID is not configured")
  }
  const version = process.env.ROBOFLOW_MODEL_VERSION?.trim()
  if (version) {
    return `${modelId.replace(/\/$/, "")}/${version}`
  }
  return modelId
}

const buildEndpoint = () => {
  const apiKey = process.env.ROBOFLOW_API_KEY
  if (!apiKey) {
    throw new Error("ROBOFLOW_API_KEY is not configured")
  }
  const baseUrl = (process.env.ROBOFLOW_API_URL ?? "https://detect.roboflow.com").replace(/\/$/, "")
  const searchParams = new URLSearchParams({ api_key: apiKey, format: "json" })
  const confidence = process.env.ROBOFLOW_CONFIDENCE
  const overlap = process.env.ROBOFLOW_OVERLAP
  if (confidence) searchParams.set("confidence", confidence)
  if (overlap) searchParams.set("overlap", overlap)
  return `${baseUrl}/${resolveModelPath()}?${searchParams.toString()}`
}

const asPredictionArray = (value: unknown): RoboflowPrediction[] | null => {
  if (!Array.isArray(value)) return null
  return value as RoboflowPrediction[]
}

const findWorkflowPredictions = (payload: RoboflowResponseBody) => {
  if (Array.isArray(payload.predictions)) {
    return { predictions: payload.predictions, image: payload.image }
  }

  if (!Array.isArray(payload.outputs)) {
    return null
  }

  const candidateKeys = ["detections", "predictions", "objects", "result", "output"]

  for (const output of payload.outputs) {
    const json = output?.json
    if (!json || typeof json !== "object") continue
    const record = json as Record<string, unknown>
    const imageFromOutput = (record.image as RoboflowImageMeta | undefined) ?? payload.image
    for (const key of candidateKeys) {
      const value = record[key]
      const direct = asPredictionArray(value)
      if (direct?.length) {
        return {
          predictions: direct,
          image: imageFromOutput,
        }
      }
      if (value && typeof value === "object") {
        const nestedRecord = value as Record<string, unknown>
        const nested = asPredictionArray(nestedRecord["detections"])
          ?? asPredictionArray(nestedRecord["predictions"])
        if (nested?.length) {
          return {
            predictions: nested,
            image: imageFromOutput,
          }
        }
      }
    }
  }

  return null
}

const deriveBox = (prediction: RoboflowPrediction) => {
  if (
    typeof prediction.x === "number" &&
    typeof prediction.y === "number" &&
    typeof prediction.width === "number" &&
    typeof prediction.height === "number"
  ) {
    return { x: prediction.x, y: prediction.y, width: prediction.width, height: prediction.height }
  }

  if (
    typeof prediction.x_min === "number" &&
    typeof prediction.x_max === "number" &&
    typeof prediction.y_min === "number" &&
    typeof prediction.y_max === "number"
  ) {
    const width = prediction.x_max - prediction.x_min
    const height = prediction.y_max - prediction.y_min
    const x = prediction.x_min + width / 2
    const y = prediction.y_min + height / 2
    return { x, y, width, height }
  }

  if (Array.isArray(prediction.bbox) && prediction.bbox.length >= 4) {
    const [xMin, yMin, boxWidth, boxHeight] = prediction.bbox
    const x = xMin + boxWidth / 2
    const y = yMin + boxHeight / 2
    return { x, y, width: boxWidth, height: boxHeight }
  }

  return null
}

const toDetection = (prediction: RoboflowPrediction, imageMeta?: RoboflowImageMeta): Detection | null => {
  let width = imageMeta?.width ?? prediction.image_width
  let height = imageMeta?.height ?? prediction.image_height
  const rawBox = deriveBox(prediction)
  if (!rawBox) {
    return null
  }
  const boxValues = [rawBox.x, rawBox.y, rawBox.width, rawBox.height]
  const looksNormalized = boxValues.every((value) => typeof value === "number" && value >= 0 && value <= 1)
  if ((!width || !height) && looksNormalized) {
    width = 1
    height = 1
  }
  if (!width || !height) {
    return null
  }
  const xMin = clamp01((rawBox.x - rawBox.width / 2) / width)
  const yMin = clamp01((rawBox.y - rawBox.height / 2) / height)
  const boxWidth = clamp01(rawBox.width / width)
  const boxHeight = clamp01(rawBox.height / height)
  return {
    label: prediction.class ?? "object",
    confidence: typeof prediction.confidence === "number" ? clamp01(prediction.confidence) : 0,
    boundingBox: {
      x: xMin,
      y: yMin,
      width: boxWidth,
      height: boxHeight,
    },
  }
}

export const analyzeImageWithRoboflow = async (imageBuffer: Buffer): Promise<RoboflowResult> => {
  const endpoint = buildEndpoint()

  const binaryBody = imageBuffer.buffer.slice(
    imageBuffer.byteOffset,
    imageBuffer.byteOffset + imageBuffer.byteLength
  ) as ArrayBuffer

  const binaryRequest = () =>
    fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: binaryBody,
    })

  const base64Request = () => {
    const body = new URLSearchParams({ image: imageBuffer.toString("base64") })
    return fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    })
  }

  let response = await binaryRequest()
  if (response.status === 405 || response.status === 415) {
    console.warn(`Roboflow binary upload rejected (${response.status}), retrying with base64 payload`)
    response = await base64Request()
  }

  if (!response.ok) {
    const text = await response.text()
    throw new Error(`Roboflow request failed (${response.status}): ${text}`)
  }

  const payload = (await response.json()) as RoboflowResponseBody
  const extracted = findWorkflowPredictions(payload)
  const predictions = extracted?.predictions ?? []
  const imageMeta = extracted?.image ?? payload.image

  const detections = predictions
    .map((prediction) => toDetection(prediction, imageMeta))
    .filter((det): det is Detection => Boolean(det))

  if (payload.error) {
    throw new Error(payload.error)
  }

  return {
    detections,
    image: imageMeta,
    raw: payload,
  }
}
