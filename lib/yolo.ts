import fs from "fs"
import path from "path"
import sharp from "sharp"
import * as ort from "onnxruntime-node"

export interface NormalizedBoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface YoloDetectionResult {
  name: string
  score: number
  boundingBox: NormalizedBoundingBox
}

interface PreprocessMetadata {
  originalWidth: number
  originalHeight: number
  scale: number
  padX: number
  padY: number
  inputSize: number
}

interface DetectionCandidate {
  label: string
  score: number
  box: { x1: number; y1: number; x2: number; y2: number }
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value))

const DEFAULT_MODEL_RELATIVE_PATH = "models/floorplan-yolov8n.onnx"
const DEFAULT_LABELS = [
  "toilet",
]
const DEFAULT_CLASS_COUNT = DEFAULT_LABELS.length

const normalizeLabels = (labels: string[], targetLength: number) => {
  const trimmed = labels.slice(0, targetLength)
  while (trimmed.length < targetLength) {
    trimmed.push(`class-${trimmed.length}`)
  }
  return trimmed
}

const resolvedClassCount = Number(process.env.FLOORPLAN_YOLO_CLASS_COUNT ?? DEFAULT_CLASS_COUNT)
const classCount = Number.isFinite(resolvedClassCount) && resolvedClassCount > 0 ? resolvedClassCount : DEFAULT_CLASS_COUNT
const initialLabels = process.env.FLOORPLAN_YOLO_LABELS
  ? process.env.FLOORPLAN_YOLO_LABELS.split(",").map((label) => label.trim()).filter(Boolean)
  : [...DEFAULT_LABELS]
const normalizedLabels = normalizeLabels(initialLabels, classCount)

const yoloConfig = {
  modelPath: process.env.FLOORPLAN_YOLO_MODEL ?? DEFAULT_MODEL_RELATIVE_PATH,
  inputSize: Number(process.env.FLOORPLAN_YOLO_SIZE ?? 640),
  confidenceThreshold: Number(process.env.FLOORPLAN_YOLO_CONFIDENCE ?? 0.35),
  iouThreshold: Number(process.env.FLOORPLAN_YOLO_IOU ?? 0.45),
  maxDetections: Number(process.env.FLOORPLAN_YOLO_MAX_DETECTIONS ?? 100),
  classCount,
  labels: normalizedLabels,
}

let sessionPromise: Promise<ort.InferenceSession> | null = null

const resolveModelPath = () => {
  if (path.isAbsolute(yoloConfig.modelPath)) {
    return yoloConfig.modelPath
  }
  return path.join(process.cwd(), yoloConfig.modelPath)
}

const ensureModelSession = async () => {
  if (sessionPromise) return sessionPromise
  const resolvedPath = resolveModelPath()
  if (!fs.existsSync(resolvedPath)) {
    throw new Error(
      `YOLO model not found at ${resolvedPath}. Download a YOLO ONNX model (e.g., via "yolo export model=yolov8n.pt format=onnx") and update FLOORPLAN_YOLO_MODEL.`
    )
  }
  const availableBackends = typeof (ort as any).listSupportedBackends === "function"
    ? ((ort as any).listSupportedBackends() as Array<{ name: string }>).map((backend) => backend.name)
    : []
  const preferredProviders = availableBackends.includes("cpu")
    ? ["cpu"]
    : availableBackends.length > 0
      ? availableBackends
      : undefined

  const sessionOptions = preferredProviders ? { executionProviders: preferredProviders } : {}

  sessionPromise = ort.InferenceSession.create(resolvedPath, sessionOptions)
  return sessionPromise
}

const preprocessImage = async (buffer: Buffer, inputSize: number): Promise<{ tensor: ort.Tensor; metadata: PreprocessMetadata }> => {
  const image = sharp(buffer).removeAlpha()
  const metadata = await image.metadata()
  const originalWidth = metadata.width ?? 0
  const originalHeight = metadata.height ?? 0

  if (!originalWidth || !originalHeight) {
    throw new Error("Unable to read image dimensions for YOLO preprocessing")
  }

  const scale = Math.min(inputSize / originalWidth, inputSize / originalHeight)
  const resizedWidth = Math.round(originalWidth * scale)
  const resizedHeight = Math.round(originalHeight * scale)
  const padX = (inputSize - resizedWidth) / 2
  const padY = (inputSize - resizedHeight) / 2

  const { data } = await image
    .resize({
      width: inputSize,
      height: inputSize,
      fit: "contain",
      background: { r: 0, g: 0, b: 0, alpha: 1 },
    })
    .raw()
    .toBuffer({ resolveWithObject: true })

  const area = inputSize * inputSize
  const floatData = new Float32Array(area * 3)

  for (let i = 0; i < area; i++) {
    const pixelIndex = i * 3
    floatData[i] = data[pixelIndex] / 255
    floatData[i + area] = data[pixelIndex + 1] / 255
    floatData[i + area * 2] = data[pixelIndex + 2] / 255
  }

  return {
    tensor: new ort.Tensor("float32", floatData, [1, 3, inputSize, inputSize]),
    metadata: { originalWidth, originalHeight, scale, padX, padY, inputSize },
  }
}

const transposeIfNeeded = (values: Float32Array, rows: number, cols: number) => {
  const transposed = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      transposed[c * rows + r] = values[r * cols + c]
    }
  }
  return transposed
}

const prepareBoxBuffer = (tensor: ort.Tensor, numClasses: number) => {
  const dims = tensor.dims
  const values = tensor.data as Float32Array

  if (dims.length === 3) {
    const [, dim1, dim2] = dims
    if (dim1 >= 4 && dim1 <= numClasses + 6 && dim2 > dim1) {
      return { buffer: transposeIfNeeded(values, dim1, dim2), numBoxes: dim2, attributes: dim1 }
    }
    return { buffer: values, numBoxes: dim1, attributes: dim2 }
  }

  if (dims.length === 2) {
    const [dim1, dim2] = dims
    if (dim2 >= numClasses + 4) {
      return { buffer: values, numBoxes: dim1, attributes: dim2 }
    }
    return { buffer: transposeIfNeeded(values, dim1, dim2), numBoxes: dim2, attributes: dim1 }
  }

  throw new Error(`Unsupported YOLO output shape: ${JSON.stringify(dims)}`)
}

const nonMaxSuppression = (candidates: DetectionCandidate[], iouThreshold: number, maxDetections: number) => {
  const selected: DetectionCandidate[] = []
  const sorted = [...candidates].sort((a, b) => b.score - a.score)

  const iou = (a: DetectionCandidate, b: DetectionCandidate) => {
    const x1 = Math.max(a.box.x1, b.box.x1)
    const y1 = Math.max(a.box.y1, b.box.y1)
    const x2 = Math.min(a.box.x2, b.box.x2)
    const y2 = Math.min(a.box.y2, b.box.y2)
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
    const areaA = Math.max(0, a.box.x2 - a.box.x1) * Math.max(0, a.box.y2 - a.box.y1)
    const areaB = Math.max(0, b.box.x2 - b.box.x1) * Math.max(0, b.box.y2 - b.box.y1)
    const union = areaA + areaB - intersection
    return union === 0 ? 0 : intersection / union
  }

  while (sorted.length > 0 && selected.length < maxDetections) {
    const current = sorted.shift()!
    selected.push(current)
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(current, sorted[i]) > iouThreshold) {
        sorted.splice(i, 1)
      }
    }
  }

  return selected
}

const postprocessDetections = (
  tensor: ort.Tensor,
  meta: PreprocessMetadata,
  classCount: number,
  labels: string[],
  confidenceThreshold: number,
  iouThreshold: number,
  maxDetections: number
): YoloDetectionResult[] => {
  const { buffer, numBoxes, attributes } = prepareBoxBuffer(tensor, classCount)
  const classOffset = attributes - classCount
  if (classOffset < 4) {
    throw new Error("YOLO output does not contain expected bounding box information")
  }

  const hasObjectness = classOffset > 4
  const candidates: DetectionCandidate[] = []

  for (let i = 0; i < numBoxes; i++) {
    const rowOffset = i * attributes
    const cx = buffer[rowOffset]
    const cy = buffer[rowOffset + 1]
    const boxWidth = buffer[rowOffset + 2]
    const boxHeight = buffer[rowOffset + 3]
    const objectness = hasObjectness ? sigmoid(buffer[rowOffset + 4]) : 1

    let bestScore = -Infinity
    let bestLabelIndex = -1

    for (let c = 0; c < classCount; c++) {
      const rawScore = buffer[rowOffset + classOffset + c]
      const classProbability = sigmoid(rawScore)
      const score = classProbability * objectness
      if (score > bestScore) {
        bestScore = score
        bestLabelIndex = c
      }
    }

    if (bestLabelIndex === -1 || bestScore < confidenceThreshold) {
      continue
    }

    const x1Letterbox = cx - boxWidth / 2
    const y1Letterbox = cy - boxHeight / 2
    const x2Letterbox = cx + boxWidth / 2
    const y2Letterbox = cy + boxHeight / 2

    const x1 = (x1Letterbox - meta.padX) / meta.scale
    const y1 = (y1Letterbox - meta.padY) / meta.scale
    const x2 = (x2Letterbox - meta.padX) / meta.scale
    const y2 = (y2Letterbox - meta.padY) / meta.scale

    candidates.push({
      label: labels[bestLabelIndex] ?? `class-${bestLabelIndex}`,
      score: bestScore,
      box: {
        x1: clamp(x1, 0, meta.originalWidth),
        y1: clamp(y1, 0, meta.originalHeight),
        x2: clamp(x2, 0, meta.originalWidth),
        y2: clamp(y2, 0, meta.originalHeight),
      },
    })
  }

  const nmsResults = nonMaxSuppression(candidates, iouThreshold, maxDetections)

  return nmsResults.map((candidate) => {
    const width = Math.max(0, candidate.box.x2 - candidate.box.x1)
    const height = Math.max(0, candidate.box.y2 - candidate.box.y1)
    return {
      name: candidate.label,
      score: candidate.score,
      boundingBox: {
        x: clamp(candidate.box.x1 / meta.originalWidth, 0, 1),
        y: clamp(candidate.box.y1 / meta.originalHeight, 0, 1),
        width: clamp(width / meta.originalWidth, 0, 1),
        height: clamp(height / meta.originalHeight, 0, 1),
      },
    }
  })
}

export const detectFloorplanObjects = async (imageBuffer: Buffer): Promise<YoloDetectionResult[]> => {
  const session = await ensureModelSession()
  const { tensor, metadata } = await preprocessImage(imageBuffer, yoloConfig.inputSize)
  const inputName = session.inputNames[0]
  const feeds: Record<string, ort.Tensor> = { [inputName]: tensor }
  const results = await session.run(feeds)
  const outputName = session.outputNames[0]
  const outputTensor = results[outputName]
  if (!outputTensor) {
    throw new Error("YOLO inference did not return any outputs")
  }

  return postprocessDetections(
    outputTensor,
    metadata,
    yoloConfig.classCount,
    yoloConfig.labels,
    yoloConfig.confidenceThreshold,
    yoloConfig.iouThreshold,
    yoloConfig.maxDetections
  )
}
