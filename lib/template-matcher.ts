import { performance } from "node:perf_hooks"
import sharp from "sharp"
import { type Detection } from "./detections"

export interface NormalizedBox {
  x: number
  y: number
  width: number
  height: number
}

interface PixelBox {
  left: number
  top: number
  width: number
  height: number
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const DEFAULT_TEMPLATE_SIZE = Number(process.env.TEMPLATE_MATCH_SIZE ?? 96)
const DEFAULT_SCAN_STRIDE_FACTOR = Number(process.env.TEMPLATE_GRID_STRIDE ?? 0.55)
const DEFAULT_SCAN_TOP_K = Math.max(1, Number(process.env.TEMPLATE_GRID_TOP_K ?? 32))
const DEFAULT_SCAN_MAX_CANDIDATES = Number(process.env.TEMPLATE_GRID_MAX_CANDIDATES ?? 1200)
const parsedScaleSteps = (process.env.TEMPLATE_GRID_SCALES ?? "0.65,0.85,1,1.25,1.5")
  .split(",")
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value > 0)
  .map((value) => clamp(value, 0.25, 4))
  .slice(0, 8)
const DEFAULT_SCAN_SCALE_STEPS = parsedScaleSteps.length > 0 ? parsedScaleSteps : [1]

export type TemplatePreprocessMode = "grayscale" | "binary" | "hybrid" | "edges"
export type TemplateMatcherPresetName = "auto" | "linework" | "texture"

interface DescriptorOptions {
  size: number
  preprocessMode?: TemplatePreprocessMode
  blurSigma?: number
  sharpen?: boolean
}

const DEFAULT_PREPROCESS_MODE: TemplatePreprocessMode = "hybrid"

const applyBinaryThreshold = (source: Float32Array, threshold = 0.52) => {
  const output = new Float32Array(source.length)
  for (let i = 0; i < source.length; i++) {
    output[i] = source[i] >= threshold ? 1 : 0
  }
  return output
}

const computeGradientDescriptor = (source: Float32Array, size: number) => {
  const output = new Float32Array(source.length)
  const clampIndex = (x: number, y: number) => source[y * size + x]
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const gx =
        -clampIndex(x - 1, y - 1) - 2 * clampIndex(x - 1, y) - clampIndex(x - 1, y + 1) +
        clampIndex(x + 1, y - 1) + 2 * clampIndex(x + 1, y) + clampIndex(x + 1, y + 1)
      const gy =
        -clampIndex(x - 1, y - 1) - 2 * clampIndex(x, y - 1) - clampIndex(x + 1, y - 1) +
        clampIndex(x - 1, y + 1) + 2 * clampIndex(x, y + 1) + clampIndex(x + 1, y + 1)
      const magnitude = Math.sqrt(gx * gx + gy * gy)
      output[y * size + x] = Math.min(1, magnitude / 4)
    }
  }
  return output
}

const buildHybridDescriptor = (base: Float32Array, size: number) => {
  const binary = applyBinaryThreshold(base)
  const gradient = computeGradientDescriptor(base, size)
  const output = new Float32Array(base.length)
  for (let i = 0; i < base.length; i++) {
    output[i] = Math.min(1, 0.55 * base[i] + 0.3 * gradient[i] + 0.15 * binary[i])
  }
  return output
}

const preprocessDescriptor = (base: Float32Array, size: number, mode?: TemplatePreprocessMode) => {
  switch (mode ?? DEFAULT_PREPROCESS_MODE) {
    case "binary":
      return applyBinaryThreshold(base)
    case "edges":
      return computeGradientDescriptor(base, size)
    case "grayscale":
      return base
    case "hybrid":
    default:
      return buildHybridDescriptor(base, size)
  }
}

const normalizeDescriptor = (descriptor: Float32Array) => {
  if (!descriptor.length) return descriptor
  let sum = 0
  for (let i = 0; i < descriptor.length; i++) {
    sum += descriptor[i]
  }
  const mean = sum / descriptor.length
  let variance = 0
  for (let i = 0; i < descriptor.length; i++) {
    const centered = descriptor[i] - mean
    descriptor[i] = centered
    variance += centered * centered
  }
  const std = Math.sqrt(Math.max(variance / descriptor.length, 1e-6))
  for (let i = 0; i < descriptor.length; i++) {
    descriptor[i] = descriptor[i] / std
  }
  return descriptor
}

const mergeScaleSteps = (...lists: Array<number[] | undefined>) => {
  const values = new Set<number>()
  for (const list of lists) {
    list?.forEach((value) => {
      if (!Number.isFinite(value)) return
      const clampedValue = clamp(value, 0.2, 4)
      values.add(Number(clampedValue.toFixed(3)))
    })
  }
  return values.size ? Array.from(values).sort((a, b) => a - b) : []
}

const resolveScaleSteps = (base?: number[], extra?: number[]) => {
  const merged = mergeScaleSteps(base?.length ? base : DEFAULT_SCAN_SCALE_STEPS, extra)
  return merged.length ? merged : DEFAULT_SCAN_SCALE_STEPS
}

interface TemplateMatcherPresetConfig extends Partial<DetectTemplateOptions> {
  passes?: TemplateMatchPass[]
}

const TEMPLATE_PRESETS: Record<TemplateMatcherPresetName, TemplateMatcherPresetConfig> = {
  auto: {
    searchPadding: 0.4,
  },
  linework: {
    preprocessMode: "edges",
    size: 128,
    searchPadding: 0.2,
    strideFactor: 0.4,
    scaleSteps: [0.55, 0.75, 0.95, 1.1, 1.35, 1.6],
    blurSigma: 0.2,
    sharpen: true,
  },
  texture: {
    preprocessMode: "grayscale",
    size: 112,
    searchPadding: 0.5,
    strideFactor: 0.6,
    scaleSteps: [0.65, 0.85, 1, 1.2],
    blurSigma: 0.45,
    sharpen: false,
  },
}

const applyPreset = (options?: DetectTemplateOptions): DetectTemplateOptions => {
  const provided = options ?? {}
  const presetName = provided.mode ?? "auto"
  const preset = TEMPLATE_PRESETS[presetName] ?? TEMPLATE_PRESETS.auto
  return {
    ...preset,
    ...provided,
    mode: presetName,
    size: provided.size ?? preset.size,
    strideFactor: provided.strideFactor ?? preset.strideFactor,
    scaleSteps: provided.scaleSteps ?? preset.scaleSteps,
    preprocessMode: provided.preprocessMode ?? preset.preprocessMode,
    blurSigma: provided.blurSigma ?? preset.blurSigma,
    sharpen: provided.sharpen ?? preset.sharpen,
    searchPadding: provided.searchPadding ?? preset.searchPadding,
    passes: provided.passes ?? preset.passes,
  }
}

const buildDefaultPasses = (options?: DetectTemplateOptions): TemplateMatchPass[] => {
  const baseScale = options?.scaleSteps ?? DEFAULT_SCAN_SCALE_STEPS
  const baseSize = options?.size ?? DEFAULT_TEMPLATE_SIZE
  return [
    {
      name: "coarse",
      strideFactor: options?.strideFactor ?? DEFAULT_SCAN_STRIDE_FACTOR,
      scaleSteps: resolveScaleSteps(baseScale),
      size: baseSize,
      preprocessMode: options?.preprocessMode ?? DEFAULT_PREPROCESS_MODE,
      topK: options?.topK ?? DEFAULT_SCAN_TOP_K,
      maxCandidates: options?.maxCandidates ?? DEFAULT_SCAN_MAX_CANDIDATES,
      searchPadding: options?.searchPadding ?? 0.35,
    },
    {
      name: "dense",
      strideFactor: 0.35,
      scaleSteps: resolveScaleSteps(baseScale, [0.7, 1, 1.3]),
      size: Math.round(baseSize * 1.15),
      preprocessMode: options?.preprocessMode ?? "grayscale",
      blurSigma: 0.5,
      topK: Math.max(options?.topK ?? DEFAULT_SCAN_TOP_K, 40),
      maxCandidates: Math.max(options?.maxCandidates ?? DEFAULT_SCAN_MAX_CANDIDATES, 2200),
      searchPadding: options?.searchPadding ?? 0.25,
    },
    {
      name: "precision",
      strideFactor: 0.22,
      scaleSteps: resolveScaleSteps(baseScale, [0.55, 0.75, 1.4, 1.65]),
      size: Math.round(baseSize * 1.3),
      preprocessMode: "edges",
      blurSigma: 0.3,
      sharpen: true,
      topK: Math.max(options?.topK ?? DEFAULT_SCAN_TOP_K, 60),
      maxCandidates: Math.max(options?.maxCandidates ?? DEFAULT_SCAN_MAX_CANDIDATES, 4500),
      searchPadding: options?.searchPadding ?? 0.18,
    },
  ]
}

const preparePassOptions = (options?: DetectTemplateOptions) => {
  const passes = options?.passes?.length ? options.passes : buildDefaultPasses(options)
  const baseScale = options?.scaleSteps ?? DEFAULT_SCAN_SCALE_STEPS
  return passes.map((pass, index) => ({
    size: pass.size ?? options?.size ?? DEFAULT_TEMPLATE_SIZE,
    strideFactor: pass.strideFactor ?? options?.strideFactor ?? DEFAULT_SCAN_STRIDE_FACTOR,
    scaleSteps: resolveScaleSteps(baseScale, pass.scaleSteps),
    topK: pass.topK ?? options?.topK ?? DEFAULT_SCAN_TOP_K,
    maxCandidates: pass.maxCandidates ?? options?.maxCandidates ?? DEFAULT_SCAN_MAX_CANDIDATES,
    preprocessMode: pass.preprocessMode ?? options?.preprocessMode,
    blurSigma: pass.blurSigma ?? options?.blurSigma,
    sharpen: pass.sharpen ?? options?.sharpen,
    searchPadding: pass.searchPadding ?? options?.searchPadding,
    searchRegion: pass.searchRegion ?? options?.searchRegion,
    passName: pass.passName ?? pass.name ?? `pass-${index + 1}`,
  }))
}

const normalizedToPixelBox = (box: NormalizedBox, imageWidth: number, imageHeight: number): PixelBox | null => {
  const width = Math.max(1, Math.round(box.width * imageWidth))
  const height = Math.max(1, Math.round(box.height * imageHeight))
  if (!width || !height) return null
  const left = clamp(Math.round(box.x * imageWidth), 0, Math.max(0, imageWidth - 1))
  const top = clamp(Math.round(box.y * imageHeight), 0, Math.max(0, imageHeight - 1))
  const adjustedWidth = Math.min(width, imageWidth - left)
  const adjustedHeight = Math.min(height, imageHeight - top)
  if (adjustedWidth <= 0 || adjustedHeight <= 0) return null
  return { left, top, width: adjustedWidth, height: adjustedHeight }
}

const expandNormalizedBox = (box: NormalizedBox, padding: number): NormalizedBox => {
  if (padding <= 0) return box
  const padX = box.width * padding
  const padY = box.height * padding
  const x = clamp(box.x - padX, 0, 1)
  const y = clamp(box.y - padY, 0, 1)
  const right = clamp(box.x + box.width + padX, 0, 1)
  const bottom = clamp(box.y + box.height + padY, 0, 1)
  return {
    x,
    y,
    width: clamp(right - x, 0, 1),
    height: clamp(bottom - y, 0, 1),
  }
}

const buildDescriptor = async (
  imageBuffer: Buffer,
  dimensions: { width?: number; height?: number },
  box: NormalizedBox,
  descriptorOptions: DescriptorOptions
): Promise<Float32Array | null> => {
  if (!dimensions.width || !dimensions.height) return null
  const pixelBox = normalizedToPixelBox(box, dimensions.width, dimensions.height)
  if (!pixelBox) return null
  const { left, top, width, height } = pixelBox
  const targetSize = Math.max(4, Math.round(descriptorOptions.size))

  let pipeline = sharp(imageBuffer)
    .extract({ left, top, width, height })
    .resize(targetSize, targetSize, { fit: "fill" })
    .removeAlpha()
    .greyscale()
    .normalize()

  if (descriptorOptions.blurSigma) {
    pipeline = pipeline.blur(descriptorOptions.blurSigma)
  }

  if (descriptorOptions.sharpen) {
    pipeline = pipeline.sharpen()
  }

  const rawBuffer = await pipeline.raw().toBuffer({ resolveWithObject: false })
  const base = new Float32Array(rawBuffer.length)
  for (let i = 0; i < rawBuffer.length; i++) {
    base[i] = rawBuffer[i] / 255
  }

  const processed = preprocessDescriptor(base, targetSize, descriptorOptions.preprocessMode)
  return normalizeDescriptor(processed)
}

const pixelBoxToNormalized = (box: PixelBox, imageWidth: number, imageHeight: number): NormalizedBox => ({
  x: clamp(box.left / imageWidth, 0, 1),
  y: clamp(box.top / imageHeight, 0, 1),
  width: clamp(box.width / imageWidth, 0, 1),
  height: clamp(box.height / imageHeight, 0, 1),
})

const generateGridBoxes = (
  reference: PixelBox,
  imageWidth: number,
  imageHeight: number,
  scaleSteps: number[],
  strideFactor: number,
  searchRegion?: PixelBox
) => {
  const boxes: PixelBox[] = []
  const safeStrideFactor = clamp(strideFactor, 0.1, 1)
  const bounds = searchRegion ?? { left: 0, top: 0, width: imageWidth, height: imageHeight }
  const boundsRight = Math.min(imageWidth, bounds.left + bounds.width)
  const boundsBottom = Math.min(imageHeight, bounds.top + bounds.height)
  for (const scale of scaleSteps) {
    const width = Math.max(4, Math.round(reference.width * scale))
    const height = Math.max(4, Math.round(reference.height * scale))
    const maxLeft = Math.max(bounds.left, Math.min(boundsRight - width, imageWidth - width))
    const maxTop = Math.max(bounds.top, Math.min(boundsBottom - height, imageHeight - height))
    if (maxLeft < bounds.left || maxTop < bounds.top) {
      boxes.push({ left: bounds.left, top: bounds.top, width: Math.min(width, bounds.width), height: Math.min(height, bounds.height) })
      continue
    }
    const strideX = Math.max(1, Math.round(width * safeStrideFactor))
    const strideY = Math.max(1, Math.round(height * safeStrideFactor))
    for (let top = bounds.top; top <= maxTop; top += strideY) {
      for (let left = bounds.left; left <= maxLeft; left += strideX) {
        boxes.push({ left, top, width, height })
      }
    }
  }
  return boxes
}

const cosineSimilarity = (a: Float32Array, b: Float32Array) => {
  if (a.length !== b.length) return 0
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < a.length; i++) {
    const v1 = a[i]
    const v2 = b[i]
    dot += v1 * v2
    normA += v1 * v1
    normB += v2 * v2
  }
  if (normA === 0 || normB === 0) return 0
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

export const computeTemplateSimilarities = async (
  imageBuffer: Buffer,
  selectionBox: NormalizedBox,
  targetBoxes: NormalizedBox[],
  options?: Partial<DescriptorOptions>
) => {
  const templateSize = options?.size ?? DEFAULT_TEMPLATE_SIZE
  const descriptorOptions: DescriptorOptions = {
    size: templateSize,
    preprocessMode: options?.preprocessMode,
    blurSigma: options?.blurSigma,
    sharpen: options?.sharpen,
  }
  const metadata = await sharp(imageBuffer).metadata()
  const referenceDescriptor = await buildDescriptor(imageBuffer, metadata, selectionBox, descriptorOptions)
  if (!referenceDescriptor) {
    return targetBoxes.map(() => null)
  }
  const similarities: Array<number | null> = []
  for (const box of targetBoxes) {
    try {
      const descriptor = await buildDescriptor(imageBuffer, metadata, box, descriptorOptions)
      if (!descriptor) {
        similarities.push(null)
        continue
      }
      similarities.push(cosineSimilarity(referenceDescriptor, descriptor))
    } catch (error) {
      console.error("Failed to compute descriptor", error)
      similarities.push(null)
    }
  }
  return similarities
}

export interface TemplateMatch {
  boundingBox: NormalizedBox
  templateSimilarity: number
  pass?: string
}

export interface TemplateMatchStats {
  pass: string
  candidates: number
  kept: number
  durationMs: number
  bestSimilarity: number | null
}

export interface TemplateMatchOptions {
  size?: number
  strideFactor?: number
  scaleSteps?: number[]
  topK?: number
  maxCandidates?: number
  preprocessMode?: TemplatePreprocessMode
  blurSigma?: number
  sharpen?: boolean
  passName?: string
  searchRegion?: NormalizedBox
  searchPadding?: number
  statsCollector?: (stats: TemplateMatchStats) => void
}

export interface TemplateMatchPass extends TemplateMatchOptions {
  name?: string
}

export interface DetectTemplateOptions extends TemplateMatchOptions {
  label?: string
  minSimilarity?: number
  passes?: TemplateMatchPass[]
  relaxFallback?: boolean
  mode?: TemplateMatcherPresetName
}

export const scanTemplateMatches = async (
  imageBuffer: Buffer,
  selectionBox: NormalizedBox,
  options?: TemplateMatchOptions
) => {
  const templateSize = options?.size ?? DEFAULT_TEMPLATE_SIZE
  const passName = options?.passName ?? "pass"
  const metadata = await sharp(imageBuffer).metadata()
  const imageWidth = metadata.width ?? 0
  const imageHeight = metadata.height ?? 0
  if (!imageWidth || !imageHeight) return []

  const pixelSelection = normalizedToPixelBox(selectionBox, imageWidth, imageHeight)
  if (!pixelSelection) return []

  const scaleSteps = (options?.scaleSteps?.length ? options.scaleSteps : DEFAULT_SCAN_SCALE_STEPS) ?? [1]
  const strideFactor = options?.strideFactor ?? DEFAULT_SCAN_STRIDE_FACTOR
  const searchRegionBox = options?.searchRegion
    ? normalizedToPixelBox(options.searchRegion, imageWidth, imageHeight)
    : normalizedToPixelBox(expandNormalizedBox(selectionBox, options?.searchPadding ?? 0.35), imageWidth, imageHeight)
  const candidateBoxes = generateGridBoxes(
    pixelSelection,
    imageWidth,
    imageHeight,
    scaleSteps,
    strideFactor,
    searchRegionBox ?? undefined
  )
  if (candidateBoxes.length === 0) return []

  const maxCandidates = options?.maxCandidates ?? DEFAULT_SCAN_MAX_CANDIDATES
  const step = candidateBoxes.length > maxCandidates ? Math.ceil(candidateBoxes.length / maxCandidates) : 1
  const boundedBoxes = step > 1 ? candidateBoxes.filter((_, index) => index % step === 0) : candidateBoxes
  const normalizedBoxes = boundedBoxes.map((box) => pixelBoxToNormalized(box, imageWidth, imageHeight))

  const start = performance.now ? performance.now() : Date.now()
  const similarities = await computeTemplateSimilarities(imageBuffer, selectionBox, normalizedBoxes, {
    size: templateSize,
    preprocessMode: options?.preprocessMode,
    blurSigma: options?.blurSigma,
    sharpen: options?.sharpen,
  })
  const matches: TemplateMatch[] = normalizedBoxes
    .map((box, index) => ({ boundingBox: box, templateSimilarity: similarities[index] ?? null }))
    .filter((match): match is TemplateMatch => match.templateSimilarity !== null)
    .sort((a, b) => (b.templateSimilarity ?? 0) - (a.templateSimilarity ?? 0))

  const topK = options?.topK ?? DEFAULT_SCAN_TOP_K
  const trimmed = matches.slice(0, topK)
  const durationMs = (performance.now ? performance.now() : Date.now()) - start
  const stats: TemplateMatchStats = {
    pass: passName,
    candidates: candidateBoxes.length,
    kept: trimmed.length,
    durationMs,
    bestSimilarity: trimmed[0]?.templateSimilarity ?? null,
  }
  trimmed.forEach((match) => {
    match.pass = passName
  })
  options?.statsCollector?.(stats)
  return trimmed
}

export type TemplateDetection = Detection & { templateSimilarity: number; templatePass?: string }

export interface TemplateDetectionResult {
  matches: TemplateDetection[]
  bestSimilarity: number | null
  stats: TemplateMatchStats[]
}

export const detectTemplateMatches = async (
  imageBuffer: Buffer,
  selectionBox: NormalizedBox,
  options?: DetectTemplateOptions
): Promise<TemplateDetectionResult> => {
  const resolvedOptions = applyPreset(options)
  const label = resolvedOptions.label ?? "template-match"
  const minSimilarity = clamp(resolvedOptions.minSimilarity ?? 0.4, -1, 1)
  const passOptions = preparePassOptions(resolvedOptions)
  const stats: TemplateMatchStats[] = []
  let bestMatches: TemplateMatch[] = []

  for (const pass of passOptions) {
    const matches = await scanTemplateMatches(imageBuffer, selectionBox, {
      ...pass,
      statsCollector: (stat) => stats.push(stat),
    })
    if (!matches.length) continue
    if (!bestMatches.length || matches[0].templateSimilarity > bestMatches[0].templateSimilarity) {
      bestMatches = matches
    }

    const filtered = matches.filter((match) => match.templateSimilarity >= minSimilarity)
    if (filtered.length) {
      return {
        matches: filtered.map((match) => formatDetection(match, label)),
        bestSimilarity: filtered[0]?.templateSimilarity ?? null,
        stats,
      }
    }
  }

  if (!bestMatches.length) {
    return { matches: [], bestSimilarity: null, stats }
  }

  if (resolvedOptions.relaxFallback === false) {
    return {
      matches: bestMatches.map((match) => formatDetection(match, label)),
      bestSimilarity: bestMatches[0]?.templateSimilarity ?? null,
      stats,
    }
  }

  const relaxedThreshold = clamp(minSimilarity * 0.8, -1, 1)
  const relaxedMatches = bestMatches.filter((match) => match.templateSimilarity >= relaxedThreshold)
  const finalMatches = relaxedMatches.length
    ? relaxedMatches
    : bestMatches.slice(0, Math.min(bestMatches.length, passOptions[passOptions.length - 1]?.topK ?? DEFAULT_SCAN_TOP_K))

  return {
    matches: finalMatches.map((match) => formatDetection(match, label)),
    bestSimilarity: bestMatches[0]?.templateSimilarity ?? null,
    stats,
  }
}

const formatDetection = (match: TemplateMatch, label: string): TemplateDetection => ({
  label,
  confidence: clamp((match.templateSimilarity + 1) / 2, 0, 1),
  boundingBox: match.boundingBox,
  templateSimilarity: match.templateSimilarity,
  templatePass: match.pass,
})
