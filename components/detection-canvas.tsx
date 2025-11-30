"use client"
import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import {
  ChevronLeft,
  ChevronRight,
  FileJson,
  Hand,
  Loader2,
  MousePointer2,
  Sparkles,
  Trash2,
  Download,
  Upload,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { DetectedObject as CocoDetectedObject, ObjectDetection } from "@tensorflow-models/coco-ssd"

interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

interface DetectedObject {
  name: string
  score: number
  boundingBox: BoundingBox
  templateSimilarity?: number | null
  templatePass?: string | null
}

interface SelectionRect {
  x: number
  y: number
  width: number
  height: number
}

interface DeepZoomPage {
  id: string
  label: string
  imageSrc: string
  width: number
  height: number
  originalFileName?: string
  pageNumber?: number
}

interface TemplatePassStat {
  pass: string
  candidates: number
  kept: number
  durationMs: number
  bestSimilarity: number | null
}

const TEMPLATE_SIMILARITY_THRESHOLD = Number(process.env.NEXT_PUBLIC_TEMPLATE_MATCH_THRESHOLD ?? 0.82)

const getTemplateMatches = (objects: DetectedObject[], selected: DetectedObject | null) => {
  if (!selected) return []
  const scoredMatches = objects.filter(
    (obj) =>
      obj !== selected && typeof obj.templateSimilarity === "number" && obj.templateSimilarity >= TEMPLATE_SIMILARITY_THRESHOLD
  )
  if (scoredMatches.length > 0) {
    return scoredMatches
  }
  return objects.filter((obj) => obj !== selected)
}

const normalizeSimilarity = (similarity?: number | null) => {
  if (typeof similarity !== "number" || Number.isNaN(similarity)) return null
  return clamp((similarity + 1) / 2, 0, 1)
}

const buildSimilarityPalette = (similarity?: number | null) => {
  const normalized = normalizeSimilarity(similarity)
  if (normalized === null) {
    return {
      stroke: "#10b981",
      fill: "rgba(16, 185, 129, 0.18)",
      badgeBg: "rgba(16, 185, 129, 0.12)",
      panelBg: "rgba(16, 185, 129, 0.08)",
      text: "#065f46",
    }
  }
  const hue = 12 + normalized * 110
  return {
    stroke: `hsl(${hue}, 78%, 46%)`,
    fill: `hsla(${hue}, 78%, 46%, 0.22)`,
    badgeBg: `hsla(${hue}, 85%, 52%, 0.16)`,
    panelBg: `hsla(${hue}, 78%, 46%, 0.09)`,
    text: `hsl(${hue}, 40%, 22%)`,
  }
}

const formatSimilarityPercent = (similarity?: number | null) =>
  typeof similarity === "number" ? `${(similarity * 100).toFixed(1)}% sim` : null

const describeDetectionLabel = (detection: DetectedObject) => {
  const similarityLabel = formatSimilarityPercent(detection.templateSimilarity)
  const confidenceLabel = `${Math.round(detection.score * 100)}% conf`
  const passLabel = detection.templatePass ? ` (${detection.templatePass})` : ""
  return `${detection.name}${passLabel} · ${similarityLabel ?? confidenceLabel}`
}

const formatMs = (duration: number) => {
  if (!Number.isFinite(duration)) return "—"
  if (duration >= 1000) {
    return `${(duration / 1000).toFixed(2)} s`
  }
  return `${duration.toFixed(duration >= 10 ? 0 : 1)} ms`
}

const formatErrorDetails = (details: unknown) => {
  if (!details) return "No additional details"
  if (typeof details === "string") return details.trim() || "No additional details"
  if (typeof details === "object") {
    const entries = Object.entries(details as Record<string, unknown>)
    return entries.length > 0 ? JSON.stringify(details, null, 2) : "No additional details"
  }
  return String(details)
}

const readApiError = async (response: Response) => {
  let message = `HTTP error ${response.status}`
  let details: unknown = null
  try {
    const data = await response.clone().json()
    message = data?.message || data?.error || message
    details = data
  } catch (jsonError) {
    try {
      const text = await response.text()
      if (text) {
        message = text
        details = text
      }
    } catch (textError) {
      console.error("Failed to parse API error payload", jsonError, textError)
    }
  }
  return { message, details }
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const mapServerDetections = (annotations: any[]): DetectedObject[] => {
  if (!Array.isArray(annotations)) return []
  return annotations.map((annotation: any, index: number) => {
    const templateSimilarity =
      typeof annotation?.templateSimilarity === "number"
        ? clamp(annotation.templateSimilarity, -1, 1)
        : typeof annotation?.confidence === "number"
          ? clamp(annotation.confidence, 0, 1)
          : null

    return {
      name: annotation?.label || `region-${index + 1}`,
      score: typeof annotation?.confidence === "number" ? clamp(annotation.confidence, 0, 1) : 0,
      boundingBox: {
        x: annotation?.boundingBox?.x ?? 0,
        y: annotation?.boundingBox?.y ?? 0,
        width: annotation?.boundingBox?.width ?? 0,
        height: annotation?.boundingBox?.height ?? 0,
      },
      templateSimilarity,
      templatePass: annotation?.templatePass ?? null,
    }
  })
}

const normalizeToPage = (value: number, dimension: number) => {
  if (!dimension || Number.isNaN(dimension)) return 0
  return clamp(value / dimension, 0, 1)
}

const boundingBoxesClose = (a?: BoundingBox, b?: BoundingBox, epsilon = 0.002) => {
  if (!a || !b) return false
  return (
    Math.abs(a.x - b.x) <= epsilon &&
    Math.abs(a.y - b.y) <= epsilon &&
    Math.abs(a.width - b.width) <= epsilon &&
    Math.abs(a.height - b.height) <= epsilon
  )
}

const shouldDisableCloudVision = (message: string) => {
  if (!message) return false
  return /permission|billing|credential|quota|vision client/i.test(message)
}

const summarizeCloudVisionIssue = (message?: string | null) => {
  if (!message) return "Cloud Vision is unavailable"
  if (/billing/i.test(message)) return "Google Cloud Vision billing is disabled"
  if (/permission/i.test(message)) return "Google Cloud Vision request was denied"
  if (/credential/i.test(message)) return "Google Cloud Vision credentials are missing or invalid"
  if (/quota/i.test(message)) return "Google Cloud Vision quota was exceeded"
  if (/unavailable/i.test(message)) return "Google Cloud Vision service is unavailable"
  return message
}

const loadImageElement = (src: string) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error("Failed to load image"))
    img.src = src
  })

export default function SelectionCanvas() {
  const [pages, setPages] = useState<DeepZoomPage[]>([])
  const [activePageIndex, setActivePageIndex] = useState(0)
  const [mode, setMode] = useState<"pan" | "select">("pan")
  const [analysisResult, setAnalysisResult] = useState<string | null>(null)
  const [detectedObjects, setDetectedObjects] = useState<DetectedObject[]>([])
  const [selectedObject, setSelectedObject] = useState<DetectedObject | null>(null)
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null)
  const [uploadedPath, setUploadedPath] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isPreparingPages, setIsPreparingPages] = useState(false)
  const [viewerReady, setViewerReady] = useState(false)
  const [osdReady, setOsdReady] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionRect | null>(null)
  const [matchStats, setMatchStats] = useState<TemplatePassStat[]>([])
  const [bestSimilarity, setBestSimilarity] = useState<number | null>(null)

  const assetInputRef = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)
  const viewerContainerRef = useRef<HTMLDivElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const interactionLayerRef = useRef<HTMLDivElement>(null)
  const selectionStartRef = useRef<{ x: number; y: number } | null>(null)
  const selectionBoxRef = useRef<SelectionRect | null>(null)
  const viewerRef = useRef<any>(null)
  const osdModuleRef = useRef<any>(null)
  const pdfjsRef = useRef<any>(null)
  const detectedObjectsRef = useRef<DetectedObject[]>([])
  const selectedObjectRef = useRef<DetectedObject | null>(null)
  const tfModuleRef = useRef<any>(null)
  const cocoModelRef = useRef<ObjectDetection | null>(null)
  const skipCloudVisionRef = useRef(false)
  const cloudVisionIssueRef = useRef<string | null>(null)
  const [isSelecting, setIsSelecting] = useState(false)

  const currentPage = pages[activePageIndex] ?? null

  useEffect(() => {
    if (typeof window === "undefined") return
    let cancelled = false
    const load = async () => {
      try {
        const mod = await import("openseadragon")
        if (!cancelled) {
          osdModuleRef.current = mod.default ?? mod
          setOsdReady(true)
        }
      } catch (err) {
        console.error("Failed to load OpenSeadragon", err)
        if (!cancelled) {
          setError("Unable to load deep zoom viewer")
        }
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    detectedObjectsRef.current = detectedObjects
  }, [detectedObjects])

  useEffect(() => {
    selectedObjectRef.current = selectedObject
  }, [selectedObject])

  const resetDetectionState = useCallback(() => {
    setAnalysisResult(null)
    setDetectedObjects([])
    setSelectedObject(null)
    setAnnotatedImage(null)
    setUploadedPath(null)
    setSelectionBox(null)
    setMatchStats([])
    setBestSimilarity(null)
    selectionBoxRef.current = null
  }, [])

  useEffect(() => {
    resetDetectionState()
    setViewerReady(false)
  }, [activePageIndex, resetDetectionState])
  const destroyViewer = useCallback(() => {
    if (viewerRef.current) {
      viewerRef.current.destroy()
      viewerRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => {
      destroyViewer()
    }
  }, [destroyViewer])

  const ensureLocalModel = useCallback(async () => {
    if (!tfModuleRef.current) {
      const tf = await import("@tensorflow/tfjs")
      if (typeof tf.ready === "function") {
        await tf.ready()
      }
      tfModuleRef.current = tf
    }
    if (!cocoModelRef.current) {
      const coco = await import("@tensorflow-models/coco-ssd")
      cocoModelRef.current = await coco.load({ base: "lite_mobilenet_v2" })
    }
    return cocoModelRef.current
  }, [])

  const runLocalDetection = useCallback(
    async (canvas: HTMLCanvasElement, page: DeepZoomPage, rect: SelectionRect, reason?: string | null) => {
      try {
        const model = await ensureLocalModel()
        if (!model) throw new Error("Unable to initialize local detection model")
        const predictions: CocoDetectedObject[] = await model.detect(canvas, undefined, 0.5)

        const objects: DetectedObject[] = predictions.map((prediction) => {
          const [localX, localY, localWidth, localHeight] = prediction.bbox
          const offsetX = rect.x + localX
          const offsetY = rect.y + localY
          return {
            name: prediction.class?.toLowerCase?.() || prediction.class,
            score: prediction.score,
            boundingBox: {
              x: normalizeToPage(offsetX, page.width),
              y: normalizeToPage(offsetY, page.height),
              width: normalizeToPage(localWidth, page.width),
              height: normalizeToPage(localHeight, page.height),
            },
          }
        })

        setDetectedObjects(objects)
        setSelectedObject(objects[0] ?? null)
        setError(null)
        setMatchStats([])
        setBestSimilarity(null)

        const readableReason = summarizeCloudVisionIssue(reason ?? cloudVisionIssueRef.current)
        const prefix = readableReason ? `Local detection fallback (${readableReason})` : "Local detection fallback"

        if (objects.length > 0) {
          const summary = objects
            .slice(0, Math.min(5, objects.length))
            .map((obj) => `${obj.name} (${Math.round(obj.score * 100)}%)`)
            .join(", ")
          setAnalysisResult(`${prefix}\nObjects: ${summary}`)
        } else {
          setAnalysisResult(`${prefix}\nNo recognizable objects found in the selected area.`)
        }

        return { success: true as const }
      } catch (err) {
        console.error("Local detection fallback failed", err)
        const message = err instanceof Error ? err.message : "Unknown error"
        return { success: false as const, error: message }
      }
    },
    [ensureLocalModel]
  )

  const drawOverlay = useCallback(() => {
    const viewer = viewerRef.current
    const canvas = overlayCanvasRef.current
    const page = currentPage
    if (!viewer || !canvas || !page) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const containerSize = viewer.viewport.getContainerSize()
    canvas.width = containerSize.x
    canvas.height = containerSize.y
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const drawRect = (
      rect: SelectionRect,
      strokeStyle: string,
      fillStyle: string,
      dashed = false,
      label?: string
    ) => {
      const topLeft = viewer.viewport.imageToViewportCoordinates(rect.x, rect.y)
      const bottomRight = viewer.viewport.imageToViewportCoordinates(rect.x + rect.width, rect.y + rect.height)
      const pixelTopLeft = viewer.viewport.viewportToViewerElementCoordinates(topLeft)
      const pixelBottomRight = viewer.viewport.viewportToViewerElementCoordinates(bottomRight)
      const x = pixelTopLeft.x
      const y = pixelTopLeft.y
      const width = pixelBottomRight.x - pixelTopLeft.x
      const height = pixelBottomRight.y - pixelTopLeft.y

      ctx.save()
      ctx.strokeStyle = strokeStyle
      ctx.lineWidth = 2
      ctx.setLineDash(dashed ? [6, 4] : [])
      ctx.strokeRect(x, y, width, height)
      if (fillStyle) {
        ctx.fillStyle = fillStyle
        ctx.fillRect(x, y, width, height)
      }
      ctx.restore()

      if (label) {
        const padding = 6
        const textWidth = ctx.measureText(label).width
        const labelWidth = textWidth + padding * 2
        const labelHeight = 18
        const labelX = clamp(x, 0, canvas.width - labelWidth)
        const labelY = y - labelHeight - 4 > 0 ? y - labelHeight - 4 : y + 4
        ctx.save()
        ctx.fillStyle = strokeStyle
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight)
        ctx.fillStyle = "#fff"
        ctx.font = "bold 11px Arial"
        ctx.textBaseline = "middle"
        ctx.fillText(label, labelX + padding, labelY + labelHeight / 2)
        ctx.restore()
      }
    }

    const selection = selectionBoxRef.current
    if (selection) {
      drawRect(selection, "#3b82f6", "rgba(59, 130, 246, 0.18)")
    }

    const selected = selectedObjectRef.current
    if (!selected) return
    const templateMatches = getTemplateMatches(detectedObjectsRef.current, selected)
    const relatedObjects = [selected, ...templateMatches]

    relatedObjects.forEach((obj) => {
      const rect: SelectionRect = {
        x: obj.boundingBox.x * page.width,
        y: obj.boundingBox.y * page.height,
        width: obj.boundingBox.width * page.width,
        height: obj.boundingBox.height * page.height,
      }
      const isPrimary = obj === selected
      const palette = isPrimary
        ? { stroke: "#3b82f6", fill: "rgba(59, 130, 246, 0.2)" }
        : buildSimilarityPalette(obj.templateSimilarity)
      const label = describeDetectionLabel(obj)
      drawRect(
        rect,
        palette.stroke,
        palette.fill,
        !isPrimary,
        label
      )
    })
  }, [currentPage])

  useEffect(() => {
    drawOverlay()
  }, [drawOverlay, selectionBox, detectedObjects, selectedObject])

  useEffect(() => {
    if (!currentPage || !osdReady || !viewerContainerRef.current || !osdModuleRef.current) return

    setViewerReady(false)
    destroyViewer()
    const viewer = osdModuleRef.current({
      element: viewerContainerRef.current,
      prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
      tileSources: { type: "image", url: currentPage.imageSrc },
      showNavigationControl: false,
      showNavigator: false,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: true,
        dragToPan: true,
      },
      maxZoomPixelRatio: 6,
      minZoomLevel: 0.4,
      visibilityRatio: 1,
      constrainDuringPan: true,
      zoomPerScroll: 1.25,
    })

    viewerRef.current = viewer
    viewer.addHandler("open", () => {
      setViewerReady(true)
      drawOverlay()
    })
    viewer.addHandler("animation", drawOverlay)
    viewer.addHandler("resize", drawOverlay)
    viewer.addHandler("update-viewport", drawOverlay)

    return () => {
      viewer.destroy()
      viewerRef.current = null
    }
  }, [currentPage, osdReady, destroyViewer, drawOverlay])

  useEffect(() => {
    const viewer = viewerRef.current
    if (!viewer) return
    viewer.setMouseNavEnabled(mode === "pan")
    viewer.gestureSettingsMouse.clickToZoom = mode === "pan"
    viewer.gestureSettingsMouse.dragToPan = mode === "pan"
  }, [mode])

  const ensurePdfjs = useCallback(async () => {
    if (pdfjsRef.current) return pdfjsRef.current
    const pdfjs = await import("pdfjs-dist/build/pdf")
    if (pdfjs.GlobalWorkerOptions && !pdfjs.GlobalWorkerOptions.workerSrc) {
      const version = (pdfjs as { version?: string }).version ?? "4.6.82"
      pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${version}/build/pdf.worker.min.js`
    }
    pdfjsRef.current = pdfjs
    return pdfjs
  }, [])

  const convertImageFile = async (file: File): Promise<DeepZoomPage> => {
    const base64 = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (event) => resolve(event.target?.result as string)
      reader.onerror = () => reject(new Error("Failed to read image"))
      reader.readAsDataURL(file)
    })

    const img = await loadImageElement(base64)
    return {
      id: `${file.name}-${Date.now()}`,
      label: file.name,
      imageSrc: base64,
      width: img.width,
      height: img.height,
      originalFileName: file.name,
    }
  }

  const extractPdfPages = async (file: File): Promise<DeepZoomPage[]> => {
    const pdfjs = await ensurePdfjs()
    const pdfData = await file.arrayBuffer()
    const pdf = await pdfjs.getDocument({ data: pdfData }).promise
    const rendered: DeepZoomPage[] = []

    for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
      const page = await pdf.getPage(pageNumber)
      const viewport = page.getViewport({ scale: 2 })
      const canvas = document.createElement("canvas")
      canvas.width = viewport.width
      canvas.height = viewport.height
      const ctx = canvas.getContext("2d")
      if (!ctx) continue
      await page.render({ canvasContext: ctx, viewport }).promise
      rendered.push({
        id: `${file.name}-page-${pageNumber}-${Date.now()}`,
        label: `${file.name} · Page ${pageNumber}`,
        imageSrc: canvas.toDataURL("image/png"),
        width: viewport.width,
        height: viewport.height,
        originalFileName: file.name,
        pageNumber,
      })
    }
    return rendered
  }

  const handleAssetUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files
    if (!fileList || fileList.length === 0) return
    setIsPreparingPages(true)
    setError(null)
    try {
      const files = Array.from(fileList)
      const loadedPages: DeepZoomPage[] = []
      for (const file of files) {
        if (file.type === "application/pdf") {
          const pdfPages = await extractPdfPages(file)
          loadedPages.push(...pdfPages)
        } else if (file.type.startsWith("image/")) {
          loadedPages.push(await convertImageFile(file))
        } else {
          console.warn("Unsupported file type", file.type)
        }
      }
      if (loadedPages.length === 0) {
        setError("No supported pages were found in the uploaded files")
        return
      }
      setPages(loadedPages)
      setActivePageIndex(0)
      resetDetectionState()
    } catch (err) {
      console.error("Failed to load assets", err)
      const message = err instanceof Error ? err.message : "Unknown error while loading files"
      setError(message)
    } finally {
      setIsPreparingPages(false)
      if (assetInputRef.current) {
        assetInputRef.current.value = ""
      }
    }
  }

  const handleExportSelection = () => {
    if (!currentPage || !selectionBoxRef.current) return
    const box = selectionBoxRef.current
    const payload = {
      image: currentPage.imageSrc,
      imageSize: { width: currentPage.width, height: currentPage.height },
      selection: {
        startX: box.x / currentPage.width,
        startY: box.y / currentPage.height,
        endX: (box.x + box.width) / currentPage.width,
        endY: (box.y + box.height) / currentPage.height,
      },
      pageId: currentPage.id,
      pageLabel: currentPage.label,
      timestamp: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement("a")
    anchor.href = url
    anchor.download = `selection-${Date.now()}.json`
    anchor.click()
    URL.revokeObjectURL(url)
  }

  const handleDownloadAnnotated = () => {
    if (!annotatedImage) return
    const link = document.createElement("a")
    link.href = annotatedImage
    link.download = `annotated-${Date.now()}.png`
    link.click()
  }

  const importFromJSON = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = async (e) => {
      try {
        const content = e.target?.result as string
        const data = JSON.parse(content)
        if (!data.image) throw new Error("Missing image data in file")
        let width = data.imageSize?.width
        let height = data.imageSize?.height
        if (!width || !height) {
          const img = await loadImageElement(data.image)
          width = img.width
          height = img.height
        }
        const importedPage: DeepZoomPage = {
          id: data.pageId || `import-${Date.now()}`,
          label: data.pageLabel || "Imported Page",
          imageSrc: data.image,
          width,
          height,
        }
        setPages([importedPage])
        setActivePageIndex(0)

        if (data.selection) {
          const startX = data.selection.startX ?? data.selection.x ?? 0
          const startY = data.selection.startY ?? data.selection.y ?? 0
          const endX = data.selection.endX ?? startX + (data.selection.width ?? 0)
          const endY = data.selection.endY ?? startY + (data.selection.height ?? 0)
          const rect: SelectionRect = {
            x: startX * width,
            y: startY * height,
            width: Math.max(1, (endX - startX) * width),
            height: Math.max(1, (endY - startY) * height),
          }
          selectionBoxRef.current = rect
          setSelectionBox(rect)
        } else {
          resetDetectionState()
        }
      } catch (err) {
        console.error("Failed to import selection", err)
        alert("Could not import selection file")
      } finally {
        if (jsonInputRef.current) jsonInputRef.current.value = ""
      }
    }
    reader.readAsText(file)
  }

  const getImagePoint = (event: React.PointerEvent<HTMLDivElement>) => {
    const viewer = viewerRef.current
    const page = currentPage
    if (!viewer || !page || !interactionLayerRef.current || !osdModuleRef.current) return null
    const rect = interactionLayerRef.current.getBoundingClientRect()
    const localX = clamp(event.clientX - rect.left, 0, rect.width)
    const localY = clamp(event.clientY - rect.top, 0, rect.height)
    const Point = osdModuleRef.current.Point
    const viewportPoint = viewer.viewport.pointFromPixel(new Point(localX, localY))
    const imagePoint = viewer.viewport.viewportToImageCoordinates(viewportPoint)
    return {
      x: clamp(imagePoint.x, 0, page.width),
      y: clamp(imagePoint.y, 0, page.height),
    }
  }

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (mode !== "select" || !currentPage) return
    event.preventDefault()
    const start = getImagePoint(event)
    if (!start) return
    selectionStartRef.current = start
    const rect: SelectionRect = { x: start.x, y: start.y, width: 0, height: 0 }
    selectionBoxRef.current = rect
    setSelectionBox(rect)
    setIsSelecting(true)
    interactionLayerRef.current?.setPointerCapture(event.pointerId)
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isSelecting || !selectionStartRef.current) return
    const current = getImagePoint(event)
    if (!current) return
    const start = selectionStartRef.current
    const rect: SelectionRect = {
      x: Math.min(start.x, current.x),
      y: Math.min(start.y, current.y),
      width: Math.abs(current.x - start.x),
      height: Math.abs(current.y - start.y),
    }
    selectionBoxRef.current = rect
    setSelectionBox(rect)
  }

  const finalizeSelection = () => {
    setIsSelecting(false)
    selectionStartRef.current = null
    const rect = selectionBoxRef.current
    if (!rect || rect.width < 5 || rect.height < 5 || !currentPage) {
      selectionBoxRef.current = null
      setSelectionBox(null)
      return
    }
    analyzeSelection(currentPage, rect)
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!isSelecting) return
    interactionLayerRef.current?.releasePointerCapture(event.pointerId)
    handlePointerMove(event)
    finalizeSelection()
  }

  const handlePointerLeave = () => {
    if (!isSelecting) return
    finalizeSelection()
  }

  const handleClearSelection = () => {
    selectionBoxRef.current = null
    setSelectionBox(null)
    setAnalysisResult(null)
    setDetectedObjects([])
    setSelectedObject(null)
    setAnnotatedImage(null)
    setUploadedPath(null)
    setMatchStats([])
    setBestSimilarity(null)
  }

  const handleZoom = (direction: "in" | "out") => {
    const viewer = viewerRef.current
    if (!viewer) return
    const factor = direction === "in" ? 1.3 : 1 / 1.3
    viewer.viewport.zoomBy(factor)
    viewer.viewport.applyConstraints()
  }

  const analyzeSelection = async (page: DeepZoomPage, rect: SelectionRect, options?: { forceRefresh?: boolean }) => {
    setIsProcessing(true)
    setError(null)
    setAnalysisResult(null)

    const normalizedSelection = {
      x: rect.x / page.width,
      y: rect.y / page.height,
      width: rect.width / page.width,
      height: rect.height / page.height,
    }

    let selectionCanvas: HTMLCanvasElement | null = null
    let fallbackAttempted = false

    try {
      const img = await loadImageElement(page.imageSrc)
      const canvas = document.createElement("canvas")
      canvas.width = img.width
      canvas.height = img.height
      const ctx = canvas.getContext("2d")
      if (!ctx) throw new Error("Failed to create canvas context")
      ctx.drawImage(img, 0, 0, img.width, img.height)

      const imageData = ctx.getImageData(rect.x, rect.y, rect.width, rect.height)
      selectionCanvas = document.createElement("canvas")
      selectionCanvas.width = rect.width
      selectionCanvas.height = rect.height
      const tempCtx = selectionCanvas.getContext("2d")
      if (!tempCtx) throw new Error("Failed to create temp canvas")
      tempCtx.putImageData(imageData, 0, 0)

      const serverImagePayload = page.imageSrc

      if (skipCloudVisionRef.current && selectionCanvas) {
        fallbackAttempted = true
        const fallbackResult = await runLocalDetection(selectionCanvas, page, rect, cloudVisionIssueRef.current)
        if (fallbackResult.success) return
        throw new Error(`Local detection failed: ${fallbackResult.error}`)
      }

      setAnnotatedImage(null)
      setUploadedPath(null)

      const response = await fetch("/api/match-selection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: serverImagePayload,
          coordinates: normalizedSelection,
          options: {
            label: page.label,
            mode: "texture", // or "texture" / "auto"
          },
        }),
      })

      if (!response.ok) {
        const { message, details } = await readApiError(response)
        console.error("API Error Details:", formatErrorDetails(details))
        throw new Error(message)
      }

      const result = await response.json()
      if (!result || result.success === false) {
        throw new Error(result?.message || "Template matching failed")
      }

      setUploadedPath(null)
      setAnnotatedImage(null)

      const objects = mapServerDetections(result.matches)
      setDetectedObjects(objects)
      setSelectedObject(objects[0] ?? null)
      setMatchStats(Array.isArray(result.metadata?.passStats) ? result.metadata.passStats : [])
      setBestSimilarity(
        typeof result.metadata?.bestSimilarity === "number" ? result.metadata.bestSimilarity : null
      )

      const summaryChunks: string[] = []
      summaryChunks.push(`Template matching found ${objects.length} region${objects.length === 1 ? "" : "s"}.`)
      if (typeof result.metadata?.bestSimilarity === "number") {
        summaryChunks.push(`Best similarity ${(result.metadata.bestSimilarity * 100).toFixed(1)}%`)
      } else if (typeof objects[0]?.templateSimilarity === "number") {
        summaryChunks.push(`Top similarity ${(objects[0].templateSimilarity * 100).toFixed(1)}%`)
      }
      if (objects.length === 0) {
        summaryChunks.push("No confident matches were found.")
      }
      setAnalysisResult(summaryChunks.join("\n"))
    } catch (err) {
      console.error("Error analyzing selection", err)
      const message = err instanceof Error ? err.message : "Unknown error"
      const disableCloudVision = shouldDisableCloudVision(message)
      if (disableCloudVision) {
        skipCloudVisionRef.current = true
        cloudVisionIssueRef.current = message
      }

      if (!fallbackAttempted && selectionCanvas) {
        fallbackAttempted = true
        const fallbackResult = await runLocalDetection(selectionCanvas, page, rect, message)
        if (fallbackResult.success) return
        const fallbackError = fallbackResult.error ? ` Local fallback error: ${fallbackResult.error}` : ""
        setError(`Failed to analyze the selected area: ${message}.${fallbackError}`)
      } else {
        setError(`Failed to analyze the selected area: ${message}`)
      }
      setAnnotatedImage(null)
      setUploadedPath(null)
      setMatchStats([])
      setBestSimilarity(null)
    } finally {
      setIsProcessing(false)
    }
  }

  const handleAnalyzeAgain = () => {
    if (!selectionBoxRef.current || !currentPage) return
    analyzeSelection(currentPage, selectionBoxRef.current, { forceRefresh: true })
  }

  const handleResetWorkspace = () => {
    destroyViewer()
    setPages([])
    setActivePageIndex(0)
    resetDetectionState()
    setError(null)
    skipCloudVisionRef.current = false
    cloudVisionIssueRef.current = null
    if (assetInputRef.current) assetInputRef.current.value = ""
  }

  const similarObjects = useMemo(() => {
    return getTemplateMatches(detectedObjects, selectedObject)
  }, [detectedObjects, selectedObject])

  const renderSimilarityBadge = useCallback(
    (detection?: DetectedObject | null) => {
      if (!detection) return null
      const label = formatSimilarityPercent(detection.templateSimilarity)
      if (!label) return null
      const palette = buildSimilarityPalette(detection.templateSimilarity)
      return (
        <span
          className="rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide"
          style={{
            borderColor: palette.stroke,
            backgroundColor: palette.badgeBg,
            color: palette.text,
          }}
        >
          {label}
        </span>
      )
    },
    []
  )

  return (
    <div className="space-y-6">
      <Card className="p-4 space-y-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex flex-wrap gap-2">
            <Button
              variant={mode === "pan" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("pan")}
              className="flex items-center gap-2"
            >
              <Hand className="h-4 w-4" />
              Pan
            </Button>
            <Button
              variant={mode === "select" ? "default" : "outline"}
              size="sm"
              onClick={() => setMode("select")}
              className="flex items-center gap-2"
            >
              <MousePointer2 className="h-4 w-4" />
              Select
            </Button>
            <Button variant="outline" size="sm" onClick={handleResetWorkspace} className="flex items-center gap-2">
              <Trash2 className="h-4 w-4" />
              Reset Workspace
            </Button>
          </div>
          <div className="flex flex-wrap gap-2">
            <input
              ref={assetInputRef}
              type="file"
              accept="application/pdf,image/*"
              multiple
              onChange={handleAssetUpload}
              className="hidden"
            />
            <Button asChild variant="outline" size="sm">
              <label htmlFor="asset-upload" className="cursor-pointer flex items-center gap-2" onClick={() => assetInputRef.current?.click()}>
                <Upload className="h-4 w-4" />
                Upload PDF / Image
              </label>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportSelection}
              disabled={!selectionBox || !currentPage}
              className="flex items-center gap-2"
            >
              <FileJson className="h-4 w-4" />
              Export Selection
            </Button>
            <input
              ref={jsonInputRef}
              type="file"
              accept="application/json"
              onChange={importFromJSON}
              className="hidden"
            />
            <Button
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
              onClick={() => jsonInputRef.current?.click()}
            >
              <Upload className="h-4 w-4" />
              Import Selection
            </Button>
          </div>
        </div>

        {pages.length > 0 && (
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setActivePageIndex((idx) => Math.max(0, idx - 1))}
              disabled={activePageIndex === 0}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Select value={String(activePageIndex)} onValueChange={(value) => setActivePageIndex(Number(value))}>
              <SelectTrigger className="w-60">
                <SelectValue placeholder="Select page" />
              </SelectTrigger>
              <SelectContent>
                {pages.map((page, index) => (
                  <SelectItem key={page.id} value={String(index)}>
                    {page.label || `Page ${index + 1}`}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setActivePageIndex((idx) => Math.min(pages.length - 1, idx + 1))}
              disabled={activePageIndex === pages.length - 1}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" onClick={() => handleZoom("out")} disabled={!viewerReady}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon" onClick={() => handleZoom("in")} disabled={!viewerReady}>
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
          </div>
        )}
      </Card>

      <Card className="overflow-hidden">
        <div className="relative min-h-[520px]">
          {!currentPage && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted-foreground">
              <Upload className="h-10 w-10" />
              <p className="text-sm">Upload a PDF or large image to start annotating just like Bluebeam Revu.</p>
            </div>
          )}

          <div ref={viewerContainerRef} className="relative h-[640px] w-full bg-muted" />
          <canvas ref={overlayCanvasRef} className="pointer-events-none absolute inset-0" />
          <div
            ref={interactionLayerRef}
            className="absolute inset-0"
            style={{ pointerEvents: mode === "select" ? "auto" : "none", cursor: mode === "select" ? "crosshair" : "default" }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
          />

          {(!viewerReady || isPreparingPages) && currentPage && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/70">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="text-xs text-muted-foreground">
                {isPreparingPages ? "Preparing high-resolution tiles..." : "Loading viewer..."}
              </p>
            </div>
          )}

          {selectionBox && (
            <div className="absolute right-4 top-4 z-20 flex flex-col gap-2">
              <Button variant="outline" size="sm" onClick={handleClearSelection} className="flex items-center gap-2">
                <X className="h-4 w-4" />
                Clear Selection
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleAnalyzeAgain}
                disabled={isProcessing}
                className="flex items-center gap-2"
              >
                {isProcessing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                {isProcessing ? "Analyzing..." : "Analyze"}
              </Button>
            </div>
          )}
        </div>
      </Card>

      {analysisResult && (
        <Card className="space-y-4 p-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Analysis</h3>
            {isProcessing && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
          </div>
          <p className="whitespace-pre-line text-sm text-muted-foreground">{analysisResult}</p>

          {bestSimilarity !== null && (
            <p className="text-xs text-muted-foreground">
              Best similarity {(bestSimilarity * 100).toFixed(1)}% across all passes.
            </p>
          )}

          {matchStats.length > 0 && (
            <div className="space-y-2 rounded-md border border-dashed border-primary/30 p-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-primary">Template Pass Diagnostics</span>
                <span className="text-muted-foreground">{matchStats.length} pass{matchStats.length === 1 ? "" : "es"}</span>
              </div>
              <div className="space-y-1">
                {matchStats.map((stat) => (
                  <div key={stat.pass} className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-col">
                      <span className="font-medium text-foreground">{stat.pass}</span>
                      <span className="text-muted-foreground">
                        {formatSimilarityPercent(stat.bestSimilarity) ?? "No hit"} · {stat.kept}/{stat.candidates} windows
                      </span>
                    </div>
                    <span className="font-semibold text-foreground/80">{formatMs(stat.durationMs)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {annotatedImage && (
            <div className="space-y-2 rounded-md border border-primary/30 p-2">
              <img src={annotatedImage} alt="Annotated detections" className="w-full rounded-md border bg-white" />
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
                <span>{uploadedPath ? `Server copy: ${uploadedPath}` : "Inline preview"}</span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 gap-1"
                  onClick={handleDownloadAnnotated}
                >
                  <Download className="h-3.5 w-3.5" />
                  Download PNG
                </Button>
              </div>
            </div>
          )}

          {selectedObject && (
            <div>
              <h4 className="text-sm font-medium">Selected Object</h4>
              <div className="mt-2 rounded-md border border-blue-200 bg-blue-50 p-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="font-semibold capitalize">{selectedObject.name}</span>
                  <div className="flex items-center gap-2">
                    <span className="text-blue-700">{Math.round(selectedObject.score * 100)}% confidence</span>
                    {renderSimilarityBadge(selectedObject)}
                  </div>
                </div>
                {typeof selectedObject.templateSimilarity === "number" && (
                  <p className="mt-1 text-xs text-blue-900/80">
                    Matches {(selectedObject.templateSimilarity * 100).toFixed(1)}% of the selected patch.
                  </p>
                )}
              </div>
            </div>
          )}

          {similarObjects.length > 0 && (
            <div>
              <h4 className="text-sm font-medium">Similar Objects</h4>
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {similarObjects.map((obj, index) => {
                  const palette = buildSimilarityPalette(obj.templateSimilarity)
                  return (
                    <div
                      key={`${obj.name}-${index}`}
                      className="rounded-md border p-2 text-sm"
                      style={{ borderColor: palette.stroke, backgroundColor: palette.panelBg }}
                    >
                    <div className="flex items-center justify-between gap-2">
                      <span className="capitalize">{obj.name}</span>
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-emerald-700">{Math.round(obj.score * 100)}% conf</span>
                        {renderSimilarityBadge(obj)}
                      </div>
                    </div>
                    {typeof obj.templateSimilarity === "number" && (
                      <p className="mt-1 text-xs text-emerald-900/80">
                        Aligns {(obj.templateSimilarity * 100).toFixed(1)}% with the selection.
                      </p>
                    )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </Card>
      )}

      {error && <p className="text-sm text-red-500">{error}</p>}
    </div>
  )
}