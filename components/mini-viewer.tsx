"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, Loader2, Trash2, X } from "lucide-react"

declare global {
  interface Window {
    OpenSeadragon: any
  }
}

interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
  label: string
  confidence: number
  templateSimilarity?: number | null
}

interface MiniViewerProps {
  id: string
  onRemove?: () => void
  showRemove?: boolean
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const normalizeSimilarity = (similarity?: number | null) => {
  if (typeof similarity !== "number" || Number.isNaN(similarity)) return null
  return clamp((similarity + 1) / 2, 0, 1)
}

const buildSimilarityPalette = (similarity?: number | null) => {
  const normalized = normalizeSimilarity(similarity)
  if (normalized === null) {
    return {
      stroke: "#8b5cf6",
      fill: "rgba(139, 92, 246, 0.15)",
    }
  }
  const hue = 12 + normalized * 110
  return {
    stroke: `hsl(${hue}, 78%, 46%)`,
    fill: `hsla(${hue}, 78%, 46%, 0.2)`,
  }
}

const describeDetectionLabel = (box: BoundingBox) => {
  const similarity =
    typeof box.templateSimilarity === "number" ? `${(box.templateSimilarity * 100).toFixed(1)}% sim` : null
  const confidence = `${Math.round(box.confidence * 100)}% conf`
  return `${box.label} · ${similarity ?? confidence}`
}

export default function MiniViewer({ id, onRemove, showRemove = true }: MiniViewerProps) {
  const [image, setImage] = useState<string | null>(null)
  const [detections, setDetections] = useState<BoundingBox[]>([])
  const [osdLoaded, setOsdLoaded] = useState(false)

  const viewerRef = useRef<any>(null)
  const viewerContainerRef = useRef<HTMLDivElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (typeof window !== "undefined" && !window.OpenSeadragon) {
      const script = document.createElement("script")
      script.src = "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js"
      script.async = true
      script.onload = () => {
        setOsdLoaded(true)
      }
      document.body.appendChild(script)

      const link = document.createElement("link")
      link.rel = "stylesheet"
      link.href = "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.css"
      document.head.appendChild(link)
    } else if (window.OpenSeadragon) {
      setOsdLoaded(true)
    }
  }, [])

  useEffect(() => {
    if (image && osdLoaded && viewerContainerRef.current && window.OpenSeadragon) {
      if (viewerRef.current) {
        viewerRef.current.destroy()
      }

      const container = viewerContainerRef.current
      if (container.clientWidth === 0 || container.clientHeight === 0) {
        return
      }

      try {
        viewerRef.current = window.OpenSeadragon({
          element: viewerContainerRef.current,
          prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
          tileSources: {
            type: "image",
            url: image,
          },
          showNavigationControl: false,
          showNavigator: false,
          animationTime: 0.5,
          blendTime: 0.1,
          constrainDuringPan: true,
          maxZoomPixelRatio: 2,
          minZoomLevel: 0.8,
          visibilityRatio: 1,
          zoomPerScroll: 1.2,
          gestureSettingsMouse: {
            clickToZoom: false,
            dblClickToZoom: true,
          },
        })

        viewerRef.current.addHandler("animation", drawOverlay)
        viewerRef.current.addHandler("resize", drawOverlay)
        viewerRef.current.addHandler("update-viewport", drawOverlay)
      } catch (error) {
        console.error("[MiniViewer] Error creating viewer:", error)
      }
    }

    return () => {
      if (viewerRef.current) {
        viewerRef.current.destroy()
        viewerRef.current = null
      }
    }
  }, [image, osdLoaded])

  useEffect(() => {
    drawOverlay()
  }, [detections])

  const drawOverlay = () => {
    if (!viewerRef.current || !overlayCanvasRef.current) return

    const canvas = overlayCanvasRef.current
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const containerSize = viewerRef.current.viewport.getContainerSize()
    canvas.width = containerSize.x
    canvas.height = containerSize.y

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    detections.forEach((box) => {
        const palette = buildSimilarityPalette(box.templateSimilarity)
      const viewportPoint = viewerRef.current.viewport.imageToViewportCoordinates(box.x, box.y)
      const viewportSize = viewerRef.current.viewport.imageToViewportCoordinates(box.x + box.width, box.y + box.height)

      const pixelPoint = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportPoint)
      const pixelSize = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportSize)

      const x = pixelPoint.x
      const y = pixelPoint.y
      const width = pixelSize.x - pixelPoint.x
      const height = pixelSize.y - pixelPoint.y

        ctx.fillStyle = palette.fill
      ctx.fillRect(x, y, width, height)

        ctx.strokeStyle = palette.stroke
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, width, height)

        ctx.font = "bold 11px 'Inter', 'Segoe UI', sans-serif"
        const label = describeDetectionLabel(box)
        const metrics = ctx.measureText(label)
        const padding = 6
        const labelWidth = metrics.width + padding * 2
        const labelHeight = 18
        const labelX = Math.max(2, Math.min(x, canvas.width - labelWidth - 2))
        const labelY = y - labelHeight - 4 > 2 ? y - labelHeight - 4 : y + 4

        ctx.fillStyle = palette.stroke
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight)
        ctx.fillStyle = "#fff"
        ctx.textBaseline = "middle"
        ctx.fillText(label, labelX + padding, labelY + labelHeight / 2)
    })
  }

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === "image/png") {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setImage(result)
        setDetections([])
      }
      reader.readAsDataURL(file)
    } else {
      alert("Please upload a PNG file")
    }
  }

  const handleJSONUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || file.type !== "application/json") {
      alert("Please upload a valid JSON file")
      return
    }

    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const result = e.target?.result as string
        const data = JSON.parse(result)

        if (data.image && data.detections) {
          setImage(data.image)
          setDetections(data.detections)
        } else {
          alert("Invalid JSON format. Missing required fields.")
        }
      } catch (error) {
        console.error("[MiniViewer] JSON parse error:", error)
        alert("Failed to parse JSON file")
      }
    }
    reader.readAsText(file)
  }

  const handleReset = () => {
    setImage(null)
    setDetections([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    if (viewerRef.current) {
      viewerRef.current.destroy()
      viewerRef.current = null
    }
  }

  return (
    <Card className="relative">
      {showRemove && onRemove && (
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-2 right-2 z-50 h-8 w-8"
          onClick={onRemove}
        >
          <X className="w-4 h-4" />
        </Button>
      )}
      <div className="p-4">
        {!image ? (
          <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed border-border rounded-lg bg-muted/20">
            <Upload className="w-8 h-8 text-muted-foreground mb-3" />
            <p className="text-muted-foreground mb-3 text-sm text-center">Upload PNG or JSON</p>
            <div className="flex gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png"
                onChange={handleImageUpload}
                className="hidden"
                id={`image-upload-${id}`}
              />
              <Button asChild size="sm" variant="outline">
                <label htmlFor={`image-upload-${id}`} className="cursor-pointer">
                  PNG
                </label>
              </Button>
              <input
                type="file"
                accept="application/json,.json"
                onChange={handleJSONUpload}
                className="hidden"
                id={`json-upload-${id}`}
              />
              <Button asChild size="sm" variant="outline">
                <label htmlFor={`json-upload-${id}`} className="cursor-pointer">
                  JSON
                </label>
              </Button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">
                {detections.length > 0 ? `${detections.length} detection${detections.length !== 1 ? "s" : ""}` : "No detections"}
              </p>
              <Button variant="ghost" size="sm" onClick={handleReset}>
                <Trash2 className="w-3 h-3 mr-1" />
                Clear
              </Button>
            </div>

            <div
              className="relative w-full rounded-lg overflow-hidden border border-border bg-muted/10"
              style={{ height: "300px" }}
            >
              {!osdLoaded && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                </div>
              )}
              <div ref={viewerContainerRef} className="absolute inset-0 w-full h-full" />
              <canvas
                ref={overlayCanvasRef}
                className="absolute inset-0 pointer-events-none"
                style={{ zIndex: 10 }}
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  )
}
