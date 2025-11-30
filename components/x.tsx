"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, Loader2, Trash2, Sparkles, Hand, MousePointer2, Download, FileJson } from "lucide-react"
import { cn } from "@/lib/utils"

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
}

interface SelectionBox {
  startX: number
  startY: number
  endX: number
  endY: number
}

export default function DetectionCanvas() {
  const [image, setImage] = useState<string | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [detections, setDetections] = useState<BoundingBox[]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [selectionBox, setSelectionBox] = useState<SelectionBox | null>(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [osdLoaded, setOsdLoaded] = useState(false)
  const [mode, setMode] = useState<"pan" | "select">("pan")

  const viewerRef = useRef<any>(null)
  const viewerContainerRef = useRef<HTMLDivElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const jsonInputRef = useRef<HTMLInputElement>(null)
  const mouseTrackerRef = useRef<any>(null)
  const isMouseDownRef = useRef(false)
  const isDrawingRef = useRef(false)
  const selectionBoxRef = useRef<SelectionBox | null>(null)

  // Load OpenSeadragon
  useEffect(() => {
    if (typeof window !== "undefined" && !window.OpenSeadragon) {
      const script = document.createElement("script")
      script.src = "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.js"
      script.async = true
      script.onload = () => {
        const link = document.createElement("link")
        link.rel = "stylesheet"
        link.href = "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/openseadragon.min.css"
        document.head.appendChild(link)
        setOsdLoaded(true)
      }
      document.body.appendChild(script)
    } else if (window.OpenSeadragon) {
      setOsdLoaded(true)
    }
  }, [])

  // Initialize viewer when image or OSD is loaded
  useEffect(() => {
    if (!image || !osdLoaded || !viewerContainerRef.current) return

    const initViewer = () => {
      if (viewerRef.current) {
        viewerRef.current.destroy()
      }

      viewerRef.current = window.OpenSeadragon({
        element: viewerContainerRef.current,
        prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1.0/build/openseadragon/images/",
        tileSources: {
          type: "image",
          url: image,
        },
        showNavigationControl: true,
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
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

      // Set up mouse tracking for selection
      mouseTrackerRef.current = new window.OpenSeadragon.MouseTracker({
        element: viewerRef.current.canvas,
        pressHandler: (event: any) => {
          if (mode !== "select" || isDetecting) return
          isMouseDownRef.current = true
          const viewportPoint = viewerRef.current.viewport.pointFromPixel(event.position)
          const pixelPoint = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportPoint)
          selectionBoxRef.current = {
            startX: pixelPoint.x,
            startY: pixelPoint.y,
            endX: pixelPoint.x,
            endY: pixelPoint.y,
          }
          setSelectionBox(selectionBoxRef.current)
          setDetections([])
        },
        dragHandler: (event: any) => {
          if (mode !== "select" || !isMouseDownRef.current || !selectionBoxRef.current) return
          const viewportPoint = viewerRef.current.viewport.pointFromPixel(event.position)
          const pixelPoint = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportPoint)
          selectionBoxRef.current = {
            ...selectionBoxRef.current,
            endX: pixelPoint.x,
            endY: pixelPoint.y,
          }
          setSelectionBox(selectionBoxRef.current)
        },
        releaseHandler: () => {
          if (mode !== "select" || !isMouseDownRef.current || !selectionBoxRef.current) return
          isMouseDownRef.current = false
          setIsDrawing(false)
          detectObjects()
        },
      })
    }

    initViewer()

    return () => {
      if (viewerRef.current) {
        viewerRef.current.destroy()
        viewerRef.current = null
      }
      if (mouseTrackerRef.current) {
        mouseTrackerRef.current.destroy()
        mouseTrackerRef.current = null
      }
    }
  }, [image, osdLoaded, mode, isDetecting])

  // Update canvas overlay when needed
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!viewerRef.current || !canvas) return

    const drawOverlay = () => {
      const ctx = canvas.getContext("2d")
      if (!ctx || !viewerRef.current) return

      const containerSize = viewerRef.current.viewport.getContainerSize()
      if (!containerSize) return
      
      canvas.width = containerSize.x
      canvas.height = containerSize.y
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      // Draw selection box
      if (selectionBox) {
        const { startX, startY, endX, endY } = selectionBox
        const width = endX - startX
        const height = endY - startY

        ctx.fillStyle = "rgba(96, 165, 250, 0.1)"
        ctx.fillRect(startX, startY, width, height)

        ctx.strokeStyle = "#60a5fa"
        ctx.lineWidth = 2
        ctx.setLineDash([5, 5])
        ctx.strokeRect(startX, startY, width, height)
        ctx.setLineDash([])
      }

      // Draw detections
      detections.forEach((box) => {
        const viewportPoint = viewerRef.current.viewport.imageToViewportCoordinates(box.x, box.y)
        const viewportSize = viewerRef.current.viewport.imageToViewportCoordinates(
          box.x + box.width,
          box.y + box.height
        )

        const pixelPoint = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportPoint)
        const pixelSize = viewerRef.current.viewport.viewportToViewerElementCoordinates(viewportSize)

        const x = pixelPoint.x
        const y = pixelPoint.y
        const width = pixelSize.x - pixelPoint.x
        const height = pixelSize.y - pixelPoint.y

        ctx.fillStyle = "rgba(139, 92, 246, 0.15)"
        ctx.fillRect(x, y, width, height)

        ctx.strokeStyle = "#8b5cf6"
        ctx.lineWidth = 2
        ctx.strokeRect(x, y, width, height)

        // Draw label
        ctx.fillStyle = "#8b5cf6"
        ctx.font = "12px Arial"
        ctx.fillText(box.label, x + 5, y + 15)
      })
    }

    drawOverlay()

    const updateOverlay = () => {
      drawOverlay()
    }

    if (viewerRef.current) {
      viewerRef.current.addHandler("animation", updateOverlay)
      viewerRef.current.addHandler("resize", updateOverlay)
      viewerRef.current.addHandler("update-viewport", updateOverlay)
    }

    return () => {
      if (viewerRef.current) {
        viewerRef.current.removeHandler("animation", updateOverlay)
        viewerRef.current.removeHandler("resize", updateOverlay)
        viewerRef.current.removeHandler("update-viewport", updateOverlay)
      }
    }
  }, [selectionBox, detections])

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === "image/png") {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setImage(result)
        setDetections([])
        setSelectionBox(null)
        selectionBoxRef.current = null
        isMouseDownRef.current = false
      }
      reader.readAsDataURL(file)
    } else {
      alert("Please upload a PNG file")
    }
  }

  const detectObjects = async () => {
    if (!selectionBoxRef.current || !image) return

    setIsDetecting(true)

    try {
      // Simulate detection (replace with actual detection logic)
      await new Promise((resolve) => setTimeout(resolve, 1000))

      const newDetections: BoundingBox[] = [
        {
          x: Math.random() * 0.8,
          y: Math.random() * 0.8,
          width: 0.1 + Math.random() * 0.2,
          height: 0.1 + Math.random() * 0.2,
          label: "Object",
          confidence: 0.8 + Math.random() * 0.2,
        },
        {
          x: Math.random() * 0.8,
          y: Math.random() * 0.8,
          width: 0.1 + Math.random() * 0.2,
          height: 0.1 + Math.random() * 0.2,
          label: "Object",
          confidence: 0.8 + Math.random() * 0.2,
        },
      ]

      setDetections(newDetections)
    } catch (error) {
      console.error("Detection error:", error)
    } finally {
      setIsDetecting(false)
      setSelectionBox(null)
      selectionBoxRef.current = null
    }
  }

  const exportToJSON = () => {
    if (!image) return

    const data = {
      image: image,
      detections: detections,
      imageSize: imageSize,
      timestamp: new Date().toISOString(),
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `detections-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const importFromJSON = (e: React.ChangeEvent<HTMLInputElement>) => {
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

        if (data.image && data.detections && data.imageSize) {
          setImage(data.image)
          setDetections(data.detections)
          setImageSize(data.imageSize)
        } else {
          alert("Invalid JSON format. Missing required fields.")
        }
      } catch (error) {
        console.error("JSON parse error:", error)
        alert("Failed to parse JSON file")
      }
    }
    reader.readAsText(file)
  }

  const handleReset = () => {
    setImage(null)
    setDetections([])
    setSelectionBox(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    if (jsonInputRef.current) {
      jsonInputRef.current.value = ""
    }
  }

  const toggleMode = () => {
    setMode((prev) => (prev === "pan" ? "select" : "pan"))
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Image Detection</h2>
        <div className="flex items-center space-x-2">
          <Button
            variant={mode === "select" ? "default" : "outline"}
            size="sm"
            onClick={toggleMode}
            disabled={!image || isDetecting}
          >
            {mode === "pan" ? (
              <>
                <MousePointer2 className="w-4 h-4 mr-2" />
                Select Mode
              </>
            ) : (
              <>
                <Hand className="w-4 h-4 mr-2" />
                Pan Mode
              </>
            )}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={exportToJSON}
            disabled={!image || isDetecting}
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
          <input
            ref={jsonInputRef}
            type="file"
            accept="application/json,.json"
            onChange={importFromJSON}
            className="hidden"
            id="json-upload"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => jsonInputRef.current?.click()}
            disabled={isDetecting}
          >
            <FileJson className="w-4 h-4 mr-2" />
            Import
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            disabled={isDetecting}
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Reset
          </Button>
        </div>
      </div>

      {!image ? (
        <div className="flex flex-col items-center justify-center p-12 border-2 border-dashed border-gray-300 rounded-lg">
          <Upload className="w-12 h-12 text-gray-400 mb-4" />
          <p className="text-gray-500 mb-4">Upload an image to get started</p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png"
            onChange={handleImageUpload}
            className="hidden"
            id="image-upload"
          />
          <Button asChild>
            <label htmlFor="image-upload" className="cursor-pointer">
              <Upload className="w-4 h-4 mr-2" />
              Upload PNG Image
            </label>
          </Button>
        </div>
      ) : (
        <div className="relative w-full h-[600px] border rounded-lg overflow-hidden">
          <div
            ref={viewerContainerRef}
            className="w-full h-full"
            style={{ backgroundColor: "#f3f4f6" }}
          />
          <canvas
            ref={overlayCanvasRef}
            className="absolute top-0 left-0 w-full h-full pointer-events-none"
          />
          {isDetecting && (
            <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center">
              <div className="bg-white p-4 rounded-lg flex items-center space-x-2">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
                <span>Detecting objects...</span>
              </div>
            </div>
          )}
        </div>
      )}

      {detections.length > 0 && (
        <Card className="p-4">
          <h3 className="font-medium mb-2">Detected Objects ({detections.length})</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {detections.map((detection, index) => (
              <div
                key={index}
                className="p-2 border rounded text-sm"
              >
                <div className="font-medium">{detection.label}</div>
                <div>Confidence: {(detection.confidence * 100).toFixed(1)}%</div>
                <div>
                  Position: ({detection.x.toFixed(2)}, {detection.y.toFixed(2)})
                </div>
                <div>
                  Size: {detection.width.toFixed(2)} × {detection.height.toFixed(2)}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}