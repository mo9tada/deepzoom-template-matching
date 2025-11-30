"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Plus, Grid3x3, LayoutGrid, Columns2 } from "lucide-react"
import MiniViewer from "@/components/mini-viewer"

export default function MultiViewerPage() {
  const [viewers, setViewers] = useState<string[]>(["viewer-1", "viewer-2"])
  const [layout, setLayout] = useState<"2" | "3" | "4">("2")

  const addViewer = () => {
    const newId = `viewer-${Date.now()}`
    setViewers([...viewers, newId])
  }

  const removeViewer = (id: string) => {
    if (viewers.length > 1) {
      setViewers(viewers.filter((v) => v !== id))
    }
  }

  const getGridCols = () => {
    switch (layout) {
      case "2":
        return "md:grid-cols-2"
      case "3":
        return "md:grid-cols-3"
      case "4":
        return "md:grid-cols-2 lg:grid-cols-4"
      default:
        return "md:grid-cols-2"
    }
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-4xl font-bold tracking-tight mb-2">Multi-Viewer</h1>
              <p className="text-muted-foreground">
                View and compare multiple images side by side
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant={layout === "2" ? "default" : "outline"}
                size="sm"
                onClick={() => setLayout("2")}
                title="2 columns"
              >
                <Columns2 className="w-4 h-4" />
              </Button>
              <Button
                variant={layout === "3" ? "default" : "outline"}
                size="sm"
                onClick={() => setLayout("3")}
                title="3 columns"
              >
                <Grid3x3 className="w-4 h-4" />
              </Button>
              <Button
                variant={layout === "4" ? "default" : "outline"}
                size="sm"
                onClick={() => setLayout("4")}
                title="4 columns"
              >
                <LayoutGrid className="w-4 h-4" />
              </Button>
              <Button onClick={addViewer} size="sm">
                <Plus className="w-4 h-4 mr-2" />
                Add Viewer
              </Button>
            </div>
          </div>
        </header>

        <div className={`grid grid-cols-1 ${getGridCols()} gap-6`}>
          {viewers.map((id) => (
            <MiniViewer
              key={id}
              id={id}
              onRemove={() => removeViewer(id)}
              showRemove={viewers.length > 1}
            />
          ))}
        </div>

        {viewers.length === 0 && (
          <Card className="p-12 text-center">
            <p className="text-muted-foreground mb-4">No viewers available</p>
            <Button onClick={addViewer}>
              <Plus className="w-4 h-4 mr-2" />
              Add First Viewer
            </Button>
          </Card>
        )}
      </div>
    </main>
  )
}
