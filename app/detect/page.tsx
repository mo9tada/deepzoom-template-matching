import DetectionCanvas from "@/components/detection-canvas"
import { Sparkles } from "lucide-react"

export default function DetectPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <header className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <Sparkles className="w-8 h-8 text-primary" />
            <h1 className="text-4xl font-bold tracking-tight text-balance">Object Detection Tool</h1>
          </div>
          <p className="text-muted-foreground text-lg text-pretty">
            Upload an image, select an object, and watch AI find similar items
          </p>
        </header>

        <DetectionCanvas />
      </div>
    </main>
  )
}
