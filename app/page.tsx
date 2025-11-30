import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Sparkles, Image as ImageIcon, Zap, Target, Grid3x3 } from "lucide-react"

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 max-w-6xl">
        {/* Hero Section */}
        <section className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Sparkles className="w-12 h-12 text-primary" />
            <h1 className="text-5xl font-bold tracking-tight">AI Object Detection</h1>
          </div>
          <p className="text-muted-foreground text-xl mb-8 max-w-2xl mx-auto">
            Upload an image, select an object, and watch AI find similar items instantly
          </p>
          <div className="flex items-center justify-center gap-4">
            <Button asChild size="lg">
              <Link href="/detect">
                <Target className="w-5 h-5 mr-2" />
                Start Detecting
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/about">Learn More</Link>
            </Button>
          </div>
        </section>

        {/* Features Section */}
        <section className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-primary/10">
                <ImageIcon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Upload Images</h3>
            </div>
            <p className="text-muted-foreground">
              Upload high-resolution PNG images and explore them with our advanced viewer
            </p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-primary/10">
                <Target className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Select Objects</h3>
            </div>
            <p className="text-muted-foreground">
              Click and drag to select any object in your image with precision
            </p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-primary/10">
                <Zap className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">AI Detection</h3>
            </div>
            <p className="text-muted-foreground">
              Our AI instantly finds and highlights similar objects across your image
            </p>
          </Card>

          <Card className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-lg bg-primary/10">
                <Grid3x3 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">Multi-Viewer</h3>
            </div>
            <p className="text-muted-foreground">
              Compare multiple images side by side with flexible grid layouts
            </p>
          </Card>
        </section>

        {/* CTA Section */}
        <section className="text-center">
          <Card className="p-12 bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
            <h2 className="text-3xl font-bold mb-4">Ready to get started?</h2>
            <p className="text-muted-foreground mb-6 text-lg">
              Try our AI-powered object detection tool now
            </p>
            <Button asChild size="lg">
              <Link href="/detect">
                <Sparkles className="w-5 h-5 mr-2" />
                Launch Detection Tool
              </Link>
            </Button>
          </Card>
        </section>
      </div>
    </main>
  )
}
