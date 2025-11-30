import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Sparkles, Brain, Scan, Layers } from "lucide-react"

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        {/* Header */}
        <section className="text-center mb-16">
          <div className="flex items-center justify-center gap-3 mb-6">
            <Brain className="w-12 h-12 text-primary" />
            <h1 className="text-5xl font-bold tracking-tight">About This Tool</h1>
          </div>
          <p className="text-muted-foreground text-xl max-w-2xl mx-auto">
            Learn how our AI-powered object detection technology works
          </p>
        </section>

        {/* How It Works */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">How It Works</h2>
          <div className="space-y-6">
            <Card className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-primary/10 shrink-0">
                  <Layers className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">1. Upload Your Image</h3>
                  <p className="text-muted-foreground">
                    Start by uploading a high-resolution PNG image. Our viewer uses OpenSeadragon 
                    technology to handle large images with smooth pan and zoom capabilities.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-primary/10 shrink-0">
                  <Scan className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">2. Select an Object</h3>
                  <p className="text-muted-foreground">
                    Switch to select mode and click-and-drag to draw a box around any object 
                    in your image. This becomes your reference object for detection.
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-start gap-4">
                <div className="p-3 rounded-lg bg-primary/10 shrink-0">
                  <Brain className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">3. AI Detection</h3>
                  <p className="text-muted-foreground">
                    Our algorithm analyzes the color histogram of your selection and scans 
                    the entire image using a sliding window approach. It compares each region 
                    using correlation coefficients and applies non-maximum suppression to 
                    eliminate overlapping detections.
                  </p>
                </div>
              </div>
            </Card>
          </div>
        </section>

        {/* Technology Stack */}
        <section className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">Technology Stack</h2>
          <Card className="p-8">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-2 text-lg">Frontend</h3>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• Next.js 14 with App Router</li>
                  <li>• React 18 with TypeScript</li>
                  <li>• Tailwind CSS for styling</li>
                  <li>• shadcn/ui components</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2 text-lg">Detection Engine</h3>
                <ul className="space-y-2 text-muted-foreground">
                  <li>• OpenSeadragon for image viewing</li>
                  <li>• Color histogram analysis</li>
                  <li>• Sliding window detection</li>
                  <li>• Non-maximum suppression</li>
                </ul>
              </div>
            </div>
          </Card>
        </section>

        {/* CTA */}
        <section className="text-center">
          <Card className="p-12 bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20">
            <h2 className="text-3xl font-bold mb-4">Ready to try it?</h2>
            <p className="text-muted-foreground mb-6 text-lg">
              Experience the power of AI object detection
            </p>
            <Button asChild size="lg">
              <Link href="/detect">
                <Sparkles className="w-5 h-5 mr-2" />
                Start Detecting
              </Link>
            </Button>
          </Card>
        </section>
      </div>
    </main>
  )
}
