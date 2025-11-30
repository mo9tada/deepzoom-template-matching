import type { Metadata } from 'next'
import { GeistSans } from 'geist/font/sans'
import { GeistMono } from 'geist/font/mono'
import { Analytics } from '@vercel/analytics/next'
import Link from 'next/link'
import { Sparkles } from 'lucide-react'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI Object Detection',
  description: 'Upload an image, select an object, and watch AI find similar items',
  generator: 'v0.app',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
          <nav className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
                <Sparkles className="w-6 h-6 text-primary" />
                <span className="font-bold text-lg">AI Detection</span>
              </Link>
              <div className="flex items-center gap-6">
                <Link href="/" className="text-sm font-medium hover:text-primary transition-colors">
                  Home
                </Link>
                <Link href="/detect" className="text-sm font-medium hover:text-primary transition-colors">
                  Detect
                </Link>
                <Link href="/multi-viewer" className="text-sm font-medium hover:text-primary transition-colors">
                  Multi-Viewer
                </Link>
                <Link href="/about" className="text-sm font-medium hover:text-primary transition-colors">
                  About
                </Link>
              </div>
            </div>
          </nav>
        </header>
        {children}
        <Analytics />
      </body>
    </html>
  )
}
