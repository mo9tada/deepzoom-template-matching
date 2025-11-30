export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface Detection {
  label: string
  confidence: number
  boundingBox: BoundingBox
}
