export interface SelectionRect {
  x: number
  y: number
  width: number
  height: number
}

export const clampNormalized = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export const normalizeSelection = (coordinates?: Partial<SelectionRect>): SelectionRect | null => {
  if (!coordinates) return null
  const x = clampNormalized(coordinates.x ?? 0, 0, 1)
  const y = clampNormalized(coordinates.y ?? 0, 0, 1)
  const width = clampNormalized(coordinates.width ?? 1, 0, 1)
  const height = clampNormalized(coordinates.height ?? 1, 0, 1)
  return {
    x,
    y,
    width: Math.min(1 - x, width),
    height: Math.min(1 - y, height),
  }
}
