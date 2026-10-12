/**
 * A test image in [0, 1], row-major with the first row at the top: a smooth horizontal gradient, a disc, a bar, a dark
 * diagonal band and a checkerboard patch. The smooth and axis-aligned parts are low rank; the diagonal band and the
 * checkerboard need many components.
 */
export function testImage(size = 64): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) => {
      const x = col / (size - 1)
      const y = row / (size - 1)
      if (x > 0.6 && y > 0.6 && x < 0.9 && y < 0.9) return (Math.floor(x * 16) + Math.floor(y * 16)) % 2
      if (Math.abs(x - y - 0.25) < 0.04) return 0.05
      if (x > 0.6 && x < 0.9 && y > 0.15 && y < 0.3) return 0.85
      if ((x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.04) return 0.95
      return 0.25 + 0.2 * x
    }),
  )
}
