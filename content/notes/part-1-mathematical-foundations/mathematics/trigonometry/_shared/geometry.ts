import { formatNumber, type FilledShape } from 'aifn-render'

/** Plane geometry shared by the trigonometry figures: arcs, sectors and angle formatting. */

export type Vec = [number, number]

export const TAU = 2 * Math.PI

/** An angle in [0, 2π). */
export const wrapPositive = (a: number) => ((a % TAU) + TAU) % TAU

/** An angle in (−π, π]. */
export const wrapSigned = (a: number) => {
  const w = wrapPositive(a)
  return w > Math.PI ? w - TAU : w
}

export const toDegrees = (a: number) => (a * 180) / Math.PI

/** "2.094 rad = 120°". */
export const formatAngle = (a: number) => `${formatNumber(a)} rad = ${formatNumber(toDegrees(a))}°`

/** Points on the circle of radius r about c, from angle a0 to a1 (either direction). */
export function arc(r: number, a0: number, a1: number, n = 64, c: Vec = [0, 0]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i <= n; i++) {
    const a = a0 + ((a1 - a0) * i) / n
    x.push(c[0] + r * Math.cos(a))
    y.push(c[1] + r * Math.sin(a))
  }
  return { x, y }
}

/** The full circle of radius r about c. */
export const circle = (r = 1, c: Vec = [0, 0], n = 200) => arc(r, 0, TAU, n, c)

/** A filled circular sector of radius r between angles a0 and a1, centred at c. */
export function sector(r: number, a0: number, a1: number, tone: number, opacity = 0.2, c: Vec = [0, 0]): FilledShape {
  const { x, y } = arc(r, a0, a1, 96, c)
  const ring: [number, number][] = [c, ...x.map((xi, i): [number, number] => [xi, y[i]])]
  return { contours: [ring], tone, opacity }
}

/** A filled polygon through the given points. */
export const polygon = (points: Vec[], tone: number, opacity = 0.2): FilledShape => ({
  contours: [points],
  tone,
  opacity,
})

/** Evenly spaced values from a to b inclusive. */
export function grid(a: number, b: number, n: number): number[] {
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
}
