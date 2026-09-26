/** Small numeric helpers for in-browser widgets. Heavy computation belongs in a Python `@figure` builder instead. */

/** Deterministic PRNG (mulberry32). Same seed, same sequence, on every device. */
export function rng(seed: number) {
  let a = seed >>> 0
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const normal = () => {
    const u = Math.max(uniform(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform())
  }
  return { uniform, normal }
}

export function linspace(start: number, stop: number, n: number): number[] {
  if (n === 1) return [start]
  return Array.from({ length: n }, (_, i) => start + ((stop - start) * i) / (n - 1))
}

export const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)
export const mean = (xs: number[]) => sum(xs) / xs.length
export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

/** Least-squares slope and intercept for one feature. */
export function olsFit(x: number[], y: number[]): { slope: number; intercept: number } {
  const mx = mean(x)
  const my = mean(y)
  const sxy = sum(x.map((xi, i) => (xi - mx) * (y[i] - my)))
  const sxx = sum(x.map((xi) => (xi - mx) ** 2))
  const slope = sxx === 0 ? 0 : sxy / sxx
  return { slope, intercept: my - slope * mx }
}
