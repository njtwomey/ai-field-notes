/** Small numeric helpers for in-browser widgets, backed by aifn. */
import { stream, uniform as drawUniform, normal as drawNormal } from 'aifn/foundation/random'
import { sigmoid as aifnSigmoid } from 'aifn/numerics/special'

/** Deterministic PRNG backed by aifn stream. Same seed, same sequence, on every device. */
export function rng(seed: number) {
  const s = stream(seed)
  return {
    uniform: () => drawUniform(s),
    normal: () => drawNormal(s),
  }
}

export function linspace(start: number, stop: number, n: number): number[] {
  if (n === 1) return [start]
  return Array.from({ length: n }, (_, i) => start + ((stop - start) * i) / (n - 1))
}

export const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0)
export const mean = (xs: number[]) => sum(xs) / xs.length
export const sigmoid = (z: number) => aifnSigmoid(z) as number

/** Least-squares slope and intercept for one feature. */
export function olsFit(x: number[], y: number[]): { slope: number; intercept: number } {
  const mx = mean(x)
  const my = mean(y)
  const sxy = sum(x.map((xi, i) => (xi - mx) * (y[i] - my)))
  const sxx = sum(x.map((xi) => (xi - mx) ** 2))
  const slope = sxx === 0 ? 0 : sxy / sxx
  return { slope, intercept: my - slope * mx }
}
