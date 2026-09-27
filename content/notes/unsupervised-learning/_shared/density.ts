/** A bimodal test density for the density-estimation widgets, with samplers and summary statistics. */
import { rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'

/** 0.7 N(0, 1) + 0.3 N(3, 0.5²): a broad mode and a narrow one, so no single bandwidth suits both. */
const COMPONENTS = [
  { weight: 0.7, mean: 0, sd: 1 },
  { weight: 0.3, mean: 3, sd: 0.5 },
]

export const DOMAIN: [number, number] = [-4, 5.5]

export function trueDensity(x: number): number {
  return COMPONENTS.reduce((s, c) => s + (c.weight * normalPdf((x - c.mean) / c.sd)) / c.sd, 0)
}

export function sample(n: number, seed: number): number[] {
  const r = rng(seed)
  return Array.from({ length: n }, () => {
    const c = r.uniform() < COMPONENTS[0].weight ? COMPONENTS[0] : COMPONENTS[1]
    return c.mean + c.sd * r.normal()
  })
}

/** Linear-interpolation quantile of sorted data, as numpy's default. */
export function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, sorted.length - 1)
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo])
}

export function summary(xs: number[]) {
  const n = xs.length
  const mean = xs.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / (n - 1))
  const sorted = [...xs].sort((a, b) => a - b)
  const iqr = quantile(sorted, 0.75) - quantile(sorted, 0.25)
  return { n, sd, iqr, min: sorted[0], max: sorted[n - 1] }
}

/** Integrated squared error between an estimate on an evenly spaced grid and the true density. */
export function ise(grid: number[], estimate: number[]): number {
  const dx = grid[1] - grid[0]
  return estimate.reduce((s, f, i) => s + (f - trueDensity(grid[i])) ** 2, 0) * dx
}
