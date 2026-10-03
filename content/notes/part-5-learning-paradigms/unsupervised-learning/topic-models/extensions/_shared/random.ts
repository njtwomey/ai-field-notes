import type { rng } from '@/lib/math'

export type Rng = ReturnType<typeof rng>

/** A gamma(shape, 1) draw by Marsaglia–Tsang, boosted for shape < 1. */
export function gamma(r: Rng, shape: number): number {
  if (shape < 1) return gamma(r, shape + 1) * Math.pow(r.uniform(), 1 / shape)
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const x = r.normal()
    const v = (1 + c * x) ** 3
    if (v > 0 && Math.log(r.uniform()) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v
  }
}

/** A Dirichlet draw from normalised gamma draws. */
export function dirichlet(r: Rng, alpha: number[]): number[] {
  const g = alpha.map((a) => gamma(r, a))
  const s = g.reduce((x, y) => x + y, 0)
  return g.map((v) => v / s)
}

/** An index drawn with probability proportional to the (unnormalised) weights. */
export function categorical(r: Rng, weights: number[]): number {
  let u = r.uniform() * weights.reduce((a, b) => a + b, 0)
  for (let i = 0; i < weights.length; i++) if ((u -= weights[i]) <= 0) return i
  return weights.length - 1
}

export function softmax(x: number[]): number[] {
  const m = Math.max(...x)
  const e = x.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

export function correlation(a: number[], b: number[]): number {
  const n = a.length
  const ma = a.reduce((s, v) => s + v, 0) / n
  const mb = b.reduce((s, v) => s + v, 0) / n
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}
