/**
 * Small numerical helpers shared by the stochastic-calculus widgets: Gaussian densities, histograms, Euler–Maruyama
 * paths, and the variance-preserving (Ornstein–Uhlenbeck) noising of a two-mode Gaussian mixture, whose density and
 * score are analytic at every time.
 */
import { rng } from '@/lib/math'

export const gaussPdf = (x: number, m: number, v: number) =>
  Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)

/** Histogram of `values` on [lo, hi] with `bins` equal bins, scaled to a density; values outside are dropped. */
export function histogram(values: ArrayLike<number>, lo: number, hi: number, bins: number) {
  const counts = new Array<number>(bins).fill(0)
  const width = (hi - lo) / bins
  for (let i = 0; i < values.length; i++) {
    const b = Math.floor((values[i] - lo) / width)
    if (b >= 0 && b < bins) counts[b]++
  }
  const x = counts.map((_, b) => lo + (b + 0.5) * width)
  const y = counts.map((c) => c / (values.length * width))
  return { x, y }
}

export type Drift = (x: number, t: number) => number

/**
 * Euler–Maruyama paths of dX = f(X, t) dt + g(X, t) dW on [t0, t1] with `steps` equal steps. With t1 < t0 the scheme
 * runs backwards in time: each step moves by f · (t1 − t0)/steps and adds noise of variance g² |Δt|.
 * Returns the state of every path at every step: out[k][i] is path i after k steps.
 */
export function eulerMaruyama(
  f: Drift,
  g: Drift,
  start: number[],
  t0: number,
  t1: number,
  steps: number,
  seed: number,
): Float64Array[] {
  const { normal } = rng(seed)
  const h = (t1 - t0) / steps
  const sq = Math.sqrt(Math.abs(h))
  let x = Float64Array.from(start)
  const out = [x]
  for (let k = 0; k < steps; k++) {
    const t = t0 + k * h
    const next = new Float64Array(x.length)
    for (let i = 0; i < x.length; i++) next[i] = x[i] + f(x[i], t) * h + g(x[i], t) * sq * normal()
    out.push(next)
    x = next
  }
  return out
}

/** Standard normal draws, seeded. */
export function normals(n: number, seed: number, mean = 0, sd = 1): number[] {
  const { normal } = rng(seed)
  return Array.from({ length: n }, () => mean + sd * normal())
}

/**
 * Two-mode data distribution noised by the variance-preserving SDE dX = −½βX dt + √β dW with constant β. Given x₀,
 * X_t ~ N(a x₀, 1 − a²) with a = exp(−βt/2), so a mixture of Gaussians stays a mixture of Gaussians.
 */
export const MIX = { pi: [0.35, 0.65], mu: [-2, 1.5], sd: [0.35, 0.5] }
export const BETA = 1
export const T_END = 5

export function vpMixture(x: number, t: number, beta = BETA) {
  const a = Math.exp((-beta * t) / 2)
  let p = 0
  let dp = 0
  for (let k = 0; k < MIX.pi.length; k++) {
    const m = a * MIX.mu[k]
    const v = a * a * MIX.sd[k] ** 2 + 1 - a * a
    const w = MIX.pi[k] * gaussPdf(x, m, v)
    p += w
    dp += (w * (m - x)) / v
  }
  return { p, score: p > 1e-300 ? dp / p : -x }
}

/** Draws from the data mixture. */
export function sampleMixture(n: number, seed: number): number[] {
  const { uniform, normal } = rng(seed)
  return Array.from({ length: n }, () => {
    const k = uniform() < MIX.pi[0] ? 0 : 1
    return MIX.mu[k] + MIX.sd[k] * normal()
  })
}

/** Thin one path out of a Float64Array-per-step array into chart coordinates. */
export function pathXY(states: Float64Array[], i: number, times: number[], every = 1) {
  const x: number[] = []
  const y: number[] = []
  for (let k = 0; k < states.length; k += every) {
    x.push(times[k])
    y.push(states[k][i])
  }
  return { x, y }
}

/** Join several paths into one line series; NaN breaks the line between paths, so one legend entry covers them all. */
export function joinPaths(paths: { x: number[]; y: number[] }[]) {
  const x: number[] = []
  const y: number[] = []
  for (const p of paths) {
    x.push(...p.x, NaN)
    y.push(...p.y, NaN)
  }
  return { x, y }
}

export type ReverseKind = 'sde' | 'ode' | 'noscore'

/**
 * Integrate the variance-preserving diffusion of MIX backwards from t = T_END to 0 with `steps` Euler(–Maruyama)
 * steps, using the exact score. out[k] holds the particles at time T_END − k·h.
 * - 'sde': reverse-time SDE, drift f − g²∇log p_t, noise g.
 * - 'ode': probability-flow ODE, drift f − ½g²∇log p_t, no noise.
 * - 'noscore': the forward drift run backwards with noise, i.e. the reverse SDE without its score term.
 */
export function reverseParticles(kind: ReverseKind, start: number[], steps: number, seed: number): Float64Array[] {
  const { normal } = rng(seed)
  const h = T_END / steps
  const sq = Math.sqrt(BETA * h)
  let x = Float64Array.from(start)
  const out = [x]
  for (let k = 0; k < steps; k++) {
    const t = T_END - k * h
    const next = new Float64Array(x.length)
    for (let i = 0; i < x.length; i++) {
      const xi = x[i]
      // Moving from t to t − h: x ← x − h·drift(x, t) (+ noise), with f = −½βx.
      const f = -0.5 * BETA * xi
      if (kind === 'noscore') next[i] = xi - h * f + sq * normal()
      else {
        const s = vpMixture(xi, t).score
        next[i] = kind === 'sde' ? xi - h * (f - BETA * s) + sq * normal() : xi - h * (f - 0.5 * BETA * s)
      }
    }
    out.push(next)
    x = next
  }
  return out
}

/** Forward VP noising of data samples on the same time grid, with the exact Gaussian transition. out[k] is at k·h. */
export function forwardParticles(data: number[], steps: number, seed: number): Float64Array[] {
  const { normal } = rng(seed)
  const h = T_END / steps
  const decay = Math.exp((-BETA * h) / 2)
  const sd = Math.sqrt(1 - decay * decay)
  let x = Float64Array.from(data)
  const out = [x]
  for (let k = 0; k < steps; k++) {
    const next = x.map((xi) => decay * xi + sd * normal())
    out.push(next)
    x = next
  }
  return out
}
