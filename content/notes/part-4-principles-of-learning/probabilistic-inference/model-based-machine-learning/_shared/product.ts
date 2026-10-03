/**
 * The product factor z = s·t with a noisy observation of z, the building block of Matchbox's bilinear model. Priors
 * s ~ N(ms, vs), t ~ N(mt, vt), and z observed as zObs with noise variance β². Compares the exact marginal of s with its
 * moment-matched Gaussian (the projection EP would make) and the fully factorised variational (VMP) solution.
 */
import { gaussPdf } from './gaussian.ts'

export type ProductSetup = { ms: number; vs: number; mt: number; vt: number; zObs: number; noiseVar: number }

/** Exact posterior density of s on a grid: N(s; ms, vs) · N(zObs; s·mt, β² + s²·vt), normalised on the grid. */
export function exactS(p: ProductSetup, grid: number[]): { density: number[]; mean: number; variance: number } {
  const raw = grid.map((s) => gaussPdf(s, p.ms, p.vs) * gaussPdf(p.zObs, s * p.mt, p.noiseVar + s * s * p.vt))
  const h = grid[1] - grid[0]
  const z = raw.reduce((a, b) => a + b, 0) * h
  const density = raw.map((r) => r / z)
  const mean = density.reduce((a, d, i) => a + d * grid[i], 0) * h
  const variance = density.reduce((a, d, i) => a + d * (grid[i] - mean) ** 2, 0) * h
  return { density, mean, variance }
}

/**
 * Mean-field VMP: q(s) q(t), each Gaussian. The message from the product factor to s is N(s; zObs⟨t⟩/⟨t²⟩, β²/⟨t²⟩),
 * so its precision grows with ⟨t²⟩, the second moment of t, and uncertainty in t makes s look more certain.
 */
export function vmp(
  p: ProductSetup,
  iters = 100,
): { s: { mean: number; variance: number }; t: { mean: number; variance: number } } {
  let s = { mean: p.ms, variance: p.vs }
  let t = { mean: p.mt, variance: p.vt }
  for (let i = 0; i < iters; i++) {
    const t2 = t.variance + t.mean ** 2
    const precS = 1 / p.vs + t2 / p.noiseVar
    s = { variance: 1 / precS, mean: (p.ms / p.vs + (p.zObs * t.mean) / p.noiseVar) / precS }
    const s2 = s.variance + s.mean ** 2
    const precT = 1 / p.vt + s2 / p.noiseVar
    t = { variance: 1 / precT, mean: (p.mt / p.vt + (p.zObs * s.mean) / p.noiseVar) / precT }
  }
  return { s, t }
}
