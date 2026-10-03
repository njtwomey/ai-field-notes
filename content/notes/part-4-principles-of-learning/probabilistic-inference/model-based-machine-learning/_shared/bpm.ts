/**
 * The Bayes point machine in two dimensions: a linear classifier through the origin with weights w ~ N(0, I) and
 * labels y = sign(wᵀx + ε), ε ~ N(0, β²). Expectation propagation gives a Gaussian posterior whose mean approximates
 * the Bayes point; a grid gives the exact posterior for comparison.
 */
import { Phi, vTrunc, wTrunc } from './gaussian.ts'

export type Vec2 = [number, number]
export type Mat2 = [[number, number], [number, number]]
export type Labelled = { x: Vec2; y: 1 | -1 }

const dot = (a: Vec2, b: Vec2) => a[0] * b[0] + a[1] * b[1]
const mul = (S: Mat2, v: Vec2): Vec2 => [S[0][0] * v[0] + S[0][1] * v[1], S[1][0] * v[0] + S[1][1] * v[1]]

/**
 * EP with one scalar site per point. Site i is a Gaussian in the projection f_i = wᵀx_i with precision τ_i and
 * precision-times-mean ν_i. Each update removes the site from the marginal of f_i (the cavity), matches the moments of
 * cavity × Φ(y f / β), and puts the new site back with a rank-one update of the covariance.
 */
export function epBpm(data: Labelled[], beta: number, sweeps = 20): { mean: Vec2; cov: Mat2 } {
  let mean: Vec2 = [0, 0]
  let cov: Mat2 = [
    [1, 0],
    [0, 1],
  ]
  const tau = data.map(() => 0)
  const nu = data.map(() => 0)
  for (let s = 0; s < sweeps; s++) {
    data.forEach(({ x, y }, i) => {
      const Sx = mul(cov, x)
      const vf = dot(x, Sx)
      const mf = dot(x, mean)
      // Cavity in f: divide out the site.
      const cavPrec = 1 / vf - tau[i]
      if (cavPrec <= 0) return
      const cavVar = 1 / cavPrec
      const cavMean = cavVar * (mf / vf - nu[i])
      // Tilted moments of N(f; cavMean, cavVar) Φ(y f / β).
      const sd = Math.sqrt(beta * beta + cavVar)
      const z = (y * cavMean) / sd
      const newMean = cavMean + (y * cavVar * vTrunc(z)) / sd
      const newVar = cavVar * (1 - (cavVar / (sd * sd)) * wTrunc(z))
      const newTau = Math.max(1 / newVar - cavPrec, 0)
      const newNu = newMean / newVar - cavMean / cavVar
      const dTau = newTau - tau[i]
      tau[i] = newTau
      nu[i] = newTau > 0 ? newNu : 0
      // Rank-one update of the posterior: Σ ← Σ − Σx xᵀΣ dτ / (1 + dτ xᵀΣx); precision-mean gains dν x.
      const k = dTau / (1 + dTau * vf)
      const newCov: Mat2 = [
        [cov[0][0] - k * Sx[0] * Sx[0], cov[0][1] - k * Sx[0] * Sx[1]],
        [cov[1][0] - k * Sx[1] * Sx[0], cov[1][1] - k * Sx[1] * Sx[1]],
      ]
      // Recompute the mean from the natural parameters: μ = Σ Σ_i ν_i x_i (the prior mean is zero).
      const h: Vec2 = [0, 0]
      data.forEach((d, j) => {
        h[0] += nu[j] * d.x[0]
        h[1] += nu[j] * d.x[1]
      })
      cov = newCov
      mean = mul(cov, h)
    })
  }
  return { mean, cov }
}

/** Unnormalised posterior density on a grid of weights: N(w; 0, I) Π Φ(y wᵀx / β). Row-major z[i][j] at (ws[j], ws[i]). */
export function posteriorGrid(data: Labelled[], beta: number, ws: number[]): { z: number[][]; mean: Vec2 } {
  let total = 0
  const m: Vec2 = [0, 0]
  const z = ws.map((w2) =>
    ws.map((w1) => {
      let p = Math.exp(-0.5 * (w1 * w1 + w2 * w2))
      for (const { x, y } of data) p *= Phi((y * (w1 * x[0] + w2 * x[1])) / beta)
      total += p
      m[0] += p * w1
      m[1] += p * w2
      return p
    }),
  )
  return { z, mean: [m[0] / total, m[1] / total] }
}

/** Hard-margin direction through the origin: the unit w maximising the smallest distance y_i wᵀx_i from a point to the
 * boundary, by a search over angles. */
export function maxMarginDirection(data: Labelled[], steps = 1440): { w: Vec2; margin: number } {
  let best: Vec2 = [1, 0]
  let bestMargin = -Infinity
  for (let k = 0; k < steps; k++) {
    const a = (2 * Math.PI * k) / steps
    const w: Vec2 = [Math.cos(a), Math.sin(a)]
    const margin = Math.min(...data.map(({ x, y }) => y * dot(w, x)))
    if (margin > bestMargin) {
      bestMargin = margin
      best = w
    }
  }
  return { w: best, margin: bestMargin }
}
