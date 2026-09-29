import { sigmoid } from '@/lib/math'
import { incompleteGamma } from '@/lib/math/special'
import { logistic1d } from '../_shared/ordinal'

const K = 4

type Mat2 = number[][]
const mul = (a: Mat2, b: Mat2): Mat2 => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))

/**
 * Brant's test of proportional odds. Fit a separate binary logit to each cumulative split y > j, then ask whether the
 * K − 1 slopes are equal. The slopes' joint covariance comes from Brant (1990): Cov(β̂_j, β̂_l) = V_j X'W_jl X V_l, with
 * V_j the inverse information of fit j and W_jl = diag(π_l(1 − π_j)) for j < l.
 */
export function brant(x: number[], y: number[]) {
  const fits = [0, 1, 2].map((j) =>
    logistic1d(
      x,
      y.map((v) => (v > j ? 1 : 0)),
    ),
  )
  const pi = fits.map((f) => x.map((xi) => sigmoid(f.a + f.b * xi)))
  const slopeCov = fits.map((fj, j) =>
    fits.map((fl, l) => {
      if (j === l) return fj.cov[1][1]
      const [lo, hi] = j < l ? [j, l] : [l, j]
      const m: Mat2 = [
        [0, 0],
        [0, 0],
      ]
      x.forEach((xi, i) => {
        const w = pi[hi][i] * (1 - pi[lo][i])
        m[0][0] += w
        m[0][1] += w * xi
        m[1][0] += w * xi
        m[1][1] += w * xi * xi
      })
      return mul(mul(fj.cov, m), fl.cov)[1][1]
    }),
  )
  const b = fits.map((f) => f.b)
  // Contrasts b_1 − b_2 and b_2 − b_3, their covariance D V Dᵀ, and the Wald statistic.
  const d = [b[0] - b[1], b[1] - b[2]]
  const D = [
    [1, -1, 0],
    [0, 1, -1],
  ]
  const S = D.map((r) =>
    D.map((c) => r.reduce((s, ri, j) => s + ri * c.reduce((t, cl, l) => t + slopeCov[j][l] * cl, 0), 0)),
  )
  const det = S[0][0] * S[1][1] - S[0][1] * S[1][0]
  const chi2 = (S[1][1] * d[0] * d[0] - 2 * S[0][1] * d[0] * d[1] + S[0][0] * d[1] * d[1]) / det
  const p = 1 - incompleteGamma((K - 2) / 2, chi2 / 2)
  return { fits, slopes: b, se: b.map((_, j) => Math.sqrt(slopeCov[j][j])), chi2, p }
}
