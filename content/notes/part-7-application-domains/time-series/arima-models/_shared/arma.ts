import { normal, stream } from 'aifn/foundation/random'
/**
 * ARMA helpers for the figures in time-series/arima-models. Conventions follow the notes:
 * x_t = φ₁x_{t−1} + … + φ_p x_{t−p} + ε_t + θ₁ε_{t−1} + … + θ_q ε_{t−q}, with ε_t ~ N(0, 1).
 */

/** Simulate n values of an ARMA process, discarding a burn-in so the start-up transient is gone. */
export function simulateArma(phi: number[], theta: number[], n: number, seed: number, burn = 200): number[] {
  const g = stream(seed)
  const total = n + burn
  const e = Array.from({ length: total }, () => normal(g))
  const x = new Array<number>(total).fill(0)
  for (let t = 0; t < total; t++) {
    let v = e[t]
    phi.forEach((p, i) => {
      if (t - i - 1 >= 0) v += p * x[t - i - 1]
    })
    theta.forEach((th, j) => {
      if (t - j - 1 >= 0) v += th * e[t - j - 1]
    })
    // A non-stationary choice explodes; clip so the chart stays drawable and the readout says why.
    x[t] = Math.max(-1e6, Math.min(1e6, v))
  }
  return x.slice(burn)
}

/** Sample autocorrelation r_0 … r_H with the biased (divide-by-n) autocovariance. */
export function sampleAcf(x: number[], H: number): number[] {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const d = x.map((v) => v - m)
  const g = (h: number) => {
    let s = 0
    for (let t = 0; t + h < n; t++) s += d[t] * d[t + h]
    return s / n
  }
  const g0 = g(0)
  return Array.from({ length: H + 1 }, (_, h) => (g0 > 0 ? g(h) / g0 : 0))
}

/** Partial autocorrelations φ_11 … φ_HH from autocorrelations ρ_0 … ρ_H, by the Durbin–Levinson recursion. */
export function pacfFromAcf(rho: number[]): number[] {
  const H = rho.length - 1
  const out: number[] = []
  let prev: number[] = []
  for (let h = 1; h <= H; h++) {
    let num = rho[h]
    let den = 1
    for (let j = 1; j < h; j++) {
      num -= prev[j - 1] * rho[h - j]
      den -= prev[j - 1] * rho[j]
    }
    const k = Math.abs(den) < 1e-12 ? 0 : num / den
    const next = prev.map((p, j) => p - k * prev[h - 2 - j])
    next.push(k)
    prev = next
    out.push(k)
  }
  return out
}

/** ψ weights of the causal MA(∞) form, ψ_0 = 1, ψ_j = θ_j + Σ_i φ_i ψ_{j−i}. */
export function psiWeights(phi: number[], theta: number[], J: number): number[] {
  const psi = [1]
  for (let j = 1; j <= J; j++) {
    let v = theta[j - 1] ?? 0
    phi.forEach((p, i) => {
      if (j - i - 1 >= 0) v += p * psi[j - i - 1]
    })
    psi.push(v)
  }
  return psi
}

/** Theoretical autocorrelations ρ_0 … ρ_H of a stationary ARMA process, from a long truncation of its ψ weights. */
export function theoreticalAcf(phi: number[], theta: number[], H: number, J = 600): number[] {
  const psi = psiWeights(phi, theta, J + H)
  const g = (h: number) => {
    let s = 0
    for (let j = 0; j + h <= J + H; j++) s += psi[j] * psi[j + h]
    return s
  }
  const g0 = g(0)
  return Array.from({ length: H + 1 }, (_, h) => g(h) / g0)
}

/**
 * Roots of 1 + a₁z + a₂z² (a quadratic, or linear when a₂ = 0), as [re, im] pairs. The AR polynomial is
 * 1 − φ₁z − φ₂z², so pass (−φ₁, −φ₂); the MA polynomial is 1 + θ₁z + θ₂z², so pass (θ₁, θ₂).
 */
export function quadraticRoots(a1: number, a2: number): [number, number][] {
  if (Math.abs(a2) < 1e-12) return Math.abs(a1) < 1e-12 ? [] : [[-1 / a1, 0]]
  const disc = a1 * a1 - 4 * a2
  if (disc >= 0) {
    const s = Math.sqrt(disc)
    return [
      [(-a1 + s) / (2 * a2), 0],
      [(-a1 - s) / (2 * a2), 0],
    ]
  }
  const s = Math.sqrt(-disc)
  return [
    [-a1 / (2 * a2), s / (2 * a2)],
    [-a1 / (2 * a2), -s / (2 * a2)],
  ]
}

/** Smallest root modulus of 1 + a₁z + a₂z²; the process is stationary (or invertible) when it exceeds 1. */
export function minRootModulus(a1: number, a2: number): number {
  const roots = quadraticRoots(a1, a2)
  return roots.length ? Math.min(...roots.map(([re, im]) => Math.hypot(re, im))) : Infinity
}
