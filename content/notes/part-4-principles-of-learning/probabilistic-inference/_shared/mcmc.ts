/** Small helpers shared by the MCMC figures: autocorrelation, effective sample size and split R-hat. */

/** Sample autocorrelations ρ_0 … ρ_maxLag of a chain (ρ_0 = 1), with the biased (1/n) autocovariance. */
export function autocorrelation(x: number[], maxLag: number): number[] {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const d = x.map((v) => v - m)
  const c0 = d.reduce((a, v) => a + v * v, 0) / n
  if (c0 === 0) return [1]
  const out: number[] = []
  for (let k = 0; k <= Math.min(maxLag, n - 1); k++) {
    let s = 0
    for (let t = 0; t + k < n; t++) s += d[t] * d[t + k]
    out.push(s / n / c0)
  }
  return out
}

/**
 * Integrated autocorrelation time τ = 1 + 2 Σ ρ_k, truncated by Geyer's initial positive sequence: sum consecutive
 * pairs ρ_2m + ρ_2m+1 while they stay positive. Returns τ ≥ 1/n, so that ESS = n/τ never exceeds n by much.
 */
export function autocorrelationTime(x: number[]): number {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const d = x.map((v) => v - m)
  const c0 = d.reduce((a, v) => a + v * v, 0) / n
  // A chain that never moved carries one sample's worth of information.
  if (c0 === 0) return n
  const rho = (k: number) => {
    let s = 0
    for (let t = 0; t + k < n; t++) s += d[t] * d[t + k]
    return s / n / c0
  }
  let tau = -1
  for (let m = 0; 2 * m + 1 < n; m++) {
    const pair = rho(2 * m) + rho(2 * m + 1)
    if (pair <= 0) break
    tau += 2 * pair
  }
  return Math.max(tau, 1 / n)
}

export const effectiveSampleSize = (x: number[]) => x.length / autocorrelationTime(x)

/** Split R-hat: each chain is cut in half, and the between- and within-half variances are compared. */
export function splitRhat(chains: number[][]): number {
  const halves = chains.flatMap((c) => {
    const h = Math.floor(c.length / 2)
    return [c.slice(0, h), c.slice(c.length - h)]
  })
  const n = halves[0].length
  const means = halves.map((c) => c.reduce((a, b) => a + b, 0) / n)
  const grand = means.reduce((a, b) => a + b, 0) / means.length
  const B = (n / (means.length - 1)) * means.reduce((a, m) => a + (m - grand) ** 2, 0)
  const W = halves.reduce((a, c, j) => a + c.reduce((s, v) => s + (v - means[j]) ** 2, 0) / (n - 1), 0) / halves.length
  const V = ((n - 1) / n) * W + B / n
  return Math.sqrt(V / W)
}
