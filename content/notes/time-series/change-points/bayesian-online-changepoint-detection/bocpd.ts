/**
 * Bayesian online changepoint detection (Adams & MacKay 2007) for a Gaussian with unknown mean and known variance.
 * The underlying predictive model for a run of length r is Gaussian: a Normal prior N(μ₀, σ₀²) on the segment mean,
 * updated with the run's count and sum, gives the predictive N(μ_r, 1/τ_r + σ²).
 */

export type BocpdOptions = {
  /** Constant hazard H = 1/λ: the prior probability that a changepoint occurs at any step. */
  hazard: number
  /** Prior mean and standard deviation of each segment's mean. */
  mu0: number
  sigma0: number
  /** Known observation noise standard deviation. */
  sigma: number
}

export type BocpdResult = {
  /** posterior[t][r] = P(r_t = r | x_{1:t}) for t = 1..T (0-based array index t − 1), r = 0..t. */
  posterior: Float64Array[]
  /** Mean and standard deviation of the one-step predictive P(x_t | x_{1:t−1}), a mixture over run lengths. */
  predMean: number[]
  predSd: number[]
  /** Most probable run length at each step. */
  map: number[]
}

const normalPdf = (x: number, mean: number, variance: number) =>
  Math.exp(-((x - mean) ** 2) / (2 * variance)) / Math.sqrt(2 * Math.PI * variance)

export function bocpd(x: number[], { hazard, mu0, sigma0, sigma }: BocpdOptions): BocpdResult {
  const s2 = sigma * sigma
  const prior = 1 / (sigma0 * sigma0)
  // Run-length distribution P(r_{t−1} | x_{1:t−1}) and each run's sufficient statistics (count, sum).
  let probs = new Float64Array([1])
  let counts = new Float64Array([0])
  let sums = new Float64Array([0])
  const posterior: Float64Array[] = []
  const predMean: number[] = []
  const predSd: number[] = []
  const map: number[] = []

  for (const xt of x) {
    const m = probs.length
    // Step 3: the underlying predictive model π_t^(r) = P(x_t | r_{t−1}, x^(r)) for every current run length.
    const pi = new Float64Array(m)
    let mixMean = 0
    let mixSecond = 0
    for (let r = 0; r < m; r++) {
      const precision = prior + counts[r] / s2
      const mean = (mu0 * prior + sums[r] / s2) / precision
      const variance = 1 / precision + s2
      pi[r] = normalPdf(xt, mean, variance)
      mixMean += probs[r] * mean
      mixSecond += probs[r] * (variance + mean * mean)
    }
    predMean.push(mixMean)
    predSd.push(Math.sqrt(Math.max(mixSecond - mixMean * mixMean, 0)))

    // Steps 4–7: growth, changepoint, evidence and the normalised run-length posterior.
    const next = new Float64Array(m + 1)
    let changepoint = 0
    for (let r = 0; r < m; r++) {
      next[r + 1] = probs[r] * pi[r] * (1 - hazard)
      changepoint += probs[r] * pi[r] * hazard
    }
    next[0] = changepoint
    let evidence = 0
    for (let r = 0; r <= m; r++) evidence += next[r]
    let best = 0
    for (let r = 0; r <= m; r++) {
      next[r] /= evidence
      if (next[r] > next[best]) best = r
    }
    posterior.push(next)
    map.push(best)

    // Step 8: sufficient statistics — a fresh run at r = 0, every other run extended by x_t.
    const nextCounts = new Float64Array(m + 1)
    const nextSums = new Float64Array(m + 1)
    for (let r = 0; r < m; r++) {
      nextCounts[r + 1] = counts[r] + 1
      nextSums[r + 1] = sums[r] + xt
    }
    probs = next
    counts = nextCounts
    sums = nextSums
  }
  return { posterior, predMean, predSd, map }
}

/**
 * Changepoints read from the MAP run length: when it drops, the current run started at t − r + 1. Nearby starts are
 * merged. Returns the detection time and the located start (both 1-based).
 */
export function detections(map: number[], merge = 4): { at: number; start: number }[] {
  const out: { at: number; start: number }[] = []
  for (let i = 1; i < map.length; i++) {
    if (map[i] >= map[i - 1]) continue
    const t = i + 1
    const start = t - map[i] + 1
    const last = out.at(-1)
    if (last && Math.abs(last.start - start) <= merge) continue
    out.push({ at: t, start })
  }
  return out
}
