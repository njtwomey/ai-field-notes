/**
 * Exponential smoothing recursions for the figures in time-series/exponential-smoothing, in the component form of
 * Hyndman and Athanasopoulos: level ℓ, damped trend b and seasonal s with period m.
 */

export type Seasonality = 'none' | 'additive' | 'multiplicative'

export type SmoothingParams = {
  alpha: number
  /** Trend smoothing β*. Ignored when `trend` is false. */
  beta: number
  gamma: number
  /** Damping φ; 1 gives Holt's undamped linear trend. */
  phi: number
  trend: boolean
  seasonality: Seasonality
  m: number
}

export type SmoothingFit = {
  /** One-step-ahead forecasts ŷ_{t|t−1}, aligned with y. */
  fitted: number[]
  /** Forecasts ŷ_{T+h|T} for h = 1 … H. */
  forecast: number[]
}

/** Simple exponential smoothing from level ℓ_0; returns the one-step forecasts, which equal the previous level. */
export function ses(y: number[], alpha: number, l0 = y[0]): { fitted: number[]; level: number } {
  let l = l0
  const fitted = y.map((v) => {
    const f = l
    l = alpha * v + (1 - alpha) * l
    return f
  })
  return { fitted, level: l }
}

/**
 * Holt–Winters with optional damped trend and additive or multiplicative seasonality. Initial states come from the
 * first two seasons (or the first two values without seasonality), a common heuristic.
 */
export function smooth(y: number[], p: SmoothingParams, H: number): SmoothingFit {
  const { alpha, beta, gamma, phi, trend, seasonality, m } = p
  const seasonal = seasonality !== 'none'
  const period = seasonal ? m : 1
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const first = mean(y.slice(0, period))
  const second = mean(y.slice(period, 2 * period))
  let l = first
  let b = trend ? (second - first) / period : 0
  // s holds the last m seasonal states; s[0] is the one for the next time step.
  // Initial seasonal states: the first season's values relative to the straight-line trend through its centre.
  let s = seasonal
    ? y.slice(0, m).map((v, j) => {
        const line = first + b * (j - (m - 1) / 2)
        return seasonality === 'additive' ? v - line : v / line
      })
    : []
  // The initial level sits at the centre of the first season; move it to just before t = 1.
  l -= b * ((period - 1) / 2 + 1)
  const add = seasonality !== 'multiplicative'
  const fitted = y.map((v) => {
    const base = l + phi * b
    const sOld = seasonal ? s[0] : add ? 0 : 1
    const f = add ? base + sOld : base * sOld
    const lNew = add ? alpha * (v - sOld) + (1 - alpha) * base : alpha * (v / sOld) + (1 - alpha) * base
    const bNew = trend ? beta * (lNew - l) + (1 - beta) * phi * b : 0
    if (seasonal) {
      const sNew = add ? gamma * (v - base) + (1 - gamma) * sOld : gamma * (v / base) + (1 - gamma) * sOld
      s = [...s.slice(1), sNew]
    }
    l = lNew
    b = bNew
    return f
  })
  const forecast: number[] = []
  let damp = 0
  for (let h = 1; h <= H; h++) {
    damp += phi ** h
    const base = l + (trend ? damp * b : 0)
    const sh = seasonal ? s[(h - 1) % m] : add ? 0 : 1
    forecast.push(add ? base + sh : base * sh)
  }
  return { fitted, forecast }
}

export const rmse = (a: number[], b: number[]) =>
  Math.sqrt(a.reduce((acc, v, i) => acc + (v - b[i]) ** 2, 0) / Math.max(1, a.length))
