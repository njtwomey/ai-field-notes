import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

/** Forecast horizon H and look-back length L = 3H, in time steps. */
const H = 12
const L = 3 * H
const STEPS = Array.from({ length: L + H }, (_, i) => i - L)
const BACK = STEPS.filter((n) => n < 0)
const FORE = STEPS.filter((n) => n >= 0)
/** N-BEATS time grid: t = n / H, so the forecast window is [0, 1) and the look-back window is [-3, 0). */
const tOf = (n: number) => n / H

/** Noise-free signal: a gently curved trend plus a seasonal cycle of period H with three harmonics. */
const trueSignal = (n: number) => {
  const t = tOf(n)
  return (
    20 +
    3 * t +
    1.2 * t * t +
    4 * Math.sin(2 * Math.PI * t) +
    1.5 * Math.cos(4 * Math.PI * t) +
    0.8 * Math.sin(6 * Math.PI * t)
  )
}
const TRUTH = STEPS.map(trueSignal)

/** Polynomial trend basis [1, t, ..., t^p]. */
const trendBasis = (p: number) => (n: number) => Array.from({ length: p + 1 }, (_, j) => tOf(n) ** j)
/** Fourier seasonality basis [cos 2πit, sin 2πit], i = 1..K. The constant column is left to the trend stack. */
const seasonBasis = (K: number) => (n: number) =>
  Array.from({ length: K }, (_, j) => [
    Math.cos(2 * Math.PI * (j + 1) * tOf(n)),
    Math.sin(2 * Math.PI * (j + 1) * tOf(n)),
  ]).flat()

/** Least squares by the normal equations and Gaussian elimination with partial pivoting; bases here are small. */
function lstsq(rows: number[][], y: number[]): number[] {
  const m = rows[0]?.length ?? 0
  if (m === 0) return []
  const A = Array.from({ length: m }, (_, i) =>
    Array.from({ length: m + 1 }, (_, j) =>
      j < m ? rows.reduce((s, r) => s + r[i] * r[j], 0) : rows.reduce((s, r, k) => s + r[i] * y[k], 0),
    ),
  )
  for (let c = 0; c < m; c++) {
    let piv = c
    for (let r = c + 1; r < m; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r
    ;[A[c], A[piv]] = [A[piv], A[c]]
    for (let r = c + 1; r < m; r++) {
      const f = A[r][c] / A[c][c]
      for (let j = c; j <= m; j++) A[r][j] -= f * A[c][j]
    }
  }
  const x = new Array<number>(m).fill(0)
  for (let r = m - 1; r >= 0; r--) {
    let s = A[r][m]
    for (let j = r + 1; j < m; j++) s -= A[r][j] * x[j]
    x[r] = s / A[r][r]
  }
  return x
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0)
const rmse = (a: number[], b: number[]) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length)

/**
 * The interpretable N-BEATS decomposition with its networks replaced by least squares: a trend stack fits a polynomial
 * to the look-back window and passes on the residual, and a seasonality stack fits a Fourier series to that residual.
 * Both bases extend over the forecast window, which gives the two partial forecasts.
 */
export function BasisFit() {
  const state = useFigureState({
    degree: int(2, { min: 0, max: 4, step: 1, label: 'polynomial degree p', format: (v) => String(v) }),
    harmonics: int(3, { min: 0, max: 6, step: 1, label: 'harmonics K', format: (v) => String(v) }),
    noise: float(1, { min: 0, max: 4, step: 0.1, label: 'noise sd' }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const g = stream(state.seed)
    const y = TRUTH.map((v) => v + state.noise * normal(g))
    const yBack = y.slice(0, L)

    const T = trendBasis(state.degree)
    const thetaT = lstsq(BACK.map(T), yBack)
    const trend = STEPS.map((n) => dot(T(n), thetaT))

    // Doubly residual stacking: the seasonality stack sees the look-back window minus the trend stack's backcast.
    const residual = yBack.map((v, i) => v - trend[i])
    const S = seasonBasis(state.harmonics)
    const thetaS = state.harmonics > 0 ? lstsq(BACK.map(S), residual) : []
    const season = STEPS.map((n) => (state.harmonics > 0 ? dot(S(n), thetaS) : 0))

    const total = trend.map((v, i) => v + season[i])
    const data = [
      { name: 'observed look-back', x: BACK, y: yBack, slot: 0 },
      { name: 'future values', x: FORE, y: y.slice(L), slot: 0, dashed: true },
      { name: 'backcast + forecast', x: STEPS, y: total, emphasis: true },
    ] as const
    const parts = [
      { name: 'trend stack', x: STEPS, y: trend, slot: 1 },
      { name: 'seasonality stack', x: STEPS, y: season, slot: 2 },
    ] as const
    return {
      data,
      parts,
      backErr: rmse(total.slice(0, L), yBack),
      foreErr: rmse(total.slice(L), TRUTH.slice(L)),
      coefs: state.degree + 1 + 2 * state.harmonics,
    }
  }, [state.degree, state.harmonics, state.noise, state.seed])

  const xAxis = useAxis({ label: 'time step n', hold: 'union' })
  const yAxis = useAxis({ label: 'y', hold: 'union' })
  const xAxis2 = useAxis({ label: 'time step n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'component', hold: 'union' })
  return (
    <Figure
      title="Trend and seasonality bases"
      state={state}
      caption="A series of 48 steps: a look-back window of L = 36 steps (n < 0) and a forecast window of H = 12 (n ≥ 0), on the N-BEATS grid t = n / H. The trend stack fits a polynomial of degree p in t to the look-back window by least squares; the seasonality stack fits K Fourier harmonics of period H/i to what the trend leaves. Each basis is then evaluated over the forecast window, and the forecast is the sum of the two partial forecasts. In N-BEATS the coefficients come from fully connected networks; here they come from least squares, so the figure shows only what the bases can express. A high polynomial degree fits the look-back window more closely and extrapolates wildly. Too few harmonics leave seasonal shape in the residual."

      readouts={
        <>
          <Readout label="coefficients (p + 1) + 2K" value={String(r.coefs)} />
          <Readout label="look-back RMSE" value={formatNumber(r.backErr)} />
          <Readout label="forecast RMSE vs true signal" value={formatNumber(r.foreErr)} />
        </>
      }
    >
      <div className="space-y-3">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...r.data[0]} />
          <Curve {...r.data[1]} />
          <Curve {...r.data[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={180}>
          <Curve {...r.parts[0]} />
          <Curve {...r.parts[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
