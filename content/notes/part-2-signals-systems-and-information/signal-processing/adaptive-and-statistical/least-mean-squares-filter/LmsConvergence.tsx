import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toeplitz } from '../_shared/stat'
import { normal, stream } from 'aifn/foundation/random'

const M = 8
const STEPS = 1500
const RUNS = 20
const NOISE = 0.01 // measurement-noise variance σ_v²
const UNKNOWN = [0.8, -0.5, 0.35, 0.2, -0.15, 0.1, 0.05, -0.02]

/** Largest and smallest eigenvalues of a small symmetric matrix by power iteration (and on the shifted matrix). */
function extremeEigenvalues(R: number[][]) {
  const power = (A: number[][]) => {
    let v = A.map(() => 1)
    let lambda = 0
    for (let it = 0; it < 500; it++) {
      const w = A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0))
      lambda = Math.hypot(...w)
      v = w.map((x) => x / lambda)
    }
    // Rayleigh quotient gives the signed eigenvalue.
    const Av = A.map((row) => row.reduce((s, a, j) => s + a * v[j], 0))
    return Av.reduce((s, x, i) => s + x * v[i], 0)
  }
  const max = power(R)
  const shifted = R.map((row, i) => row.map((a, j) => (i === j ? max - a : -a)))
  return { max, min: max - power(shifted) }
}

/**
 * LMS identifying an unknown 8-tap FIR system from its input and noisy output. The input is AR(1) with correlation a,
 * scaled to unit variance, so a controls the eigenvalue spread of R and the speed of the slowest mode.
 */
export function LmsConvergence() {
  const state = useFigureState({
    logMu: float(-1.6, { min: -3.5, max: -0.3, step: 0.05, label: 'log₁₀ step size μ' }),
    corr: slider(0, 0.95, 0, { step: 0.05, label: 'input correlation a' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const mu = 10 ** state.logMu

  const r = useMemo(() => {
    const a = state.corr
    const R = toeplitz(Array.from({ length: M }, (_, k) => a ** k)) // unit-variance AR(1) autocorrelation
    const { max, min } = extremeEigenvalues(R)
    const mse = new Array(STEPS).fill(0)
    const devSq = new Array(STEPS).fill(0)
    let diverged = false
    for (let run = 0; run < RUNS; run++) {
      const g = stream(1000 * state.seed + run)
      const w = new Array(M).fill(0)
      const buf = new Array(M).fill(0)
      let u = normal(g)
      for (let n = 0; n < STEPS; n++) {
        u = a * u + Math.sqrt(1 - a * a) * normal(g)
        buf.unshift(u)
        buf.pop()
        const d = UNKNOWN.reduce((s, h, k) => s + h * buf[k], 0) + Math.sqrt(NOISE) * normal(g)
        const y = w.reduce((s, wk, k) => s + wk * buf[k], 0)
        const e = d - y
        for (let k = 0; k < M; k++) w[k] += mu * e * buf[k]
        if (!Number.isFinite(e) || Math.abs(e) > 1e6) {
          diverged = true
          break
        }
        mse[n] += (e * e) / RUNS
        devSq[n] += w.reduce((s, wk, k) => s + (wk - UNKNOWN[k]) ** 2, 0) / RUNS
      }
      if (diverged) break
    }
    const trR = M // unit-variance input: tr R = M σ_u²
    return { mse, devSq, max, min, diverged, misadjustment: (mu * trR) / 2, bound: 2 / max, trBound: 2 / trR }
  }, [mu, state.corr, state.seed])

  const t = Array.from({ length: STEPS }, (_, n) => n)
  const clampDb = (v: number) => Math.max(-60, 10 * Math.log10(Math.max(v, 1e-12)))
  const series: SeriesSpec[] = r.diverged
    ? []
    : [
        { name: 'mean-squared error (dB)', type: 'line', x: t, y: r.mse.map(clampDb), slot: 0 },
        { name: 'weight error ‖w − w°‖² (dB)', type: 'line', x: t, y: r.devSq.map(clampDb), slot: 1 },
        {
          name: 'noise floor σ_v²',
          type: 'line',
          x: [0, STEPS],
          y: [clampDb(NOISE), clampDb(NOISE)],
          dashed: true,
          slot: 2,
        },
      ]

  const xAxis = useAxis({ label: 'iteration n', hold: 'union' })
  const yAxis = useAxis({ label: 'dB', range: [-60, 10] })
  return (
    <Figure
      title="LMS system identification"
      state={state}
      caption="An 8-tap LMS filter learns an unknown 8-tap system from white or correlated input, averaged over 20 runs. The error falls towards the noise floor σ_v² = −20 dB; the gap that remains is the excess error from gradient noise, about μ tr R / 2 of the floor. Raise μ and convergence speeds up while the floor rises; past about 2/tr R the filter diverges. Correlated input spreads the eigenvalues of R, and the slowest mode, set by λ_min, dominates the tail."

      readouts={
        <>
          <Readout label="μ" value={formatNumber(mu)} />
          <Readout label="λ_max, λ_min" value={`${formatNumber(r.max)}, ${formatNumber(r.min)}`} />
          <Readout label="mean bound 2/λ_max" value={formatNumber(r.bound)} />
          <Readout label="practical bound 2/tr R" value={formatNumber(r.trBound)} />
          <Readout label="misadjustment μ tr R / 2" value={r.diverged ? 'diverged' : formatNumber(r.misadjustment)} />
        </>
      }
    >
      {r.diverged ? (
        <div className="flex h-[300px] items-center justify-center rounded-md border border-dashed text-sm text-muted-foreground">
          The filter diverged: μ is too large for this input.
        </div>
      ) : (
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(series)}
        </Plot>
      )}
    </Figure>
  )
}
