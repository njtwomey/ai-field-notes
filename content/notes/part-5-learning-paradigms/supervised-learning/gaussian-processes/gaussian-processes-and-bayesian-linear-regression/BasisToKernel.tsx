import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normal, stream } from 'aifn-compute/foundation/random'

const GRID = toFlat(linspace(-5, 5, 201))
const X_RANGE: [number, number] = [-5, 5]
const S_RANGE: [number | undefined, number | undefined] = [-4, 4]
const K_RANGE: [number | undefined, number | undefined] = [0, undefined]
/** Basis centres cover a wider interval than the plot, so the edges of the plot are not starved of bumps. */
const LO = -8
const HI = 8
const MAX_M = 80
const NORMALS = (() => {
  const g = stream(21)
  return Array.from({ length: 3 }, () => Array.from({ length: MAX_M }, () => normal(g)))
})()

/**
 * Bayesian linear regression on M Gaussian bumps of width λ, with weight variance proportional to the bump spacing Δ.
 * As M grows its covariance tends to the squared exponential kernel √π λ exp(−(x − x′)²/4λ²).
 */
export function BasisToKernel() {
  const state = useFigureState({
    m: int(8, { min: 2, max: MAX_M, step: 1, label: 'basis functions M', format: (v) => String(v) }),
    lambda: float(0.7, { min: 0.3, max: 1.5, step: 0.05, label: 'bump width λ' }),
  })

  const r = useMemo(() => {
    const centres = toFlat(linspace(LO, HI, state.m))
    const delta = (HI - LO) / (state.m - 1)
    const bump = (x: number, c: number) => Math.exp(-((x - c) ** 2) / (2 * state.lambda ** 2))
    const feats = GRID.map((x) => centres.map((c) => bump(x, c)))
    const draws = NORMALS.map((z) => feats.map((f) => f.reduce((s, v, i) => s + v * Math.sqrt(delta) * z[i], 0)))
    const kFinite = GRID.map((x) => centres.reduce((s, c) => s + delta * bump(x, c) * bump(0, c), 0))
    const varFinite = GRID.map((x) => centres.reduce((s, c) => s + delta * bump(x, c) ** 2, 0))
    const amp = Math.sqrt(Math.PI) * state.lambda
    const kLimit = GRID.map((x) => amp * Math.exp(-(x * x) / (4 * state.lambda ** 2)))
    const spread = Math.max(...varFinite) - Math.min(...varFinite)
    return { draws, kFinite, varFinite, kLimit, amp, spread, delta }
  }, [state.m, state.lambda])

  const samples: SeriesSpec[] = r.draws.map((d, i) => ({
    name: `sample ${i + 1}`,
    type: 'line',
    x: GRID,
    y: d,
    slot: i,
  }))
  const kernels = [
    { name: 'limit: SE kernel k(0, x)', x: GRID, y: r.kLimit, slot: 2, dashed: true },
    { name: 'M bumps: k(0, x)', x: GRID, y: r.kFinite, slot: 0 },
    { name: 'M bumps: variance k(x, x)', x: GRID, y: r.varFinite, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'f(x)', range: S_RANGE })
  const xAxis2 = useAxis({ label: 'x', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'covariance', range: K_RANGE })
  return (
    <Figure
      title="From basis functions to a kernel"
      state={state}
      caption="Left: three prior draws of a linear model f(x) = Σ wₘ φₘ(x) on M Gaussian bumps of width λ, centred evenly on [−8, 8], with weights wₘ ~ N(0, Δ) for spacing Δ. Right: the model's covariance k(0, x) = Σ Δ φₘ(0) φₘ(x) and its variance k(x, x), against the squared exponential limit. With few, widely spaced bumps the prior is lumpy: the variance rises and falls between centres, so the model is not stationary. When the spacing falls below about λ, the covariance matches the squared exponential kernel with length-scale √2 λ, and the linear model is a Gaussian process with that kernel."

      readouts={
        <>
          <Readout label="spacing Δ / width λ" value={formatNumber(r.delta / state.lambda)} />
          <Readout label="variance ripple (max − min)" value={formatNumber(r.spread)} />
          <Readout label="limit variance √π λ" value={formatNumber(r.amp)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          {seriesLayers(samples)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...kernels[0]} />
          <Curve {...kernels[1]} />
          <Curve {...kernels[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
