import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * A one-dimensional Gaussian mixture pushed through the DDPM forward process. Every marginal is analytic:
 * q(x_t) = Σ_k π_k N(√ᾱ_t μ_k, ᾱ_t σ_k² + 1 − ᾱ_t), with the linear β schedule of Ho et al. (1e-4 to 0.02, T = 1000).
 */
const T = 1000
const PI = [0.3, 0.5, 0.2]
const MU = [-2, 0.5, 2.5]
const SD = [0.3, 0.5, 0.25]
const X = toFlat(linspace(-4, 4, 401))
const DX = X[1] - X[0]

const ALPHA_BAR = (() => {
  const out = [1]
  let prod = 1
  for (let t = 1; t <= T; t++) {
    prod *= 1 - (1e-4 + ((0.02 - 1e-4) * (t - 1)) / (T - 1))
    out.push(prod)
  }
  return out
})()

const normal = (x: number, m: number, v: number) => Math.exp(-((x - m) ** 2) / (2 * v)) / Math.sqrt(2 * Math.PI * v)

/** Density and score (d/dx log density) of the noised mixture at one point. */
function marginal(x: number, ab: number) {
  let p = 0
  let dp = 0
  for (let k = 0; k < PI.length; k++) {
    const m = Math.sqrt(ab) * MU[k]
    const v = ab * SD[k] ** 2 + 1 - ab
    const w = PI[k] * normal(x, m, v)
    p += w
    dp += (w * (m - x)) / v
  }
  return { p, score: dp / p }
}

type View = 'density' | 'score'

export function NoisedMixture({ initialView = 'density' }: { initialView?: View }) {
  const state = useFigureState({
    step: int(100, { min: 0, max: T, step: 10, label: 'step t' }),
    view: choice<View>(
      [
        { value: 'density', label: 'density' },
        { value: 'score', label: 'score' },
      ],
      initialView,
      { label: 'show' },
    ),
  })
  const ab = ALPHA_BAR[state.step]

  const data = useMemo(() => {
    const now = X.map((x) => marginal(x, ab))
    const start = X.map((x) => marginal(x, 1))
    const prior = X.map((x) => normal(x, 0, 1))
    const kl = now.reduce((s, m, i) => (m.p > 1e-300 ? s + m.p * Math.log(m.p / prior[i]) * DX : s), 0)
    return { now, start, prior, kl }
  }, [ab])

  const series = useMemo<SeriesSpec[]>(() => {
    if (state.view === 'density')
      return [
        { name: 'data q(x₀)', type: 'line', x: X, y: data.start.map((m) => m.p), muted: true },
        { name: 'N(0, 1)', type: 'line', x: X, y: data.prior, emphasis: true, dashed: true },
        { name: 'q(x_t)', type: 'line', x: X, y: data.now.map((m) => m.p), slot: 0, area: true },
      ]
    return [
      { name: 'score of N(0, 1): −x', type: 'line', x: X, y: X.map((x) => -x), emphasis: true, dashed: true },
      { name: '∇ log q(x_t)', type: 'line', x: X, y: data.now.map((m) => m.score), slot: 1 },
    ]
  }, [state.view, data])

  const xAxis = useAxis({ label: 'x', range: [-4, 4] })
  const yAxis = useAxis({
    label: state.view === 'density' ? 'density' : 'score',
    range: state.view === 'density' ? [0, 0.5] : [-12, 12],
  })
  return (
    <Figure
      title="The forward diffusion of a Gaussian mixture"
      state={state}
      caption="Data from a three-component mixture are noised by x_t = √ᾱ_t x₀ + √(1 − ᾱ_t) ε with the linear schedule of Ho et al. (T = 1000). The marginal q(x_t) stays a Gaussian mixture: the components shrink towards 0 and widen until they merge into N(0, 1). The score view shows ∇ log q(x_t), the vector field a reverse-time sampler follows; it points towards the modes and approaches −x as t grows."

      readouts={
        <>
          <Readout label="ᾱ_t" value={formatNumber(ab)} />
          <Readout label="signal scale √ᾱ_t" value={formatNumber(Math.sqrt(ab))} />
          <Readout label="noise scale √(1 − ᾱ_t)" value={formatNumber(Math.sqrt(1 - ab))} />
          <Readout label="KL(q(x_t) ‖ N(0, 1))" value={formatNumber(data.kl)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
