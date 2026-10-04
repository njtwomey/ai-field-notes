import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { joinPaths } from '../_shared/sde'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'

const STEPS = 200
const T = 4
const TIMES = toFlat(linspace(0, T, STEPS + 1))

/**
 * Ornstein–Uhlenbeck paths dX = −θX dt + σ dW, simulated with the exact Gaussian transition, against the exact mean
 * x₀e^{−θt} and the ±2 sd band, which widens to the stationary ±2σ/√(2θ).
 */
export function OrnsteinUhlenbeckPaths() {
  const state = useFigureState({
    theta: float(1, { min: 0.1, max: 4, step: 0.05, label: 'pull θ' }),
    sigma: float(1, { min: 0, max: 2, step: 0.05, label: 'noise σ' }),
    x0: float(3, { min: -4, max: 4, step: 0.1, label: 'start x₀' }),
    count: int(20, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
  })

  // Fixed standard normals, one stream per path: changing a parameter moves every path smoothly instead of redrawing
  // the noise, and raising the count adds paths without changing the existing ones.
  const z = useMemo(
    () =>
      Array.from({ length: state.count }, (_, k) => {
        const rs = stream(9 * 1000 + k)
        return Float64Array.from({ length: STEPS }, () => normal(rs))
      }),
    [state.count],
  )

  const series = useMemo<SeriesSpec[]>(() => {
    const th = state.theta
    const sg = state.sigma
    const h = T / STEPS
    const decay = Math.exp(-th * h)
    const stepSd = Math.sqrt(((sg * sg) / (2 * th)) * (1 - decay * decay))
    const paths = joinPaths(
      z.map((zi) => {
        const y = [state.x0]
        for (let k = 0; k < STEPS; k++) y.push(y[k] * decay + stepSd * zi[k])
        return { x: TIMES, y }
      }),
    )
    const mean = TIMES.map((t) => state.x0 * Math.exp(-th * t))
    const sd = TIMES.map((t) => Math.sqrt(((sg * sg) / (2 * th)) * (1 - Math.exp(-2 * th * t))))
    const stat = sg / Math.sqrt(2 * th)
    return [
      { name: 'sample paths', type: 'line', ...paths, slot: 2, thin: z.length > 1 },
      { name: 'mean x₀e^{−θt}', type: 'line', x: TIMES, y: mean, slot: 0 },
      {
        name: 'mean ± 2 sd',
        type: 'line',
        ...joinPaths([1, -1].map((s) => ({ x: TIMES, y: mean.map((m, k) => m + 2 * s * sd[k]) }))),
        slot: 1,
      },
      {
        name: 'stationary ± 2σ/√(2θ)',
        type: 'line',
        ...joinPaths([1, -1].map((s) => ({ x: [0, T], y: [2 * s * stat, 2 * s * stat] }))),
        dashed: true,
        emphasis: true,
      },
    ]
  }, [z, state.x0, state.theta, state.sigma])

  const th = state.theta
  const sg = state.sigma
  const xAxis = useAxis({ label: 't', range: [0, T] })
  const yAxis = useAxis({ label: 'X_t', range: [-5, 5] })
  return (
    <Figure
      title="Ornstein–Uhlenbeck paths and their Gaussian band"
      state={state}
      caption="Sample paths of dX = −θX dt + σ dW started at x₀, drawn as light lines; the paths slider sets how many. The mean decays to 0 at rate θ and the band of ±2 standard deviations widens from zero to the stationary ±2σ/√(2θ) (dashed). Large θ forgets the start quickly and holds the paths in a narrow band; large σ widens the band. With θ = ½ and σ = 1 the stationary law is N(0, 1): this is the forward process of a variance-preserving diffusion model. Drag the start point at t = 0."

      readouts={
        <>
          <Readout label="half-life ln 2 / θ" value={formatNumber(Math.log(2) / th)} />
          <Readout label="stationary sd σ/√(2θ)" value={formatNumber(sg / Math.sqrt(2 * th))} />
          <Readout label="mean at t = 1" value={formatNumber(state.x0 * Math.exp(-th))} />
          <Readout
            label="sd at t = 1"
            value={formatNumber(Math.sqrt(((sg * sg) / (2 * th)) * (1 - Math.exp(-2 * th))))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
        <Handle kind="point" at={[0, state.x0]} label="x₀" onDrag={([, y]) => state.set('x0', y)} />
      </Plot>
    </Figure>
  )
}
