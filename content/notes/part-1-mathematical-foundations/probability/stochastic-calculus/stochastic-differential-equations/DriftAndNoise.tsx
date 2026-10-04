import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  slider,
  type SeriesSpec,
  useAxis,
  useFigureState,
  when,
} from 'aifn-render'
import { joinPaths } from '../_shared/sde'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf } from 'aifn/numerics/special'
import { normal, stream } from 'aifn/foundation/random'

/** Brownian paths simulated once; the paths slider draws the first few. */
const MAX_PATHS = 50
const STEPS = 250
const T = 5
const TIMES = toFlat(linspace(0, T, STEPS + 1))
const Z90 = 1.6449 // 95th percentile of N(0, 1)

type Model = 'drift' | 'gbm'

/**
 * Exact solutions of two SDEs driven by the same Brownian paths. Brownian motion with drift, X = x₀ + μt + σW, has a
 * Gaussian band that widens like √t. Geometric Brownian motion, X = x₀ exp((μ − σ²/2)t + σW), has a mean and a median
 * that separate as noise grows.
 */
export function DriftAndNoise() {
  const state = useFigureState({
    model: choice<Model>(
      [
        { value: 'drift', label: 'dX = μ dt + σ dW' },
        { value: 'gbm', label: 'dX = μX dt + σX dW' },
      ],
      'drift',
      { label: 'SDE' },
    ),
    count: int(25, { min: 1, max: MAX_PATHS, step: 1, label: 'paths', suggestions: [1, 5, 25, 50] }),
    mu: slider(-1, 1, 0.3, { step: 0.05, label: 'drift μ', when: when('model', 'drift') }),
    sigma: slider(0, 1.5, 0.8, { step: 0.05, label: 'noise σ', when: when('model', 'drift') }),
    x0: slider(-3, 3, 0, { step: 0.1, label: 'start x₀', when: when('model', 'drift') }),
    gmu: slider(-0.3, 0.4, 0.1, { step: 0.01, label: 'drift μ', when: when('model', 'gbm') }),
    gsigma: slider(0, 0.8, 0.5, { step: 0.01, label: 'noise σ', when: when('model', 'gbm') }),
    gx0: slider(0.2, 3, 1, { step: 0.05, label: 'start x₀', when: when('model', 'gbm') }),
  })
  const x0 = state.model === 'drift' ? state.x0 : state.gx0
  const mu = state.model === 'drift' ? state.mu : state.gmu
  const sigma = state.model === 'drift' ? state.sigma : state.gsigma

  const w = useMemo(() => {
    const rs = stream(5)
    const sd = Math.sqrt(T / STEPS)
    return Array.from({ length: MAX_PATHS }, () => {
      const out = [0]
      for (let k = 1; k <= STEPS; k++) out.push(out[k - 1] + sd * normal(rs))
      return out
    })
  }, [])

  const series = useMemo<SeriesSpec[]>(() => {
    const drawn = w.slice(0, state.count)
    const thin = state.count > 1
    if (state.model === 'drift') {
      const paths = joinPaths(drawn.map((wp) => ({ x: TIMES, y: wp.map((wk, k) => x0 + mu * TIMES[k] + sigma * wk) })))
      return [
        { name: 'paths', type: 'line', ...paths, slot: 3, thin },
        { name: 'mean x₀ + μt', type: 'line', x: TIMES, y: TIMES.map((t) => x0 + mu * t), slot: 0 },
        {
          name: '5% and 95% quantiles',
          type: 'line',
          ...joinPaths(
            [-1, 1].map((s) => ({ x: TIMES, y: TIMES.map((t) => x0 + mu * t + s * Z90 * sigma * Math.sqrt(t)) })),
          ),
          slot: 1,
          dashed: true,
        },
      ]
    }
    const drift = mu - (sigma * sigma) / 2
    const paths = joinPaths(
      drawn.map((wp) => ({ x: TIMES, y: wp.map((wk, k) => x0 * Math.exp(drift * TIMES[k] + sigma * wk)) })),
    )
    return [
      { name: 'paths', type: 'line', ...paths, slot: 3, thin },
      { name: 'mean x₀e^{μt}', type: 'line', x: TIMES, y: TIMES.map((t) => x0 * Math.exp(mu * t)), slot: 0 },
      {
        name: 'median x₀e^{(μ − σ²/2)t}',
        type: 'line',
        x: TIMES,
        y: TIMES.map((t) => x0 * Math.exp(drift * t)),
        slot: 2,
      },
      {
        name: '5% and 95% quantiles',
        type: 'line',
        ...joinPaths(
          [-1, 1].map((s) => ({
            x: TIMES,
            y: TIMES.map((t) => x0 * Math.exp(drift * t + s * Z90 * sigma * Math.sqrt(t))),
          })),
        ),
        slot: 1,
        dashed: true,
      },
    ]
  }, [state.model, w, x0, mu, sigma, state.count])

  const lossProb = useMemo(() => {
    // P(X_T < x₀) for GBM: Φ(−(μ − σ²/2)√T / σ).
    if (state.model !== 'gbm' || sigma === 0) return null
    const z = (-(mu - (sigma * sigma) / 2) * Math.sqrt(T)) / sigma
    return normalCdf(z)
  }, [state.model, mu, sigma])

  const xAxis = useAxis({ label: 't', range: [0, T] })
  const yAxis = useAxis({ label: 'X_t', range: state.model === 'drift' ? [-6, 8] : [0, 6] })
  return (
    <Figure
      title="Drift plus noise"
      state={state}
      caption="Paths of each SDE, drawn as light lines and driven by the same Brownian paths; the paths slider sets how many (25 by default). Brownian motion with drift moves its mean in a straight line and its 90% band widens like √t. Geometric Brownian motion multiplies the noise by the current value: its mean grows like e^{μt}, but its median grows only like e^{(μ − σ²/2)t}, and with σ²/2 > μ most paths decay while the mean still grows. Drag the start point at t = 0."
      readouts={
        state.model === 'drift' ? (
          <>
            <Readout label="mean at t = 5" value={formatNumber(x0 + mu * T)} />
            <Readout label="sd at t = 5" value={formatNumber(sigma * Math.sqrt(T))} />
          </>
        ) : (
          <>
            <Readout label="mean at t = 5" value={formatNumber(x0 * Math.exp(mu * T))} />
            <Readout label="median at t = 5" value={formatNumber(x0 * Math.exp((mu - (sigma * sigma) / 2) * T))} />
            <Readout label="P(X₅ < x₀)" value={lossProb === null ? '–' : formatNumber(lossProb)} />
          </>
        )
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(series)}
        <Handle
          kind="point"
          at={[0, x0]}
          label="x₀"
          onDrag={([, y]) => state.set(state.model === 'drift' ? 'x0' : 'gx0', y)}
        />
      </Plot>
    </Figure>
  )
}
