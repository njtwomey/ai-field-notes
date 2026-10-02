import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { normalCdf } from '@/lib/math/special'
import { joinPaths } from '../_shared/sde'

/** Brownian paths simulated once; the paths slider draws the first few. */
const MAX_PATHS = 50
const STEPS = 250
const T = 5
const TIMES = linspace(0, T, STEPS + 1)
const Z90 = 1.6449 // 95th percentile of N(0, 1)

type Model = 'drift' | 'gbm'

/**
 * Exact solutions of two SDEs driven by the same Brownian paths. Brownian motion with drift, X = x₀ + μt + σW, has a
 * Gaussian band that widens like √t. Geometric Brownian motion, X = x₀ exp((μ − σ²/2)t + σW), has a mean and a median
 * that separate as noise grows.
 */
export function DriftAndNoise() {
  const [model, setModel] = useState<Model>('drift')
  const count = useParam(25, { min: 1, max: MAX_PATHS, step: 1 })
  const bm = {
    x0: useParam(0, { min: -3, max: 3, step: 0.1 }),
    mu: useParam(0.3, { min: -1, max: 1, step: 0.05 }),
    sigma: useParam(0.8, { min: 0, max: 1.5, step: 0.05 }),
  }
  const gbm = {
    x0: useParam(1, { min: 0.2, max: 3, step: 0.05 }),
    mu: useParam(0.1, { min: -0.3, max: 0.4, step: 0.01 }),
    sigma: useParam(0.5, { min: 0, max: 0.8, step: 0.01 }),
  }
  const p = model === 'drift' ? bm : gbm
  const x0 = p.x0.value
  const mu = p.mu.value
  const sigma = p.sigma.value

  const w = useMemo(() => {
    const { normal } = rng(5)
    const sd = Math.sqrt(T / STEPS)
    return Array.from({ length: MAX_PATHS }, () => {
      const out = [0]
      for (let k = 1; k <= STEPS; k++) out.push(out[k - 1] + sd * normal())
      return out
    })
  }, [])

  const series = useMemo<XYSeries[]>(() => {
    const drawn = w.slice(0, count.value)
    const thin = count.value > 1
    if (model === 'drift') {
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
  }, [model, w, x0, mu, sigma, count.value])

  const lossProb = useMemo(() => {
    // P(X_T < x₀) for GBM: Φ(−(μ − σ²/2)√T / σ).
    if (model !== 'gbm' || sigma === 0) return null
    const z = (-(mu - (sigma * sigma) / 2) * Math.sqrt(T)) / sigma
    return normalCdf(z)
  }, [model, mu, sigma])

  return (
    <Interactive
      title="Drift plus noise"
      caption="Paths of each SDE, drawn as light lines and driven by the same Brownian paths; the paths slider sets how many (25 by default). Brownian motion with drift moves its mean in a straight line and its 90% band widens like √t. Geometric Brownian motion multiplies the noise by the current value: its mean grows like e^{μt}, but its median grows only like e^{(μ − σ²/2)t}, and with σ²/2 > μ most paths decay while the mean still grows. Drag the start point at t = 0."
      controls={
        <>
          <ParamChoice
            label="SDE"
            value={model}
            onChange={setModel}
            options={[
              { value: 'drift', label: 'dX = μ dt + σ dW' },
              { value: 'gbm', label: 'dX = μX dt + σX dW' },
            ]}
          />
          <ParamSlider label="drift μ" param={p.mu} />
          <ParamSlider label="noise σ" param={p.sigma} />
          <ParamSlider label="start x₀" param={p.x0} />
          <ParamSlider label="paths" param={count} withArrows format={(v) => String(v)} />
        </>
      }
      readout={
        model === 'drift' ? (
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
      <XYChart
        height={320}
        xLabel="t"
        yLabel="X_t"
        series={series}
        xRange={[0, T]}
        yRange={model === 'drift' ? [-6, 8] : [0, 6]}
        handles={[{ kind: 'point', at: [0, x0], label: 'x₀', onDrag: ([, y]) => p.x0.set(y) }]}
      />
    </Interactive>
  )
}
