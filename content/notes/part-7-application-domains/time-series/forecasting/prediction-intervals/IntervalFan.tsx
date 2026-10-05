import { useMemo } from 'react'
import {
  choice,
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
import { normal, stream, type Stream, uniform } from 'aifn-compute/foundation/random'
import { normalQuantile } from 'aifn-compute/numerics/special'

const N_HISTORY = 200
const SHOWN = 40
const H = 20
const N_BOOT = 1000
const N_TRUTH = 2000

type Noise = 'gaussian' | 't3'

/** A unit-variance shock: Gaussian, or Student t with 3 degrees of freedom rescaled to variance 1. */
function shock(g: Stream, noise: Noise) {
  const z = normal(g)
  if (noise === 'gaussian') return z
  const chi2 = normal(g) ** 2 + normal(g) ** 2 + normal(g) ** 2
  return z / Math.sqrt(chi2 / 3) / Math.sqrt(3)
}

function quantile(sorted: number[], p: number) {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))
  return sorted[i]
}

export function IntervalFan() {
  const state = useFigureState({
    phi: float(0.8, { min: 0, max: 0.99, step: 0.01, label: 'AR coefficient φ', format: (v) => v.toFixed(2) }),
    noise: choice<Noise>(
      [
        { value: 'gaussian', label: 'Gaussian' },
        { value: 't3', label: 'Student t, 3 df' },
      ],
      'gaussian',
      { label: 'shocks' },
    ),
    level: choice<'0.8' | '0.95' | '0.99'>(
      [
        { value: '0.8', label: '80%' },
        { value: '0.95', label: '95%' },
        { value: '0.99', label: '99%' },
      ],
      '0.95',
      { label: 'nominal coverage' },
    ),
    drawn: int(20, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
    seed: int(2, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const coverage = Number(state.level)

  const result = useMemo(() => {
    const g = stream(state.seed)
    const f = state.phi
    // History of an AR(1) with mean 0 and unit-variance shocks. The shocks double as the fitted residuals, since the
    // model is known here; the bootstrap resamples them.
    const y = [0]
    const resid: number[] = []
    for (let t = 1; t < N_HISTORY; t++) {
      const e = shock(g, state.noise)
      resid.push(e)
      y.push(f * y[t - 1] + e)
    }
    const last = y[N_HISTORY - 1]
    const z = normalQuantile(0.5 + coverage / 2)
    const mean: number[] = []
    const aLo: number[] = []
    const aHi: number[] = []
    for (let h = 1; h <= H; h++) {
      const m = f ** h * last
      const sd = Math.sqrt(f === 0 ? 1 : (1 - f ** (2 * h)) / (1 - f * f))
      mean.push(m)
      aLo.push(m - z * sd)
      aHi.push(m + z * sd)
    }
    const simulate = (draw: () => number, n: number) => {
      const paths: number[][] = Array.from({ length: H }, () => [])
      for (let b = 0; b < n; b++) {
        let v = last
        for (let h = 0; h < H; h++) {
          v = f * v + draw()
          paths[h].push(v)
        }
      }
      return paths
    }
    const boot = simulate(() => resid[Math.floor(uniform(g) * resid.length)], N_BOOT).map((p) =>
      p.sort((a, b) => a - b),
    )
    const bLo = boot.map((p) => quantile(p, 0.5 - coverage / 2))
    const bHi = boot.map((p) => quantile(p, 0.5 + coverage / 2))
    // Fresh futures from the true process, to measure how often each interval contains the outcome.
    const truth = simulate(() => shock(g, state.noise), N_TRUTH)
    let inA = 0
    let inB = 0
    truth.forEach((p, h) =>
      p.forEach((v) => {
        if (v >= aLo[h] && v <= aHi[h]) inA++
        if (v >= bLo[h] && v <= bHi[h]) inB++
      }),
    )
    return {
      history: y.slice(-SHOWN),
      last,
      resid,
      mean,
      aLo,
      aHi,
      bLo,
      bHi,
      coverA: inA / (N_TRUTH * H),
      coverB: inB / (N_TRUTH * H),
    }
  }, [state.phi, state.seed, state.noise, coverage])

  // Drawn bootstrap futures, separate from the 1000 behind the interval: path k has its own stream, so adding paths
  // leaves the existing ones and the intervals unchanged.
  const futures = useMemo((): SeriesSpec[] => {
    const many = state.drawn > 1
    const x = Array.from({ length: H + 1 }, (_, i) => i)
    return Array.from({ length: state.drawn }, (_, k) => {
      const g = stream(state.seed * 1000 + k)
      let v = result.last
      const y = [v]
      for (let h = 0; h < H; h++) {
        v = state.phi * v + result.resid[Math.floor(uniform(g) * result.resid.length)]
        y.push(v)
      }
      return { name: many ? 'bootstrap paths' : 'bootstrap path', type: 'line', x, y, slot: 2, thin: many }
    })
  }, [result, state.phi, state.seed, state.drawn])

  const series = useMemo((): SeriesSpec[] => {
    const past = Array.from({ length: SHOWN }, (_, i) => i - SHOWN + 1)
    const ahead = Array.from({ length: H }, (_, i) => i + 1)
    return [
      ...futures,
      { name: 'observed', type: 'line', x: past, y: result.history, emphasis: true },
      { name: 'point forecast', type: 'line', x: ahead, y: result.mean, slot: 0 },
      { name: 'analytic Gaussian interval', type: 'line', x: ahead, y: result.aLo, slot: 1, dashed: true },
      { name: 'analytic Gaussian interval', type: 'line', x: ahead, y: result.aHi, slot: 1, dashed: true },
      { name: 'bootstrap interval', type: 'line', x: ahead, y: result.bLo, slot: 2 },
      { name: 'bootstrap interval', type: 'line', x: ahead, y: result.bHi, slot: 2 },
    ]
  }, [result, futures])

  const stationarySd = 1 / Math.sqrt(1 - state.phi ** 2)

  const xAxis = useAxis({ label: 'time relative to forecast origin', range: [-SHOWN + 1, H] })
  const yAxis = useAxis({ label: 'y', hold: 'union' })
  return (
    <Figure
      title="Prediction intervals for an AR(1)"
      state={state}
      caption="An AR(1) series y_t = φ y_{t−1} + ε_t with unit-variance shocks is observed up to time 0 and forecast 20 steps ahead. The analytic interval assumes Gaussian shocks and has half-width z times the standard deviation √((1 − φ^{2h})/(1 − φ²)), which grows with h and levels off at the stationary standard deviation. The bootstrap interval simulates 1000 future paths by resampling the model's residuals and takes their quantiles; the light lines are further paths drawn the same way, and the paths slider sets how many. Coverage is measured on 2000 fresh futures from the true process. With Student t shocks the Gaussian interval over-covers at 80% and under-covers at 99%, most clearly at short horizons, where one shock dominates the error; the bootstrap follows the shock distribution."

      readouts={
        <>
          <Readout label="coverage, analytic" value={`${formatNumber(100 * result.coverA)}%`} />
          <Readout label="coverage, bootstrap" value={`${formatNumber(100 * result.coverB)}%`} />
          <Readout label="stationary sd" value={formatNumber(stationarySd)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
