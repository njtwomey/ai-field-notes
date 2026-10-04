import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'

const T = 1000
const BURN_IN = 200
const MAX_LAG = 30

/** Sample autocorrelations at lags 1..maxLag. */
function acf(x: number[], maxLag: number): number[] {
  const n = x.length
  const m = x.reduce((a, b) => a + b, 0) / n
  const d = x.map((v) => v - m)
  const c0 = d.reduce((a, v) => a + v * v, 0)
  return Array.from({ length: maxLag }, (_, i) => {
    const k = i + 1
    let s = 0
    for (let t = k; t < n; t++) s += d[t] * d[t - k]
    return s / c0
  })
}

export function GarchSimulator() {
  const state = useFigureState({
    persistence: float(0.95, {
      min: 0,
      max: 0.995,
      step: 0.005,
      label: 'persistence α + β',
      format: (v) => v.toFixed(3),
    }),
    alphaParam: float(0.1, { min: 0, max: 0.3, step: 0.01, label: 'reaction α', format: (v) => v.toFixed(2) }),
    seed: int(3, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  // alpha cannot exceed the persistence, since beta = persistence - alpha must stay non-negative.
  const alpha = Math.min(state.alphaParam, state.persistence)
  const beta = state.persistence - alpha
  // Variance targeting: omega makes the unconditional variance 1 for every (alpha, beta).
  const omega = 1 - state.persistence

  const sim = useMemo(() => {
    const g = stream(state.seed)
    const r: number[] = []
    const s2: number[] = []
    let prevR2 = 1
    let prevS2 = 1
    for (let t = 0; t < T + BURN_IN; t++) {
      const v = omega + alpha * prevR2 + beta * prevS2
      const x = Math.sqrt(v) * normal(g)
      if (t >= BURN_IN) {
        r.push(x)
        s2.push(v)
      }
      prevR2 = x * x
      prevS2 = v
    }
    const r2 = r.map((v) => v * v)
    const m2 = r2.reduce((a, b) => a + b, 0) / T
    const m4 = r2.reduce((a, b) => a + b * b, 0) / T
    return { r, s2, acfR: acf(r, MAX_LAG), acfR2: acf(r2, MAX_LAG), kurtosis: m4 / (m2 * m2) }
  }, [omega, alpha, beta, state.seed])

  const charts = useMemo(() => {
    const t = Array.from({ length: T }, (_, i) => i + 1)
    const band = sim.s2.map((v) => 2 * Math.sqrt(v))
    const top = Math.max(4, Math.ceil(Math.max(...sim.r.map(Math.abs), ...band)))
    const returns = [
      { name: 'return r_t', x: t, y: sim.r, slot: 0 },
      { name: '±2σ_t', x: t, y: band, slot: 1 },
      { name: '±2σ_t', x: t, y: band.map((v) => -v), slot: 1 },
    ] as const
    const lags = Array.from({ length: MAX_LAG }, (_, i) => i + 1)
    const bound = 1.96 / Math.sqrt(T)
    const acfs = [
      { name: 'ACF of r_t', x: lags, y: sim.acfR, slot: 0 },
      { name: 'ACF of r_t²', x: lags, y: sim.acfR2, slot: 1 },
      { name: '±1.96/√T', x: [1, MAX_LAG], y: [bound, bound], muted: true, dashed: true },
      { name: '±1.96/√T', x: [1, MAX_LAG], y: [-bound, -bound], muted: true, dashed: true },
    ] as const
    return { returns, acfs, top }
  }, [sim])

  const p = state.persistence
  const denom = 1 - p * p - 2 * alpha * alpha
  const kurtosis = denom > 0 ? formatNumber((3 * (1 - p * p)) / denom) : 'infinite'
  const halfLife = p > 0 ? formatNumber(Math.log(0.5) / Math.log(p)) : '0'

  const xAxis = useAxis({ label: 't', range: [1, T] })
  const yAxis = useAxis({ label: 'r_t', range: [-charts.top, charts.top] })
  const xAxis2 = useAxis({ label: 'lag k', range: [1, MAX_LAG] })
  const yAxis2 = useAxis({ label: 'autocorrelation', range: [-0.2, undefined], hold: 'union' })
  return (
    <Figure
      title="Simulating GARCH(1,1)"
      state={state}
      caption="Returns r_t = σ_t ε_t with σ_t² = ω + α r_{t−1}² + β σ_{t−1}² and Gaussian ε_t. ω is set to 1 − α − β so that the unconditional variance is 1 in every setting. The persistence α + β sets how long a burst of volatility lasts; α sets how sharply σ_t reacts to one large return. The lower panel shows the sample autocorrelation of the returns (near zero) and of the squared returns (positive and slowly decaying). At α = 0 the variance is constant and both autocorrelations vanish."

      readouts={
        <>
          <Readout label="α, β, ω" value={`${alpha.toFixed(2)}, ${beta.toFixed(3)}, ${omega.toFixed(3)}`} />
          <Readout label="half-life of a variance shock" value={`${halfLife} steps`} />
          <Readout label="kurtosis: theory" value={kurtosis} />
          <Readout label="kurtosis: this sample" value={formatNumber(sim.kurtosis)} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={240}>
          <Curve {...charts.returns[0]} />
          <Curve {...charts.returns[1]} />
          <Curve {...charts.returns[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          <Curve {...charts.acfs[0]} />
          <Curve {...charts.acfs[1]} />
          <Curve {...charts.acfs[2]} />
          <Curve {...charts.acfs[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
