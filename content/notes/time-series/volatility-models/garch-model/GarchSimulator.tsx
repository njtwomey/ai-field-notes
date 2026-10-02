import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

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
  const persistence = useParam(0.95, { min: 0, max: 0.995, step: 0.005 })
  const alphaParam = useParam(0.1, { min: 0, max: 0.3, step: 0.01 })
  const seed = useParam(3, { min: 1, max: 30, step: 1 })

  // alpha cannot exceed the persistence, since beta = persistence - alpha must stay non-negative.
  const alpha = Math.min(alphaParam.value, persistence.value)
  const beta = persistence.value - alpha
  // Variance targeting: omega makes the unconditional variance 1 for every (alpha, beta).
  const omega = 1 - persistence.value

  const sim = useMemo(() => {
    const g = rng(seed.value)
    const r: number[] = []
    const s2: number[] = []
    let prevR2 = 1
    let prevS2 = 1
    for (let t = 0; t < T + BURN_IN; t++) {
      const v = omega + alpha * prevR2 + beta * prevS2
      const x = Math.sqrt(v) * g.normal()
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
  }, [omega, alpha, beta, seed.value])

  const charts = useMemo(() => {
    const t = Array.from({ length: T }, (_, i) => i + 1)
    const band = sim.s2.map((v) => 2 * Math.sqrt(v))
    const top = Math.max(4, Math.ceil(Math.max(...sim.r.map(Math.abs), ...band)))
    const returns: XYSeries[] = [
      { name: 'return r_t', type: 'line', x: t, y: sim.r, slot: 0 },
      { name: '±2σ_t', type: 'line', x: t, y: band, slot: 1 },
      { name: '±2σ_t', type: 'line', x: t, y: band.map((v) => -v), slot: 1 },
    ]
    const lags = Array.from({ length: MAX_LAG }, (_, i) => i + 1)
    const bound = 1.96 / Math.sqrt(T)
    const acfs: XYSeries[] = [
      { name: 'ACF of r_t', type: 'line', x: lags, y: sim.acfR, slot: 0 },
      { name: 'ACF of r_t²', type: 'line', x: lags, y: sim.acfR2, slot: 1 },
      { name: '±1.96/√T', type: 'line', x: [1, MAX_LAG], y: [bound, bound], muted: true, dashed: true },
      { name: '±1.96/√T', type: 'line', x: [1, MAX_LAG], y: [-bound, -bound], muted: true, dashed: true },
    ]
    return { returns, acfs, top }
  }, [sim])

  const p = persistence.value
  const denom = 1 - p * p - 2 * alpha * alpha
  const kurtosis = denom > 0 ? formatNumber((3 * (1 - p * p)) / denom) : 'infinite'
  const halfLife = p > 0 ? formatNumber(Math.log(0.5) / Math.log(p)) : '0'

  return (
    <Interactive
      title="Simulating GARCH(1,1)"
      caption="Returns r_t = σ_t ε_t with σ_t² = ω + α r_{t−1}² + β σ_{t−1}² and Gaussian ε_t. ω is set to 1 − α − β so that the unconditional variance is 1 in every setting. The persistence α + β sets how long a burst of volatility lasts; α sets how sharply σ_t reacts to one large return. The lower panel shows the sample autocorrelation of the returns (near zero) and of the squared returns (positive and slowly decaying). At α = 0 the variance is constant and both autocorrelations vanish."
      controls={
        <>
          <ParamSlider label="persistence α + β" param={persistence} format={(v) => v.toFixed(3)} />
          <ParamSlider label="reaction α" param={alphaParam} format={(v) => v.toFixed(2)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="α, β, ω" value={`${alpha.toFixed(2)}, ${beta.toFixed(3)}, ${omega.toFixed(3)}`} />
          <Readout label="half-life of a variance shock" value={`${halfLife} steps`} />
          <Readout label="kurtosis: theory" value={kurtosis} />
          <Readout label="kurtosis: this sample" value={formatNumber(sim.kurtosis)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart
          series={charts.returns}
          xLabel="t"
          yLabel="r_t"
          xRange={[1, T]}
          yRange={[-charts.top, charts.top]}
          height={240}
        />
        <XYChart
          series={charts.acfs}
          xLabel="lag k"
          yLabel="autocorrelation"
          xRange={[1, MAX_LAG]}
          yRange={[-0.2, undefined]}
          height={200}
        />
      </div>
    </Interactive>
  )
}
