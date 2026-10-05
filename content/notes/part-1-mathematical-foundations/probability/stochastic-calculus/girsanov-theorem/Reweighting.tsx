import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { gaussPdf, normals } from '../_shared/sde'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const N = 5000
const LO = -4
const HI = 6
const BINS = 50
const XS = toFlat(linspace(LO, HI, 201))

/**
 * Brownian paths on [0, 1] reweighted by Girsanov's likelihood ratio Z = exp(μW₁ − μ²/2). The weighted histogram of W₁
 * matches N(μ, 1), the law of Brownian motion with drift μ, although no path was simulated with a drift.
 */
export function Reweighting() {
  const state = useFigureState({
    mu: float(1.5, { min: -2, max: 3, step: 0.05, label: 'drift μ' }),
  })
  const w1 = useMemo(() => normals(N, 51), [])

  const { series, ess, mean } = useMemo(() => {
    const m = state.mu
    const weights = w1.map((w) => Math.exp(m * w - (m * m) / 2))
    const total = weights.reduce((a, b) => a + b, 0)
    const width = (HI - LO) / BINS
    const counts = new Array<number>(BINS).fill(0)
    let s1 = 0
    let s2 = 0
    w1.forEach((w, i) => {
      const b = Math.floor((w - LO) / width)
      if (b >= 0 && b < BINS) counts[b] += weights[i]
      s1 += weights[i] * w
      s2 += weights[i] ** 2
    })
    const x = counts.map((_, b) => LO + (b + 0.5) * width)
    const out: SeriesSpec[] = [
      { name: 'unweighted W₁ ~ N(0, 1)', type: 'line', x: XS, y: XS.map((v) => gaussPdf(v, 0, 1)), muted: true },
      { name: 'Brownian samples, weighted by Z', type: 'bar', x, y: counts.map((c) => c / (total * width)), slot: 0 },
      {
        name: 'N(μ, 1): Brownian motion with drift μ',
        type: 'line',
        x: XS,
        y: XS.map((v) => gaussPdf(v, m, 1)),
        emphasis: true,
        dashed: true,
      },
    ]
    return { series: out, ess: total ** 2 / s2, mean: s1 / total }
  }, [state.mu, w1])

  const xAxis = useAxis({ label: 'W₁', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, 0.6] })
  return (
    <Figure
      title="Changing the drift by reweighting paths"
      state={state}
      caption="5,000 values of W₁ from standard Brownian motion, each weighted by the Girsanov likelihood ratio exp(μW₁ − μ²/2). The weighted histogram is the law N(μ, 1) of Brownian motion with drift μ. The effective sample size shows the cost: as μ grows, a few paths that happened to drift the right way carry almost all the weight."

      readouts={
        <>
          <Readout label="weighted mean of W₁" value={formatNumber(mean)} />
          <Readout label="effective sample size" value={formatNumber(ess)} />
          <Readout label="theory N e^{−μ²}" value={formatNumber(N * Math.exp(-(state.mu ** 2)))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
