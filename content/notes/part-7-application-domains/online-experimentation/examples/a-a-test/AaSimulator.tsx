import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'
import { Binomial } from 'aifn/probability/distributions'
import { sampleBinomial, twoProportionZ } from '../_shared/ab'
import { normalPdf } from 'aifn/numerics/special'

/** P(S ≤ k) for S ~ Binomial(n, p). */
const binomialLower = (k: number, n: number, p: number) => Binomial(n, p).cdf(k) as number

const P_BINS = 20
const D_BINS = 30

/** The central 95% range of Binomial(N, α): how many A/A tests should come out significant. */
function expectedRange(experiments: number, alpha: number): [number, number] {
  let lo = 0
  while (binomialLower(lo, experiments, alpha) < 0.025) lo++
  let hi = lo
  while (binomialLower(hi, experiments, alpha) < 0.975) hi++
  return [lo, hi]
}

/**
 * Many A/A experiments: both arms convert at the same rate, and each is analysed with the two-proportion z-test. The
 * p-values should be uniform, about α of them significant, and the observed differences spread as the formula says.
 */
export function AaSimulator() {
  const state = useFigureState({
    n: int(10000, { min: 500, max: 50000, step: 500, label: 'users per arm n' }),
    rate: float(0.05, { min: 0.01, max: 0.5, step: 0.01, label: 'conversion rate p' }),
    experiments: int(1000, { min: 100, max: 2000, step: 100, label: 'experiments' }),
    alpha: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'significance level α' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })

  const r = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    const pValues: number[] = []
    const diffs: number[] = []
    let significant = 0
    let covered = 0
    for (let e = 0; e < state.experiments; e++) {
      const t = twoProportionZ(
        sampleBinomial(state.n, state.rate, uniform),
        state.n,
        sampleBinomial(state.n, state.rate, uniform),
        state.n,
        state.alpha,
      )
      pValues.push(t.p)
      diffs.push(t.diff)
      if (t.p < state.alpha) significant++
      if (t.ci[0] <= 0 && t.ci[1] >= 0) covered++
    }
    const sdTheory = Math.sqrt((2 * state.rate * (1 - state.rate)) / state.n)
    const meanDiff = diffs.reduce((s, d) => s + d, 0) / state.experiments
    const sdObserved = Math.sqrt(diffs.reduce((s, d) => s + (d - meanDiff) ** 2, 0) / (state.experiments - 1))

    const pCounts = new Array(P_BINS).fill(0)
    for (const p of pValues) pCounts[Math.min(P_BINS - 1, Math.floor(p * P_BINS))]++
    const pSeries = [
      {
        name: 'A/A p-values',
        x: pCounts.map((_, i) => (i + 0.5) / P_BINS),
        y: pCounts.map((c) => (c * P_BINS) / state.experiments),
        slot: 0,
      },
      { name: 'uniform', x: [0, 1], y: [1, 1], dashed: true, slot: 2 },
    ] as const

    // Differences in percentage points, against the normal curve the formula predicts.
    const span = 4 * sdTheory
    const width = (2 * span) / D_BINS
    const dCounts = new Array(D_BINS).fill(0)
    for (const d of diffs) {
      const i = Math.floor((d + span) / width)
      if (i >= 0 && i < D_BINS) dCounts[i]++
    }
    const grid = Array.from({ length: 81 }, (_, i) => -span + (2 * span * i) / 80)
    const dSeries = [
      {
        name: 'observed B − A',
        x: dCounts.map((_, i) => 100 * (-span + (i + 0.5) * width)),
        y: dCounts.map((c) => c / (state.experiments * width * 100)),
        slot: 0,
      },
      {
        name: 'N(0, 2p(1 − p)/n)',
        x: grid.map((d) => 100 * d),
        y: grid.map((d) => normalPdf(d / sdTheory) / (sdTheory * 100)),
        slot: 1,
      },
    ] as const
    return { significant, covered, sdTheory, sdObserved, pSeries, dSeries }
  }, [state.n, state.rate, state.experiments, state.alpha, state.seed])

  const [lo, hi] = useMemo(() => expectedRange(state.experiments, state.alpha), [state.experiments, state.alpha])

  const xAxis = useAxis({ label: 'p-value', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'observed B − A (percentage points)', hold: 'union' })
  const yAxis2 = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A thousand experiments with nothing to find"
      state={state}
      caption="Each experiment splits n users per arm, both converting at the same rate p, and runs a two-sided two-proportion z-test. Left: the p-values are close to uniform, so the share below α is about α; drag the line labelled α. Right: the observed differences B − A, in percentage points, against the normal curve with standard deviation √(2p(1 − p)/n). Differences of this size are what chance alone produces."

      readouts={
        <>
          <Readout label="significant" value={`${r.significant} of ${state.experiments}`} />
          <Readout
            label="expected (95% range)"
            value={`${Math.round(state.alpha * state.experiments)} (${lo}–${hi})`}
          />
          <Readout label="intervals covering 0" value={`${((100 * r.covered) / state.experiments).toFixed(1)}%`} />
          <Readout label="sd of B − A, observed" value={`${formatNumber(100 * r.sdObserved)} pp`} />
          <Readout label="sd of B − A, formula" value={`${formatNumber(100 * r.sdTheory)} pp`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Bars {...r.pSeries[0]} />
          <Curve {...r.pSeries[1]} />
          <Handle {...state.handle('alpha', { label: 'α' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Bars {...r.dSeries[0]} />
          <Curve {...r.dSeries[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
