import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  Figure,
  float,
  formatNumber,
  Plot,
  Plots,
  Readout,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream } from 'aifn/foundation/random'
import { normalCdf } from 'aifn/numerics/special'

const PER_GROUP = 3000
const BINS = 10
const CENTRES = Array.from({ length: BINS }, (_, i) => (i + 0.5) / BINS)

/**
 * Quantile calibration is marginal; distribution calibration conditions on the prediction. Two groups of cases have true
 * noise N(0, 0.5²) and N(0, 1.5²). The model predicts N(0, s_A²) for the first group and N(0, s_B²) for the second. PIT
 * histograms are shown pooled and for each predicted distribution, optionally after a global quantile recalibration
 * (the empirical cdf of the pooled PIT values, which is what isotonic recalibration converges to).
 */
export function PitByPrediction() {
  const state = useFigureState({
    sA: float(1.5, { min: 0.3, max: 2, step: 0.05, label: 'predicted sd, first kind (true 0.5)' }),
    sB: float(0.5, { min: 0.3, max: 2, step: 0.05, label: 'predicted sd, second kind (true 1.5)' }),
    recal: setting(true, 'global quantile recalibration'),
  })

  const r = useMemo(() => {
    const g = stream(21)
    const pitA: number[] = []
    const pitB: number[] = []
    for (let i = 0; i < PER_GROUP; i++) pitA.push(normalCdf((0.5 * normal(g)) / state.sA))
    for (let i = 0; i < PER_GROUP; i++) pitB.push(normalCdf((1.5 * normal(g)) / state.sB))
    let a = pitA
    let b = pitB
    if (state.recal) {
      const pooled = [...pitA, ...pitB].sort((x, y) => x - y)
      // Empirical cdf of the pooled PIT values: the fraction of pooled values at or below u.
      const ecdf = (u: number) => {
        let lo = 0
        let hi = pooled.length
        while (lo < hi) {
          const mid = (lo + hi) >> 1
          if (pooled[mid] <= u) lo = mid + 1
          else hi = mid
        }
        return lo / pooled.length
      }
      a = pitA.map(ecdf)
      b = pitB.map(ecdf)
    }
    const hist = (v: number[]) => {
      const h = new Array<number>(BINS).fill(0)
      v.forEach((u) => h[Math.min(BINS - 1, Math.floor(u * BINS))]++)
      return h.map((c) => (c / v.length) * BINS)
    }
    const cover = (v: number[]) => v.filter((u) => u > 0.05 && u < 0.95).length / v.length
    return {
      all: hist([...a, ...b]),
      a: hist(a),
      b: hist(b),
      coverAll: cover([...a, ...b]),
      coverA: cover(a),
      coverB: cover(b),
    }
  }, [state.sA, state.sB, state.recal])

  const xAll = useAxis({ label: 'PIT, all cases', range: [0, 1] })
  const xA = useAxis({ label: 'PIT, prediction A', range: [0, 1] })
  const xB = useAxis({ label: 'PIT, prediction B', range: [0, 1] })
  const yAxis = useAxis({ label: 'density', range: [0, 4] })
  const panel = (x: typeof xAll, name: string, y: number[], slot: number) => (
    <Plot x={x} y={yAxis}>
      <Bars name={name} x={CENTRES} y={y} width={1 / BINS} slot={slot} />
      <Annotation y={1} text="uniform" dashed muted />
    </Plot>
  )

  return (
    <Figure
      title="Calibrated overall, miscalibrated for each prediction"
      state={state}
      caption="Two kinds of case have noise standard deviations 0.5 and 1.5. The model predicts a Gaussian with standard deviation s_A for the first kind and s_B for the second; at the defaults it has them swapped. Each panel is a PIT histogram, flat when calibrated. With global quantile recalibration switched on, the pooled histogram is flat, yet the histograms for each predicted distribution are not: quantile calibration holds, distribution calibration fails. Set s_A = 0.5 and s_B = 1.5 to make the model distribution-calibrated."
      readouts={
        <>
          <Readout label="90% interval coverage, all" value={formatNumber(r.coverAll)} />
          <Readout label="given prediction A" value={formatNumber(r.coverA)} />
          <Readout label="given prediction B" value={formatNumber(r.coverB)} />
        </>
      }
    >
      <Plots cols={3}>
        {panel(xAll, 'all cases', r.all, 0)}
        {panel(xA, 'prediction A', r.a, 1)}
        {panel(xB, 'prediction B', r.b, 2)}
      </Plots>
    </Figure>
  )
}
