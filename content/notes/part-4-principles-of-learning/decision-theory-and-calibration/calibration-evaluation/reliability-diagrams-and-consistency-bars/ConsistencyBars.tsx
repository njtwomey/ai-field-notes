import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  int,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const RESAMPLES = 300

type Binning = 'width' | 'mass'

/**
 * A reliability diagram with Bröcker and Smith's consistency bars. True probabilities are σ(z), z ~ N(0, 1.5²); the
 * model reports σ(k z). For each bin, labels are resampled from the model's own predictions 300 times; the 5–95%
 * range of the resulting observed frequencies is the bar. Points outside their bar are evidence of miscalibration.
 */
export function ConsistencyBars() {
  const state = useFigureState({
    k: float(1.4, { min: 0.4, max: 2.5, step: 0.05, label: 'model sharpness k' }),
    n: int(500, { min: 100, max: 5000, step: 100, label: 'cases', format: (v) => String(v) }),
    bins: int(10, { min: 3, max: 20, step: 1, label: 'bins', format: (v) => String(v) }),
    binning: choice<Binning>(
      [
        { value: 'width', label: 'equal width' },
        { value: 'mass', label: 'equal mass' },
      ],
      'width',
      { label: 'binning' },
    ),
  })

  const r = useMemo(() => {
    const g = stream(4)
    const p: number[] = []
    const y: number[] = []
    for (let i = 0; i < state.n; i++) {
      const z = 1.5 * normal(g)
      y.push(uniform(g) < sigmoid(z) ? 1 : 0)
      p.push(sigmoid(state.k * z))
    }
    const m = state.bins
    // Bin membership: equal-width bins on [0, 1], or equal-mass bins by rank.
    const order = p.map((_, i) => i).sort((a, b) => p[a] - p[b])
    const bin = new Array<number>(p.length)
    order.forEach((i, rank) => {
      bin[i] = state.binning === 'width' ? Math.min(m - 1, Math.floor(p[i] * m)) : Math.floor((rank * m) / p.length)
    })
    const members: number[][] = Array.from({ length: m }, () => [])
    bin.forEach((b, i) => members[b].push(i))
    const h = stream(9)
    const out = members
      .filter((idx) => idx.length > 0)
      .map((idx) => {
        const meanP = idx.reduce((a, i) => a + p[i], 0) / idx.length
        const freq = idx.reduce((a, i) => a + y[i], 0) / idx.length
        const sims: number[] = []
        for (let rep = 0; rep < RESAMPLES; rep++) {
          let c = 0
          for (const i of idx) if (uniform(h) < p[i]) c++
          sims.push(c / idx.length)
        }
        sims.sort((a, b) => a - b)
        const lo = sims[Math.floor(0.05 * RESAMPLES)]
        const hi = sims[Math.floor(0.95 * RESAMPLES) - 1]
        return { meanP, freq, lo, hi, count: idx.length, outside: freq < lo || freq > hi }
      })
    return out
  }, [state.k, state.n, state.bins, state.binning])

  const bars: Segment[] = r.map((b) => ({ from: [b.meanP, b.lo], to: [b.meanP, b.hi] }))
  const outside = r.filter((b) => b.outside).length

  const xAxis = useAxis({ label: 'mean predicted probability in bin', range: [0, 1] })
  const yAxis = useAxis({ label: 'observed frequency', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="Reliability diagram with consistency bars"
      state={state}
      caption="Dots are observed frequencies of positives in each bin, placed at the bin's mean prediction. Each vertical bar is the 5–95% range of observed frequencies you would see if the model were exactly calibrated, from resampling the labels 300 times from the model's own predictions. At sharpness k = 1 the model is calibrated and about one bin in ten falls outside its bar by chance. Raise k to make it overconfident, or lower the number of cases to see how wide the bars become."

      readouts={
        <>
          <Readout label="bins outside their bar" value={`${outside} of ${r.length}`} />
          <Readout label="smallest bin" value={Math.min(...r.map((b) => b.count))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="perfect calibration" x={[0, 1]} y={[0, 1]} dashed muted />
        <Points
          name="inside bar"
          x={r.filter((b) => !b.outside).map((b) => b.meanP)}
          y={r.filter((b) => !b.outside).map((b) => b.freq)}
          slot={0}
        />
        <Points
          name="outside bar"
          x={r.filter((b) => b.outside).map((b) => b.meanP)}
          y={r.filter((b) => b.outside).map((b) => b.freq)}
          slot={1}
        />
        <Segments segments={bars} />
      </Plot>
    </Figure>
  )
}
