import { useMemo } from 'react'
import {
  choice,
  Bars,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  MathText,
  Plot,
  Readout,
  seriesLayers,
  useAxis,
  useFigureState,
  type SeriesSpec,
} from 'aifn-render'
import { useClassColors } from '../_shared/classColor'
import { exceedanceProbs, fitShared1d, logistic1d, nonParallelSample } from '../_shared/ordinal'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { sigmoid } from 'aifn-compute/numerics/special'

type Heads = 'independent' | 'shared'
const HEADS = [
  { value: 'independent' as const, label: 'independent (Frank & Hall)' },
  { value: 'shared' as const, label: 'shared slope (CORAL)' },
]
const X = toFlat(linspace(-3, 3, 241))
const K = 4

/**
 * Three binary classifiers for y > 1, y > 2 and y > 3 on one feature. Fitted independently, their curves can cross, and
 * the differenced class probabilities go negative there. With one shared slope and separate intercepts (CORAL), the
 * curves are ordered everywhere.
 */
export function RankConsistency() {
  const state = useFigureState({
    heads: choice<Heads>(HEADS, 'independent', { label: 'binary heads' }),
    n: int(60, { min: 30, max: 600, step: 10, label: 'sample size n' }),
    delta: float(0.8, { min: 0, max: 1.2, step: 0.05, label: 'slope spread δ' }),
    at: float(2, { min: -3, max: 3, step: 0.02, label: 'query x' }),
  })
  const colors = useClassColors(K)

  const fits = useMemo(() => {
    const { x, y } = nonParallelSample(state.n, state.delta, 23)
    const separate = [0, 1, 2].map((j) =>
      logistic1d(
        x,
        y.map((v) => (v > j ? 1 : 0)),
      ),
    )
    const shared = fitShared1d(x, y, K, 'all-threshold')
    const exceed = {
      independent: (v: number) => separate.map((f) => sigmoid(f.a + f.b * v)),
      shared: (v: number) => shared.theta.map((t) => sigmoid(shared.w * v - t)),
    }
    const crossed = X.filter((v) => exceedanceProbs(exceed.independent(v)).some((p) => p < 0)).length / X.length
    return { exceed, crossed }
  }, [state.n, state.delta])

  const series = useMemo<SeriesSpec[]>(() => {
    const out: SeriesSpec[] = []
    for (const kind of ['independent', 'shared'] as const) {
      const rows = X.map(fits.exceed[kind])
      ;[0, 1, 2].forEach((j) =>
        out.push({
          name: `P(y > ${j + 1}), ${kind}`,
          type: 'line',
          x: X,
          y: rows.map((r) => r[j]),
          slot: j,
          dashed: kind !== state.heads,
          muted: kind !== state.heads,
        }),
      )
    }
    return out
  }, [fits, state.heads])

  const probs = exceedanceProbs(fits.exceed[state.heads](state.at))
  const lowest = Math.min(0, ...probs)

  const xAxis = useAxis({ label: 'feature x', range: [-3, 3] })
  const yAxis = useAxis({ label: 'P(y > k | x)', range: [0, 1] })
  const kAxis = useAxis({ label: 'class k', range: [0.5, 4.5], integer: true })
  const pAxis = useAxis({ label: 'probability', range: [lowest < 0 ? Math.floor(lowest * 10) / 10 : 0, 1] })
  return (
    <Figure
      title="Rank consistency of binary decompositions"
      state={state}
      caption={
        <MathText text="Each curve is one binary classifier's estimate of $P(y > k \mid x)$, and class probabilities are the differences $P(y > k - 1) - P(y > k)$. The data's true splits have slopes $1.2\,(1 + \delta(k - 2))$. Independent fits to a small sample can cross, and where they cross a class probability is negative. A shared slope with ordered intercepts keeps the curves nested. Drag the cursor to read the class probabilities of the selected heads." />
      }
      readouts={
        <>
          {probs.map((p, k) => (
            <Readout key={k} label={`P(y = ${k + 1})`} value={formatNumber(p)} />
          ))}
          <Readout label="share of x with a negative class (independent)" value={formatNumber(fits.crossed)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[3fr_2fr]">
        <Plot x={xAxis} y={yAxis} height={300} ariaLabel={'Binary classifier outputs against the feature'}>
          {seriesLayers(series)}
          <Handle {...state.handle('at', { label: 'x' })} />
        </Plot>
        <Plot x={kAxis} y={pAxis} height={300} ariaLabel="Class probabilities at the cursor">
          <Bars name="P(y = k | x)" x={[1, 2, 3, 4]} y={probs} colors={colors} />
        </Plot>
      </div>
    </Figure>
  )
}
