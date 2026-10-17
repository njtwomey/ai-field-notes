import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/figure-helpers'
import { drawBlockModel, homophilySweep, labelledSplit, viewAccuracies } from '../_shared/linear-node-models'

const PANEL = 280
const BLOCK = 100
const DEGREE = 6
const SIGNAL = 1
const GRID = Array.from({ length: 11 }, (_, i) => i / 10)

const mean = (xs: readonly number[]) => xs.reduce((s, x) => s + x, 0) / xs.length
const sd = (xs: readonly number[]) => {
  const m = mean(xs)
  return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, xs.length - 1))
}
const pct = (m: number, s: number) => `${(100 * m).toFixed(1)}% ± ${(100 * s).toFixed(1)}`

/** How much the accuracy of two node classifiers, and their difference, moves between random labelled splits. */
export function SplitVariance() {
  const state = useFigureState({
    homophily: slider(0, 1, 0.6, { step: 0.01, label: 'edge homophily h' }),
    perClass: int(10, { ge: 2, le: 80, suggestions: [5, 10, 20, 50], label: 'labelled nodes per class' }),
    splits: int(40, { ge: 5, le: 200, suggestions: [10, 40, 100], label: 'random splits' }),
    graphSeed: int(1, { ge: 1, le: 999, label: 'graph seed' }),
  })
  const { homophily, perClass, splits, graphSeed } = state

  const runs = useComputed(
    () => {
      const s = stream(`benchmarks/${graphSeed}`)
      const d = drawBlockModel(child(s, 'graph'), { blockSize: BLOCK, degree: DEGREE, signal: SIGNAL, homophily })
      const y = toFlat(d.y)
      const features: number[] = []
      const gcn: number[] = []
      for (let k = 0; k < splits; k++) {
        const acc = viewAccuracies(d, labelledSplit(child(s, 'split', k), y, perClass))
        features.push(acc.features)
        gcn.push(acc.gcn)
      }
      return { features, gcn }
    },
    [homophily, perClass, splits, graphSeed],
    { mode: 'release' },
  )

  const sweep = useComputed(
    () =>
      homophilySweep({
        homophilies: GRID,
        setting: { blockSize: BLOCK, degree: DEGREE, signal: SIGNAL },
        perClass,
        seeds: 3,
      }),
    [perClass],
    { mode: 'release' },
  )

  const stats = useMemo(() => {
    const { features, gcn } = runs.value
    const gain = gcn.map((g, k) => g - features[k])
    return {
      features: { mean: mean(features), sd: sd(features) },
      gcn: { mean: mean(gcn), sd: sd(gcn) },
      gain: { mean: mean(gain), sd: sd(gain), min: Math.min(...gain), max: Math.max(...gain) },
      wins: gain.filter((g) => g > 0).length,
    }
  }, [runs.value])

  const fx = useAxis({ label: 'accuracy, features only', range: [0.4, 1], nice: false })
  const gy = useAxis({ label: 'accuracy, GCN mean', range: [0.4, 1], nice: false, equal: fx })
  const hx = useAxis({ label: 'edge homophily h', range: [0, 1], nice: false })
  const dy = useAxis({ label: 'GCN minus features-only accuracy', range: [-0.35, 0.3], nice: false })
  const gains = GRID.map((_, i) => sweep.value.gcn[i] - sweep.value.features[i])

  return (
    <Figure
      title="One split is not a benchmark"
      state={state}
      defaultSize="L"
      caption={`One two-class contextual stochastic block model with ${BLOCK} nodes per class, expected degree ${DEGREE} and edge homophily h, and many random choices of which nodes carry labels. For each split, two logistic regressions are fitted on the labelled nodes and tested on the rest: one on the node features alone, one on the mean over each node and its neighbours (a linear GCN layer). Left: one point per split, GCN accuracy against features-only accuracy; points above the diagonal are splits where the graph helped, and the ringed point is split 1, the one a paper with a single fixed split would report. Right: the mean gain of the GCN over the features-only model against homophily, averaged over three graphs; the gain is positive only at high homophily (and, by a sign flip, near h = 0). Drag the vertical line or use the slider to set h; fewer labelled nodes widen the spread.`}
      readouts={
        <>
          <Readout label="features only, mean ± sd" value={pct(stats.features.mean, stats.features.sd)} />
          <Readout label="GCN mean, mean ± sd" value={pct(stats.gcn.mean, stats.gcn.sd)} />
          <Readout
            label="gain per split, from … to"
            value={`${(100 * stats.gain.min).toFixed(1)} … ${(100 * stats.gain.max).toFixed(1)} points`}
          />
          <Readout label="splits where the GCN wins" value={`${stats.wins} of ${runs.value.gcn.length}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={fx} y={gy} title={`${runs.value.gcn.length} random splits at h = ${formatNumber(homophily)}`}>
            <Curve name="equal accuracy" x={[0.4, 1]} y={[0.4, 1]} dashed muted />
            <Points name="split" x={runs.value.features} y={runs.value.gcn} slot={0} stale={runs.stale} />
            {runs.value.gcn.length > 0 && (
              <Points
                name="split 1 (a fixed split)"
                x={[runs.value.features[0]]}
                y={[runs.value.gcn[0]]}
                emphasis
                size={16}
              />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={hx} y={dy} title="gain of the GCN over features only">
            <Curve name="no gain" x={[0, 1]} y={[0, 0]} dashed muted />
            <Curve name="mean gain" x={GRID} y={gains} slot={1} showPoints stale={sweep.stale} />
            <Handle {...state.handle('homophily', { label: 'h' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
