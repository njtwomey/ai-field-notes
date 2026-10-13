import { useMemo } from 'react'
import {
  Annotation,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset, takeData } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { shuffleSplit } from 'aifn-compute/learning/validate'
import { Binomial } from 'aifn-compute/probability/distributions'
import { quantile, standardDeviation } from 'aifn-compute/probability/stats'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const P = 5
const TEST_FRACTION = 1 / 3
const FRESH = 3000
const TRUTH_SETS = 20
const ROWS = ['seed', 'split', 'both'] as const
/** Vertical offsets within a row, so that runs with equal accuracy (multiples of 1/test size) do not hide each other. */
const JITTER = [-0.24, -0.12, 0, 0.12, 0.24]

/** n rows: five standard-normal inputs, class 1 when the radius in the first two plus noise exceeds its median. */
function draw(s: Stream, n: number) {
  const x = toFlat(normals(child(s, 'x'), [n, P]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const y = Float64Array.from({ length: n }, (_, i) =>
    Math.hypot(x[P * i], x[P * i + 1]) + 0.3 * e[i] > 1.1774 ? 1 : 0,
  )
  return dataset(fromData(Float64Array.from(x), [n, P]), fromData(y))
}

/**
 * The test accuracy of a random forest on one benchmark data set, re-run with a new seed, a new train–test split, or
 * both, beside the algorithm's expected accuracy on the population.
 */
export function SourcesOfVariance() {
  const state = useFigureState({
    n: int(450, { ge: 90, le: 1500, suggestions: [150, 450, 1200], label: 'rows in the benchmark n' }),
    trees: int(15, { ge: 1, le: 100, suggestions: [1, 15, 50], label: 'trees' }),
    runs: int(30, { ge: 5, le: 100, suggestions: [10, 30, 60], label: 'runs per source' }),
  })
  const { n, trees, runs } = state
  const computed = useComputed(
    () => {
      const root = stream('comparing/sources-of-variance')
      const data = draw(child(root, 'benchmark'), n)
      const forest = randomForest({ trees })
      const split = shuffleSplit({ splits: 1, testSize: TEST_FRACTION })
      const run = (splitKey: string | number, seedKey: string | number) => {
        const [sp] = split.split({ n }, child(root, 'split', splitKey))
        const model = forest.fit(takeData(data, toFlat(sp.train)), { stream: child(root, 'seed', seedKey) })
        const test = takeData(data, toFlat(sp.test))
        return accuracy(test.y, model.decide(test.x))
      }
      const series = [
        Array.from({ length: runs }, (_, r) => run('fixed', r)),
        Array.from({ length: runs }, (_, r) => run(r, 'fixed')),
        Array.from({ length: runs }, (_, r) => run(`both-${r}`, `both-${r}`)),
      ]
      // The estimand: the forest's expected accuracy when trained on as many rows as a split gives it.
      const fresh = draw(child(root, 'fresh'), FRESH)
      const trainSize = n - Math.ceil(n * TEST_FRACTION)
      let truth = 0
      for (let r = 0; r < TRUTH_SETS; r++) {
        const model = forest.fit(draw(child(root, 'population', r), trainSize), { stream: child(root, 'truth', r) })
        truth += accuracy(fresh.y, model.decide(fresh.x)) / TRUTH_SETS
      }
      return { series, truth, testSize: Math.ceil(n * TEST_FRACTION) }
    },
    [n, trees, runs],
    { mode: 'release' },
  )
  const { series, truth, testSize } = computed.value
  const view = useMemo(() => {
    const sds = series.map((s) => standardDeviation(s, { sample: true }))
    const both = series[2]
    const gaps: number[] = []
    for (let i = 0; i < both.length; i++)
      for (let j = i + 1; j < both.length; j++) gaps.push(Math.abs(both[i] - both[j]))
    const mean = both.reduce((a, v) => a + v, 0) / both.length
    const binomial = (Binomial(testSize, mean).stddev() as number) / testSize
    return { sds, gap95: quantile(gaps, 0.95), binomial }
  }, [series, testSize])
  const x = useAxis({ label: 'test accuracy' })
  const y = useAxis({ categories: [...ROWS], inverse: true })
  return (
    <Figure
      title="Seeds, splits and the spread of a benchmark score"
      state={state}
      caption={`One benchmark data set of n rows (five standard-normal inputs; the class depends on the radius in the first two). A random forest is trained on two thirds and tested on the remaining third. Each dot is one run, spread vertically within its row so that equal accuracies stay visible. The row seed re-runs with new random seeds for the forest on a fixed split; split keeps the seed and draws a new split; both changes both, as an honest re-run would. The vertical line is the forest's expected accuracy on the population, estimated from ${TRUTH_SETS} fresh training sets scored on ${FRESH} fresh rows: the quantity every run is trying to measure.`}
      readouts={{
        'standard deviation over runs': (
          <>
            <Readout label="seed only" value={fmt(view.sds[0])} />
            <Readout label="split only" value={fmt(view.sds[1])} />
            <Readout label="seed and split" value={fmt(view.sds[2])} />
            <Readout label={`binomial, ${testSize} test rows`} value={fmt(view.binomial)} />
          </>
        ),
        'two runs of the same algorithm': <Readout label="95th percentile of their gap" value={fmt(view.gap95)} />,
      }}
    >
      <Plot x={x} y={y}>
        {series.map((s, row) => (
          <Points
            key={ROWS[row]}
            name={ROWS[row]}
            x={s}
            y={s.map((_, r) => row + JITTER[r % JITTER.length])}
            slot={row}
            size={5}
            stale={computed.stale}
          />
        ))}
        <Annotation x={truth} text="truth" emphasis />
      </Plot>
    </Figure>
  )
}
