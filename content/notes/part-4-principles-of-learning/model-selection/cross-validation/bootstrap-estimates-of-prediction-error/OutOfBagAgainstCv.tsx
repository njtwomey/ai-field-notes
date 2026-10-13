import { useMemo } from 'react'
import {
  Curve,
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
import { dataset } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import { randomForest } from 'aifn-methods/learning/trees-and-ensembles'

const FEATURES = 5
const TEST = 4000

/** Five standard-normal inputs; the label is 1 when x₁ + x₂ + x₁x₃ plus unit noise is positive. */
function draw(s: Stream, rows: number) {
  const x = toFlat(normals(child(s, 'x'), [rows, FEATURES]))
  const e = toFlat(normals(child(s, 'e'), [rows]))
  const y = Float64Array.from({ length: rows }, (_, i) => {
    const r = i * FEATURES
    return x[r] + x[r + 1] + x[r] * x[r + 2] + e[i] > 0 ? 1 : 0
  })
  return dataset(fromData(Float64Array.from(x), [rows, FEATURES]), fromData(y))
}

/** The out-of-bag error of a random forest and its 5-fold cross-validation error, each beside the true error. */
export function OutOfBagAgainstCv() {
  const state = useFigureState({
    n: int(100, { ge: 40, le: 400, suggestions: [50, 100, 200, 400], label: 'training rows n' }),
    trees: int(30, { ge: 5, le: 200, suggestions: [10, 30, 100], label: 'trees B' }),
    replicates: int(6, { ge: 2, le: 30, suggestions: [4, 6, 16], label: 'replicate data sets' }),
  })
  const { n, trees, replicates } = state
  const computed = useComputed(
    () => {
      const test = draw(child(stream('oob-cv'), 'test'), TEST)
      const forest = randomForest({ trees })
      const rows = Array.from({ length: replicates }, (_, r) => {
        const s = child(stream('oob-cv'), r)
        const data = draw(s, n)
        const model = forest.fit(data, { stream: child(s, 'fit') })
        const cv = crossValidate(forest, data, kFold({ k: 5, shuffle: true }), [accuracy], { stream: child(s, 'cv') })
        return {
          truth: 1 - accuracy(test.y, model.decide(test.x)),
          oob: 1 - model.outOfBag.accuracy,
          cv: 1 - cv.mean.accuracy,
        }
      })
      return rows
    },
    [n, trees, replicates],
    { mode: 'release' },
  )
  const rows = computed.value
  const view = useMemo(() => {
    const avg = (f: (r: (typeof rows)[number]) => number) => rows.reduce((a, r) => a + f(r), 0) / rows.length
    const all = rows.flatMap((r) => [r.truth, r.oob, r.cv])
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    return {
      diag: [lo, hi],
      truth: rows.map((r) => r.truth),
      oob: rows.map((r) => r.oob),
      cv: rows.map((r) => r.cv),
      mean: { truth: avg((r) => r.truth), oob: avg((r) => r.oob), cv: avg((r) => r.cv) },
    }
  }, [rows])
  const tx = useAxis({ label: 'true error (fresh test set)' })
  const ey = useAxis({ label: 'estimated error', equal: tx })
  return (
    <Figure
      title="Out-of-bag error against cross-validation"
      state={state}
      caption={`A random forest of B trees classifies points whose label depends on three of five inputs, with noise. Each replicate data set gives one point per estimator: its true error on ${TEST} fresh points (horizontal) against the out-of-bag error from the one fitted forest, and against 5-fold cross-validation, which fits five more forests (vertical). Points on the dashed diagonal are exact. Both estimators scatter about as widely around the truth and track it poorly from data set to data set. With few trees the out-of-bag error is pessimistic, because each point is predicted by only about 0.37B trees.`}
      readouts={{
        'mean over replicates': (
          <>
            <Readout label="true error" value={formatNumber(view.mean.truth)} />
            <Readout label="out-of-bag" value={formatNumber(view.mean.oob)} />
            <Readout label="5-fold CV" value={formatNumber(view.mean.cv)} />
          </>
        ),
        'forests fitted per estimate': (
          <>
            <Readout label="out-of-bag" value="1" />
            <Readout label="5-fold CV" value="5" />
          </>
        ),
      }}
    >
      <Plot x={tx} y={ey}>
        <Curve name="estimate = truth" x={view.diag} y={view.diag} muted dashed />
        <Points name="out-of-bag" x={view.truth} y={view.oob} slot={0} stale={computed.stale} />
        <Points name="5-fold CV" x={view.truth} y={view.cv} slot={1} stale={computed.stale} />
      </Plot>
    </Figure>
  )
}
