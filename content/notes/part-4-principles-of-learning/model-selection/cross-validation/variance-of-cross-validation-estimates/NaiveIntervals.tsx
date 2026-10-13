import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { type Tensor } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, type Fittable } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { linearRegression, type LinearRegressionModel } from 'aifn-methods/learning/linear'
import { kNearestNeighboursRegression, type NeighboursRegressor } from 'aifn-methods/learning/neighbours'
import { draw, LEARNERS, ROOT, TEST } from './problem'

/** The two-sided 90% normal quantile. */
const Z90 = 1.645

/** Naive 90% intervals from the fold errors of K-fold cross-validation, against the errors they should cover. */
export function NaiveIntervals() {
  const state = useFigureState({
    learner: choice(LEARNERS, 'least squares', { label: 'learner' }),
    n: int(100, { ge: 30, le: 400, suggestions: [50, 100, 200], label: 'training rows n' }),
    p: int(20, { ge: 5, le: 60, suggestions: [5, 20, 40], label: 'inputs p' }),
    folds: int(10, { ge: 2, le: 20, suggestions: [2, 5, 10, 20], label: 'folds K' }),
    replicates: int(100, { ge: 20, le: 400, suggestions: [50, 100, 200], label: 'replicate data sets' }),
  })
  const { learner, n, p, folds, replicates } = state
  const computed = useComputed(
    () => {
      const root = stream(ROOT)
      const test = draw(child(root, 'test'), TEST, p)
      const est = (
        learner === 'least squares' ? linearRegression({ intercept: false }) : kNearestNeighboursRegression({ k: 10 })
      ) as Fittable<Tensor, LinearRegressionModel | NeighboursRegressor>
      return Array.from({ length: replicates }, (_, r) => {
        const s = child(root, r)
        const data = draw(s, n, p)
        const cv = crossValidate(est, data, kFold({ k: folds, shuffle: true }), [meanSquaredError], {
          stream: child(s, 'cv'),
        })
        const model = est.fit(data)
        return {
          cv: cv.mean.meanSquaredError,
          se: cv.std.meanSquaredError / Math.sqrt(folds),
          err: meanSquaredError(test.y, model.decide(test.x)),
        }
      })
    },
    [learner, n, p, folds, replicates],
    { mode: 'release' },
  )
  const rows = computed.value
  const view = useMemo(() => {
    const sorted = [...rows].sort((a, b) => a.cv - b.cv)
    const average = rows.reduce((a, r) => a + r.err, 0) / rows.length
    const covers = (r: (typeof rows)[number], target: number) => Math.abs(r.cv - target) <= Z90 * r.se
    const bars = sorted.map((r, i) => ({ from: [i, r.cv - Z90 * r.se] as const, to: [i, r.cv + Z90 * r.se] as const }))
    return {
      average,
      hit: bars.filter((_, i) => covers(sorted[i], sorted[i].err)),
      miss: bars.filter((_, i) => !covers(sorted[i], sorted[i].err)),
      index: sorted.map((_, i) => i),
      err: sorted.map((r) => r.err),
      cv: sorted.map((r) => r.cv),
      coverConditional: rows.filter((r) => covers(r, r.err)).length / rows.length,
      coverExpected: rows.filter((r) => covers(r, average)).length / rows.length,
      spread: standardDeviation(rows.map((r) => r.cv)),
      meanSe: rows.reduce((a, r) => a + r.se, 0) / rows.length,
    }
  }, [rows])
  const ix = useAxis({ label: 'replicate data set, sorted by CV estimate', integer: true })
  const ey = useAxis({ label: 'mean squared error' })
  return (
    <Figure
      title="Naive cross-validation intervals are too narrow"
      state={state}
      caption={`Each column is one training set of n rows (y is the sum of five of p standard-normal inputs plus unit noise). The bar is the naive 90% interval CV ± 1.645·s/√K, with s the standard deviation of the K fold errors; the dot is the conditional error Err_D of the model fitted to all n rows, measured on ${TEST} fresh rows. Bars that miss their dot take the second colour of the legend. The dashed line is the average of Err_D over the replicates, an estimate of Err. A valid 90% interval misses about one replicate in ten. The naive bars miss more often, because the fold errors are positively correlated and s/√K understates the spread of CV across data sets.`}
      readouts={{
        coverage: (
          <>
            <Readout label="covers Err_D" value={formatNumber(view.coverConditional)} />
            <Readout label="covers Err" value={formatNumber(view.coverExpected)} />
            <Readout label="nominal" value="0.9" />
          </>
        ),
        'standard error of CV': (
          <>
            <Readout label="naive s/√K (mean)" value={formatNumber(view.meanSe)} />
            <Readout label="actual spread over data sets" value={formatNumber(view.spread)} />
          </>
        ),
      }}
    >
      <Plot x={ix} y={ey}>
        {view.hit.length > 0 && <Segments name="interval covers Err_D" segments={view.hit} slot={0} width={2} />}
        {view.miss.length > 0 && <Segments name="interval misses Err_D" segments={view.miss} slot={1} width={2} />}
        <Points name="Err_D" x={view.index} y={view.err} emphasis size={4} stale={computed.stale} />
        <Annotation y={view.average} text="Err" dashed emphasis />
      </Plot>
    </Figure>
  )
}
