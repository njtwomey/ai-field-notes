import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  setting,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, type Fittable } from 'aifn-compute/learning/validate'
import { correlation } from 'aifn-compute/probability/stats'
import { linearRegression, type LinearRegressionModel } from 'aifn-methods/learning/linear'
import { kNearestNeighboursRegression, type NeighboursRegressor } from 'aifn-methods/learning/neighbours'

const TEST = 5000
const LEARNERS = ['least squares', '10-nearest neighbours'] as const
type Learner = (typeof LEARNERS)[number]

/** y = x₁ + … + x₅ + ε with standard-normal inputs and noise; the other p − 5 inputs are irrelevant. */
function targets(x: Float64Array, e: Float64Array, rows: number, p: number) {
  return Float64Array.from({ length: rows }, (_, i) => {
    let v = e[i]
    for (let j = 0; j < Math.min(5, p); j++) v += x[i * p + j]
    return v
  })
}

function inputs(s: Stream, rows: number, p: number) {
  return Float64Array.from(toFlat(normals(child(s, 'x'), [rows, p])))
}

/** The 10-fold cross-validation estimate against the conditional error of the fitted model, over replicate data sets. */
export function CvAgainstConditionalError() {
  const state = useFigureState({
    learner: choice(LEARNERS, 'least squares', { label: 'learner' }),
    n: int(100, { ge: 30, le: 400, suggestions: [50, 100, 200], label: 'training rows n' }),
    p: int(20, { ge: 5, le: 60, suggestions: [5, 20, 40], label: 'inputs p' }),
    replicates: int(100, { ge: 20, le: 400, suggestions: [50, 100, 200], label: 'replicate data sets' }),
    fixedX: setting(false, { label: 'same inputs in every replicate' }),
  })
  const { learner, n, p, replicates, fixedX } = state
  const computed = useComputed(
    () => {
      const root = stream('what-cv-estimates')
      const testX = inputs(child(root, 'test'), TEST, p)
      const testY = targets(testX, Float64Array.from(toFlat(normals(child(root, 'test-noise'), [TEST]))), TEST, p)
      const test = { x: fromData(testX, [TEST, p]), y: fromData(testY) }
      const designX = inputs(child(root, 'design'), n, p)
      const make = (l: Learner) =>
        l === 'least squares' ? linearRegression({ intercept: false }) : kNearestNeighboursRegression({ k: 10 })
      const est = make(learner as Learner) as Fittable<Tensor, LinearRegressionModel | NeighboursRegressor>
      const out = { cv: [] as number[], err: [] as number[] }
      for (let r = 0; r < replicates; r++) {
        const s = child(root, r)
        const x = fixedX ? designX : inputs(s, n, p)
        const e = Float64Array.from(toFlat(normals(child(s, 'noise'), [n])))
        const data = dataset(fromData(x, [n, p]), fromData(targets(x, e, n, p)))
        const cv = crossValidate(est, data, kFold({ k: 10, shuffle: true }), [meanSquaredError], {
          stream: child(s, 'cv'),
        })
        const model = est.fit(data)
        out.cv.push(cv.mean.meanSquaredError)
        out.err.push(meanSquaredError(test.y, model.decide(test.x as Tensor)))
      }
      return out
    },
    [learner, n, p, replicates, fixedX],
    { mode: 'release' },
  )
  const { cv, err } = computed.value
  const view = useMemo(() => {
    const all = [...cv, ...err]
    const lo = Math.min(...all)
    const hi = Math.max(...all)
    const average = err.reduce((a, b) => a + b, 0) / err.length
    return { diag: [lo, hi], average, r: correlation(err, cv) }
  }, [cv, err])
  const ex = useAxis({ label: 'conditional error Err_D (fresh test set)' })
  const cy = useAxis({ label: '10-fold CV estimate', equal: ex })
  return (
    <Figure
      title="Cross-validation against the error of the model at hand"
      state={state}
      caption={`Each point is one training set of n rows from the same distribution: y is the sum of the first five of p standard-normal inputs plus unit noise. Horizontal: the conditional error Err_D of the model fitted to that training set, measured on ${TEST} fresh rows. Vertical: its 10-fold cross-validation estimate. The vertical line is the average of Err_D over the replicates, an estimate of the expected error Err. Points on the dashed diagonal would be exact. The cloud is taller than it is wide and does not lie along the diagonal: the estimate varies more than the error it is meant to track, and its correlation with that error is near zero. With "same inputs in every replicate" only the noise is redrawn; for least squares the correlation is then exactly zero in expectation.`}
      readouts={
        <>
          <Readout label="correlation of CV with Err_D" value={formatNumber(view.r)} />
          <Readout label="average Err_D" value={formatNumber(view.average)} />
          <Readout label="average CV" value={formatNumber(cv.reduce((a, b) => a + b, 0) / cv.length)} />
        </>
      }
    >
      <Plot x={ex} y={cy}>
        <Curve name="estimate = Err_D" x={view.diag} y={view.diag} muted dashed />
        <Annotation x={view.average} text="Err" emphasis dashed />
        <Points name="replicate data sets" x={err} y={cv} slot={0} stale={computed.stale} />
      </Plot>
    </Figure>
  )
}
