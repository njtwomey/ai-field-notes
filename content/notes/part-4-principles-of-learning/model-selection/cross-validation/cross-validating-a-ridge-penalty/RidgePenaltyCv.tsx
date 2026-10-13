import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError, ridgeCv } from 'aifn-compute/learning/selection'
import { gridSearch, kFold } from 'aifn-compute/learning/validate'
import { linearRegression } from 'aifn-methods/learning/linear'

const D = 20
const NOISE_SD = 2
const TEST = 20000
/** True coefficients 2 · 0.75^(j−1): every input matters, most only a little. */
const BETA = Array.from({ length: D }, (_, j) => 2 * 0.75 ** j)
/** 25 penalties, log-spaced from 0.1 to 1000. */
const LAMBDAS = Array.from({ length: 25 }, (_, i) => Number((10 ** (-1 + (i * 4) / 24)).toPrecision(3)))
const PENALTIES = space({ l2: oneOf(LAMBDAS) })

/** rows of x ~ N(0, I_20) and y = xᵀβ + ε with ε ~ N(0, 2²). */
function draw(s: Stream, rows: number) {
  const x = Float64Array.from(toFlat(normals(child(s, 'x'), [rows, D])))
  const e = toFlat(normals(child(s, 'e'), [rows]))
  const y = Float64Array.from({ length: rows }, (_, i) => {
    let v = NOISE_SD * e[i]
    for (let j = 0; j < D; j++) v += x[i * D + j] * BETA[j]
    return v
  })
  return { x, y, data: dataset(fromData(x, [rows, D]), fromData(y)) }
}

/** 10-fold CV with standard errors, leave-one-out and GCV for a ridge penalty, beside the true error. */
export function RidgePenaltyCv() {
  const state = useFigureState({
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3, 4], label: 'data set (seed)' }),
    n: int(50, { ge: 30, le: 200, suggestions: [30, 50, 100, 200], label: 'training rows n' }),
    folds: int(10, { ge: 2, le: 20, suggestions: [5, 10, 20], label: 'folds K' }),
  })
  const { seed, n, folds } = state
  const computed = useComputed(
    () => {
      const s = stream(`ridge-penalty/${seed}`)
      const train = draw(child(s, 'train'), n)
      const test = draw(child(stream('ridge-penalty'), 'test'), TEST).data
      const search = gridSearch((p) => linearRegression({ l2: p.l2 }), PENALTIES, {
        metrics: [meanSquaredError],
        refit: false,
      })
      const kf = search.run(train.data, kFold({ k: folds, shuffle: true }), child(s, 'folds'))
      const rule = oneStandardError(kf, { simplicity: LAMBDAS.map((l) => -l) })
      const smoother = ridgeCv(fromData(train.x, [n, D]), fromData(train.y), LAMBDAS)
      const truth = LAMBDAS.map((l2) => {
        const model = linearRegression({ l2 }).fit(train.data)
        return meanSquaredError(test.y, model.decide(test.x))
      })
      return {
        cv: Array.from(toFlat(rule.mean)),
        se: Array.from(toFlat(rule.se)),
        cvBest: LAMBDAS[rule.best],
        oneSe: LAMBDAS[rule.chosen],
        threshold: rule.threshold,
        loo: Array.from(toFlat(smoother.loo)),
        looBest: LAMBDAS[smoother.bestLoo],
        gcv: Array.from(toFlat(smoother.gcv)),
        gcvBest: LAMBDAS[smoother.bestGcv],
        truth,
      }
    },
    [seed, n, folds],
    { mode: 'release' },
  )
  const r = computed.value
  const view = useMemo(() => {
    const at = (l: number) => LAMBDAS.indexOf(l)
    const truthBest = LAMBDAS[r.truth.indexOf(Math.min(...r.truth))]
    const bars = LAMBDAS.map((l, i) => ({ from: [l, r.cv[i] - r.se[i]] as const, to: [l, r.cv[i] + r.se[i]] as const }))
    return { at, truthBest, bars }
  }, [r])
  const lx = useAxis({ label: 'ridge penalty λ', log: true })
  const ey = useAxis({ label: 'mean squared error' })
  const chosen = (label: string, l: number) => (
    <Readout label={label} value={`λ = ${formatNumber(l)}, true error ${formatNumber(r.truth[view.at(l)])}`} />
  )
  return (
    <Figure
      title="Three cross-validation curves for one ridge penalty"
      state={state}
      caption={`Ridge regression on n rows of 20 standard-normal inputs, with true coefficients 2·0.75^(j−1) and noise of standard deviation 2. Curves over 25 penalties: K-fold cross-validation with bars of ±1 standard error (the standard deviation of the K fold errors over √K), leave-one-out cross-validation and generalised cross-validation (both in closed form from the ridge hat matrix; refitting n times gives the same leave-one-out numbers), and the true error of the ridge fit to all n rows, measured on ${TEST} fresh rows. The estimates differ most at small λ, where the fit has most effective parameters, and their minima fall close together. The dashed horizontal line is the one-standard-error threshold, the K-fold minimum plus its standard error; the one-standard-error rule picks the largest λ whose K-fold error lies under it. Change the seed to see how far each chosen λ, and the true best λ, move between data sets.`}
      readouts={{
        'chosen penalty': (
          <>
            {chosen('K-fold minimum', r.cvBest)}
            {chosen('leave-one-out minimum', r.looBest)}
            {chosen('GCV minimum', r.gcvBest)}
            {chosen('one-standard-error rule', r.oneSe)}
            {chosen('best possible', view.truthBest)}
          </>
        ),
      }}
    >
      <Plot x={lx} y={ey}>
        <Segments name="±1 SE" segments={view.bars} slot={0} />
        <Curve name={`${folds}-fold CV`} x={LAMBDAS} y={r.cv} slot={0} showPoints stale={computed.stale} />
        <Curve name="leave-one-out" x={LAMBDAS} y={r.loo} slot={1} stale={computed.stale} />
        <Curve name="GCV" x={LAMBDAS} y={r.gcv} slot={2} dashed stale={computed.stale} />
        <Curve name="true error" x={LAMBDAS} y={r.truth} emphasis stale={computed.stale} />
        <Annotation y={r.threshold} text="CV min + 1 SE" dashed muted />
        <Annotation x={r.cvBest} text="CV min" dashed slot={0} />
        <Annotation x={r.oneSe} text="1-SE choice" dashed muted />
      </Plot>
    </Figure>
  )
}
