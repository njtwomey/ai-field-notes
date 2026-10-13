import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, shuffle, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { accuracy } from 'aifn-compute/learning/metrics'
import { bootstrapPredictionError } from 'aifn-compute/learning/selection'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'

const TEST = 4000
const FOLDS = 10
const ESTIMATES = ['training error', 'leave-one-out bootstrap', '.632', '.632+', `${FOLDS}-fold CV`] as const
/** Short names for the bar chart's category axis, from the bottom up: the truth, then the estimates in reverse. */
const ROWS = ['true error', `${FOLDS}-fold CV`, '.632+', '.632', 'LOO boot', 'training']

/**
 * `rows` points in two standard-normal inputs with exactly balanced labels in random order; the class means sit
 * `separation` apart along the first input. With separation 0 the labels carry no information about the inputs.
 */
function draw(s: Stream, rows: number, separation: number) {
  const labels = shuffle(
    child(s, 'labels'),
    Array.from({ length: rows }, (_, i) => (i < Math.floor(rows / 2) ? 1 : 0)),
  )
  const x = Float64Array.from(toFlat(normals(child(s, 'x'), [rows, 2])))
  for (let i = 0; i < rows; i++) x[2 * i] += labels[i] ? separation / 2 : -separation / 2
  return dataset(fromData(x, [rows, 2]), fromData(Float64Array.from(labels)))
}

/** The leave-one-out bootstrap, .632 and .632+ estimates for a k-nearest-neighbour classifier, beside the truth. */
export function BootstrapEstimators() {
  const state = useFigureState({
    neighbours: int(1, { ge: 1, le: 15, suggestions: [1, 3, 5, 9], label: 'neighbours k' }),
    separation: slider(0, 3, 0, { step: 0.1, label: 'class separation' }),
    n: int(50, { ge: 20, le: 200, suggestions: [30, 50, 100, 200], label: 'training rows n' }),
    B: int(200, { ge: 20, le: 1000, suggestions: [50, 100, 200, 500], label: 'bootstrap samples B' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3, 4], label: 'data set (seed)' }),
  })
  const { neighbours, separation, n, B, seed } = state
  const computed = useComputed(
    () => {
      const s = stream(`bootstrap-error/${seed}`)
      const data = draw(child(s, 'train'), n, separation)
      const test = draw(child(stream('bootstrap-error'), 'test'), TEST, separation)
      const learner = kNearestNeighbours({ k: neighbours })
      const e = bootstrapPredictionError(learner, data, { B, stream: child(s, 'boot'), loss: 'zeroOne' })
      const cv = crossValidate(learner, data, kFold({ k: FOLDS, shuffle: true }), [accuracy], {
        stream: child(s, 'cv'),
      })
      const truth = 1 - accuracy(test.y, learner.fit(data).decide(test.x))
      return {
        estimates: [e.apparent, e.looBootstrap, e.err632, e.err632plus, 1 - cv.mean.accuracy],
        noInformation: e.noInformation,
        relativeOverfitting: e.relativeOverfitting,
        weight: e.weight,
        counts: Array.from(toFlat(e.outOfBagCount)),
        truth,
      }
    },
    [neighbours, separation, n, B, seed],
    { mode: 'release' },
  )
  const r = computed.value
  const view = useMemo(() => {
    const rows = Array.from({ length: r.counts.length }, (_, i) => i)
    const meanCount = r.counts.reduce((a, c) => a + c, 0) / r.counts.length
    return { rows, meanCount }
  }, [r])
  const ex = useAxis({ label: 'error rate (0–1 loss)', range: [0, 0.7] })
  const ey = useAxis({ label: 'estimate', categories: ROWS })
  const cx = useAxis({ label: 'training row i', range: [-0.5, r.counts.length - 0.5], nice: false })
  const cy = useAxis({ label: 'samples leaving row i out, C_i', range: [0, undefined] })
  return (
    <Figure
      title="Bootstrap estimates of prediction error"
      state={state}
      caption={`A k-nearest-neighbour classifier on n points in two dimensions, half of each class, whose class means sit a set distance apart along the first input; at separation 0 the labels are pure noise. Left: the training error, the leave-one-out bootstrap (LOO boot), the .632 and .632+ estimates from B bootstrap samples, and ${FOLDS}-fold cross-validation, against the true error of the classifier fitted to all n points (bottom bar, measured on ${TEST} fresh points) and the no-information rate γ (dashed line). Right: how many of the B samples leave out each training point, C_i, with their mean, close to 0.368B. The defaults, one neighbour on pure noise, are the worked example: the training error is 0, the .632 estimate is far too low, and .632+ moves almost all its weight onto the leave-one-out bootstrap. Raise k or the separation to see the relative overfitting rate fall and the .632+ weight return towards 0.632.`}
      readouts={{
        estimates: (
          <>
            {ESTIMATES.map((label, i) => (
              <Readout key={label} label={label} value={formatNumber(r.estimates[i])} />
            ))}
            <Readout label="true error" value={formatNumber(r.truth)} />
          </>
        ),
        '.632+ ingredients': (
          <>
            <Readout label="no-information rate γ" value={formatNumber(r.noInformation)} />
            <Readout label="relative overfitting R" value={formatNumber(r.relativeOverfitting)} />
            <Readout label=".632+ weight w" value={formatNumber(r.weight)} />
          </>
        ),
      }}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={ex} y={ey} height={300}>
          <Bars name="training" x={[5]} y={[r.estimates[0]]} width={0.6} orient="y" muted />
          <Bars
            name="bootstrap"
            x={[4, 3, 2]}
            y={r.estimates.slice(1, 4)}
            width={0.6}
            orient="y"
            slot={0}
            stale={computed.stale}
          />
          <Bars
            name={`${FOLDS}-fold CV`}
            x={[1]}
            y={[r.estimates[4]]}
            width={0.6}
            orient="y"
            slot={1}
            stale={computed.stale}
          />
          <Bars name="truth" x={[0]} y={[r.truth]} width={0.6} orient="y" emphasis stale={computed.stale} />
          <Annotation x={r.noInformation} text="no information γ" dashed muted />
        </Plot>
        <Plot x={cx} y={cy} height={300}>
          <Bars name="C_i" x={view.rows} y={r.counts} slot={0} stale={computed.stale} />
          <Annotation y={view.meanCount} text={`mean ${formatNumber(view.meanCount)}`} dashed />
        </Plot>
      </div>
    </Figure>
  )
}
