import { useMemo } from 'react'
import { drawTrainingSet, targetFunction, unitGrid } from 'aifn-methods/theory'
import { linearRegression } from 'aifn-methods/learning/linear'
import { splineFeatures } from 'aifn-methods/learning/preprocessing'
import { pipeline } from 'aifn-compute/learning/compose'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold } from 'aifn-compute/learning/validate'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const KNOTS = 30
const FOLDS = 5
const TEST_SIZE = 4000
const LOG_MIN = -6
const LOG_MAX = 3
const LOG_LAMBDAS = Array.from({ length: 37 }, (_, i) => LOG_MIN + i * 0.25)
const GRID = unitGrid(201)
const TRUTH = Array.from(GRID, targetFunction('sine'))
const KEY = meanSquaredError.info.key
const LOW = 0.01
const HIGH = 10

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const clip = (v: number) => Math.max(LOW, Math.min(HIGH, v))
const model = (logLambda: number) =>
  pipeline(splineFeatures({ knots: KNOTS, degree: 3 }), linearRegression({ l2: 10 ** logLambda }))
const column = (x: Float64Array) => fromData(x, [x.length, 1])

/**
 * The validation curve of ridge regression on a cubic B-spline basis with 30 knots: 5-fold cross-validated training and
 * validation error against log₁₀ λ, beside the true test error of the model fitted to all n points. Right: the data and
 * the fit at the λ under the handle.
 */
export function RidgeValidationCurve() {
  const state = useFigureState({
    logLambda: float(-1, { min: LOG_MIN, max: LOG_MAX, step: 0.25, label: 'log₁₀ λ', format: (v) => v.toFixed(2) }),
    n: int(60, { min: 20, max: 400, suggestions: [30, 60, 200], label: 'training points n' }),
    noise: slider(0.05, 1, 0.3, { step: 0.05, label: 'noise σ' }),
    seed: int(0, { ge: 0, label: 'seed' }),
  })

  const data = useMemo(() => {
    const train = drawTrainingSet(stream(`validation-curve/train/${state.seed}`), 'sine', state.n, state.noise)
    const test = drawTrainingSet(stream(`validation-curve/test/${state.seed}`), 'sine', TEST_SIZE, state.noise)
    return { train, test, set: dataset(column(train.x), fromData(train.y)), testX: column(test.x) }
  }, [state.seed, state.n, state.noise])

  const curve = useComputed(
    () => {
      const rows = LOG_LAMBDAS.map((lg) => {
        const cv = crossValidate(model(lg), data.set, kFold({ k: FOLDS, shuffle: true }), [meanSquaredError], {
          stream: stream(`validation-curve/folds/${state.seed}`),
          trainMetrics: true,
        })
        const training = cv.folds.reduce((a, f) => a + (f.trainMetrics?.[KEY] ?? NaN), 0) / FOLDS
        const truth = meanSquaredError(data.test.y, toFlat(model(lg).fit(data.set).decide(data.testX)))
        return { training, validation: cv.mean[KEY], se: cv.std[KEY] / Math.sqrt(FOLDS), truth }
      })
      const argmin = (key: 'validation' | 'truth') =>
        rows.reduce((best, r, i) => (r[key] < rows[best][key] ? i : best), 0)
      return { rows, cvBest: argmin('validation'), trueBest: argmin('truth') }
    },
    [data, state.seed],
    { mode: 'release' },
  )
  const { rows, cvBest, trueBest } = curve.value
  const at = Math.round((state.logLambda - LOG_MIN) / 0.25)
  const here = rows[Math.max(0, Math.min(rows.length - 1, at))]

  const fit = useMemo(
    () =>
      Array.from(toFlat(model(state.logLambda).fit(data.set).decide(column(GRID))), (v) =>
        Math.max(-3, Math.min(3, v)),
      ),
    [data, state.logLambda],
  )

  const lambdaAxis = useAxis({ label: 'log₁₀ λ (more regularised →)', range: [LOG_MIN, LOG_MAX] })
  const errorAxis = useAxis({ label: 'mean squared error', range: [LOW, HIGH], log: true })
  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2.5, 2.5] })
  return (
    <Figure
      title="Validation curve of a ridge penalty"
      state={state}
      caption={`Ridge regression on a cubic B-spline basis with ${KNOTS} knots, fitted to n points with x uniform on [−1, 1] and y = sin(πx) + σε. Left: ${FOLDS}-fold cross-validated error on the training folds and on the held-out folds, with a band of one standard error (fold standard deviation over √${FOLDS}), against log₁₀ λ; the dashed line is the true test error of the model fitted to all n points, measured on ${TEST_SIZE} fresh points. Small λ (left) overfits: the training error is low and the validation error high. Large λ (right) underfits: both errors are high and close. Right: the data, the true curve (dashed) and the fit at the λ under the handle. Drag the handle on the left chart. Errors are clipped to [${LOW}, ${HIGH}].`}
      readouts={
        <>
          <Readout label="training error at λ" value={fmt(here.training)} />
          <Readout label="validation error at λ" value={fmt(here.validation)} />
          <Readout label="true error at λ" value={fmt(here.truth)} />
          <Readout label="log₁₀ λ chosen by CV" value={LOG_LAMBDAS[cvBest].toFixed(2)} />
          <Readout label="log₁₀ λ best on truth" value={LOG_LAMBDAS[trueBest].toFixed(2)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={lambdaAxis} y={errorAxis} title={`n = ${state.n}`}>
          <Area
            name="± 1 standard error"
            x={LOG_LAMBDAS}
            y={rows.map((r) => clip(r.validation + r.se))}
            base={rows.map((r) => clip(r.validation - r.se))}
            slot={0}
            line={false}
            stale={curve.stale}
          />
          <Curve
            name="training error"
            slot={1}
            x={LOG_LAMBDAS}
            y={rows.map((r) => clip(r.training))}
            stale={curve.stale}
          />
          <Curve
            name="validation error"
            slot={0}
            x={LOG_LAMBDAS}
            y={rows.map((r) => clip(r.validation))}
            showPoints
            stale={curve.stale}
          />
          <Curve
            name="true test error"
            emphasis
            dashed
            x={LOG_LAMBDAS}
            y={rows.map((r) => clip(r.truth))}
            stale={curve.stale}
          />
          <Handle {...state.handle('logLambda', { label: 'λ' })} />
        </Plot>
        <Plot x={xAxis} y={yAxis} title={`log₁₀ λ = ${state.logLambda.toFixed(2)}`}>
          <Points name="training points" x={data.train.x} y={data.train.y} muted />
          <Curve name="true curve" x={GRID} y={TRUTH} emphasis dashed />
          <Curve name="ridge fit" x={GRID} y={fit} slot={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}
