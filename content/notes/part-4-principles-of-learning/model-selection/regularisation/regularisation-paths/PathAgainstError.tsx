import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linearRegressionData } from 'aifn-methods/data/synthetic'
import { elasticNetPath, ridgePath } from 'aifn-methods/learning/linear'
import { child, stream } from 'aifn-compute/foundation/random'
import { matmul, toRows, transpose, type Tensor } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'

/** True coefficients: three large, three small, six null. */
const WEIGHTS = [3, -2, 1.5, 0.5, -0.5, 0.5, 0, 0, 0, 0, 0, 0]
const KIND = WEIGHTS.map((w) => (Math.abs(w) > 1 ? 'large' : w !== 0 ? 'small' : 'null'))
const NOISE = 2
const TEST = 4000
const LOG_MIN = -1
const LOG_MAX = 4
const STEP = 0.0625
/** One grid of λ for both penalties, decreasing, in the note's scaling ½‖y − Xw‖² + λΩ(w). */
const LOG_LAMBDAS = Array.from({ length: (LOG_MAX - LOG_MIN) / STEP + 1 }, (_, k) => LOG_MAX - k * STEP)
const LAMBDAS = LOG_LAMBDAS.map((l) => 10 ** l)

const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/** Mean squared error of the fit at each grid point: predictions Xw + b for every column of the path. */
function errors(x: Tensor, y: Tensor, coef: Tensor, intercept: readonly number[]) {
  const preds = toRows(transpose(matmul(x, coef))) as number[][]
  return preds.map((row, k) =>
    meanSquaredError(
      y,
      row.map((v) => v + intercept[k]),
    ),
  )
}

/**
 * Ridge and lasso paths on one simulated regression, against the training error and the true error of each fit. The
 * handle marks a penalty; the readouts give its degrees of freedom and errors.
 */
export function PathAgainstError() {
  const state = useFigureState({
    penalty: choice(['lasso', 'ridge'] as const, 'lasso', { label: 'penalty' }),
    logLambda: float(1.5, { min: LOG_MIN, max: LOG_MAX, step: STEP, label: 'log₁₀ λ', format: (v) => v.toFixed(2) }),
    n: int(50, { ge: 20, le: 400, suggestions: [20, 50, 100, 400], label: 'training rows n' }),
    correlation: slider(0, 0.9, 0.5, { step: 0.1, label: 'neighbour correlation ρ' }),
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3], label: 'data set (seed)' }),
  })
  const { penalty, n, correlation, seed } = state

  const computed = useComputed(
    () => {
      const options = { weights: WEIGHTS, noise: NOISE, correlation }
      const train = linearRegressionData(child(stream(`regularisation-paths/${seed}`), 'train'), { n, ...options })
      const test = linearRegressionData(child(stream('regularisation-paths'), 'test'), { n: TEST, ...options })
      if (penalty === 'ridge') {
        const path = ridgePath(train.x, train.y!, LAMBDAS)
        return {
          coef: toRows(path.coef) as number[][],
          dof: [...path.dof],
          train: errors(train.x, train.y!, path.coef, path.intercept),
          truth: errors(test.x, test.y!, path.coef, path.intercept),
          sweeps: null,
        }
      }
      const path = elasticNetPath(train.x, train.y!, { l1Ratio: 1, alphas: LAMBDAS.map((l) => l / n) })
      return {
        coef: toRows(path.coef) as number[][],
        dof: [...path.nonzero],
        train: errors(train.x, train.y!, path.coef, path.intercept),
        truth: errors(test.x, test.y!, path.coef, path.intercept),
        sweeps: path.steps.reduce((a, b) => a + b, 0),
      }
    },
    [penalty, n, correlation, seed],
    { mode: 'release' },
  )
  const r = computed.value
  const at = Math.max(0, Math.min(LOG_LAMBDAS.length - 1, Math.round((LOG_MAX - state.logLambda) / STEP)))
  const best = useMemo(() => r.truth.reduce((b, v, i) => (v < r.truth[b] ? i : b), 0), [r])

  const lx = useAxis({ label: 'log₁₀ λ (more regularised →)', range: [LOG_MIN, LOG_MAX] })
  const cy = useAxis({ label: 'coefficient' })
  const ey = useAxis({ label: 'mean squared error' })
  const slotOf = { large: 0, small: 1 } as const
  return (
    <Figure
      title="A regularisation path and the error along it"
      state={state}
      caption={`Ridge or lasso fits to n rows of 12 standard-normal predictors, neighbours correlated ρ (an AR(1) design), with true coefficients (3, −2, 1.5) (large), (0.5, −0.5, 0.5) (small) and six zeros (grey), and noise of standard deviation ${NOISE}. Top: the coefficient path against log₁₀ λ, in the scaling ½‖y − Xw‖² + λΩ(w). Bottom: the training error and the true error of each fit, measured on ${TEST} fresh rows. Drag the λ handle. The lasso zeroes the null predictors one by one as λ grows and keeps the large ones longest; ridge shrinks all twelve together and zeroes none. The degrees of freedom are Σ d²/(d² + λ) for ridge and the number of nonzero coefficients for the lasso.`}
      readouts={
        <>
          <Readout label="λ" value={fmt(LAMBDAS[at])} />
          <Readout
            label={penalty === 'ridge' ? 'degrees of freedom' : 'nonzero coefficients (df estimate)'}
            value={fmt(r.dof[at])}
          />
          <Readout label="training error at λ" value={fmt(r.train[at])} />
          <Readout label="true error at λ" value={fmt(r.truth[at])} />
          <Readout
            label="λ with the smallest true error"
            value={`${fmt(LAMBDAS[best])} (error ${fmt(r.truth[best])})`}
          />
          {r.sweeps !== null && <Readout label="coordinate sweeps for the whole path" value={String(r.sweeps)} />}
        </>
      }
    >
      <Plots rows={2} cols={1} heights={[1, 1]}>
        <Plot x={lx} y={cy}>
          {r.coef.map((row, j) =>
            KIND[j] === 'null' ? (
              <Curve key={j} id={`c${j}`} name="null predictors" x={LOG_LAMBDAS} y={row} muted stale={computed.stale} />
            ) : (
              <Curve
                key={j}
                id={`c${j}`}
                name={`${KIND[j]} true coefficients`}
                x={LOG_LAMBDAS}
                y={row}
                slot={slotOf[KIND[j] as 'large' | 'small']}
                stale={computed.stale}
              />
            ),
          )}
          <Handle {...state.handle('logLambda', { label: 'λ' })} />
        </Plot>
        <Plot x={lx} y={ey}>
          <Curve name="training error" x={LOG_LAMBDAS} y={r.train} slot={2} stale={computed.stale} />
          <Curve name="true error" x={LOG_LAMBDAS} y={r.truth} emphasis stale={computed.stale} />
          <Handle {...state.handle('logLambda', { label: 'λ' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
