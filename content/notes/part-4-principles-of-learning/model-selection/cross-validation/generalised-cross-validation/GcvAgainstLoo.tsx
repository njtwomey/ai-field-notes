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
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, leaveOneOut } from 'aifn-compute/learning/validate'
import { psplineGcvPath } from 'aifn-compute/numerics/interpolate'
import { regression1d } from 'aifn-methods/data/synthetic'
import { psplineLearner, testError, truth } from '../_shared/problems'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const LOG_LAMBDAS = Array.from({ length: 29 }, (_, i) => -5 + i * 0.25)
const SPACINGS = [
  { value: 'even', label: 'evenly spaced' },
  { value: 'random', label: 'uniform at random' },
  { value: 'gapped', label: 'with a gap' },
] as const
type Spacing = (typeof SPACINGS)[number]['value']

const draw = (seed: string, n: number, noise: number, spacing: Spacing) =>
  regression1d(stream(seed), { n, fn: truth, noise, range: [-1, 1], spacing })
const flat = (t: unknown) => Float64Array.from(toFlat(t as never))

/**
 * GCV, exact leave-one-out (by refitting) and the true error of a P-spline smoother over its penalty, with the
 * leverages at one penalty against their average, for three designs.
 */
export function GcvAgainstLoo() {
  const state = useFigureState({
    spacing: choice(SPACINGS, 'gapped', { label: 'inputs' }),
    n: int(40, { ge: 10, le: 120, suggestions: [20, 40, 80], label: 'data points n' }),
    noise: float(0.3, { min: 0.05, max: 1, step: 0.05, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    logLambda: float(-1, { min: -5, max: 2, step: 0.25, label: 'log₁₀ penalty λ shown' }),
  })
  const { spacing, n, noise, seed } = state
  const data = useMemo(() => draw(`gcv/${spacing}/${seed}`, n, noise, spacing), [spacing, n, noise, seed])
  const run = useComputed(
    () => {
      const x = flat(data.x)
      const y = flat(data.y)
      const fresh = draw(`gcv/${spacing}/fresh`, 2000, noise, spacing)
      const path = psplineGcvPath(
        fromData(x, [n]),
        fromData(y, [n]),
        fromData(Float64Array.from(LOG_LAMBDAS), [LOG_LAMBDAS.length]),
        {
          segments: 15,
          range: [-1, 1],
        },
      )
      const loo: number[] = []
      const trueError: number[] = []
      const train: number[] = []
      const leverage: number[][] = []
      for (const l of LOG_LAMBDAS) {
        const learner = psplineLearner(10 ** l)
        const cv = crossValidate(learner, data as never, leaveOneOut(), [meanSquaredError])
        const full = learner.fit(data as never)
        loo.push(cv.mean.meanSquaredError)
        trueError.push(testError(full, fresh as never))
        train.push(meanSquaredError(y, full.fit.fitted))
        leverage.push(Array.from(toFlat(full.fit.leverage)))
      }
      return {
        x: Array.from(x),
        gcv: Array.from(toFlat(path.gcv)),
        edf: Array.from(toFlat(path.edf)),
        loo,
        trueError,
        train,
        leverage,
      }
    },
    [data, spacing, noise, n],
    { mode: 'release' },
  )
  const v = run.value
  const j = Math.max(0, Math.min(LOG_LAMBDAS.length - 1, Math.round((state.logLambda + 5) / 0.25)))
  const argmin = (a: number[]) => a.reduce((b, val, i) => (val < a[b] ? i : b), 0)
  const best = { gcv: argmin(v.gcv), loo: argmin(v.loo), truth: argmin(v.trueError) }
  const lx = useAxis({ label: 'log₁₀ penalty λ', range: [-5, 2], nice: false })
  const ey = useAxis({
    label: 'mean squared error (log scale)',
    log: true,
    hold: 'union',
    key: `${spacing}/${n}/${noise}/${seed}`,
  })
  const hx = useAxis({ label: 'x', range: [-1.05, 1.05], nice: false })
  const hy = useAxis({ label: 'leverage Sᵢᵢ', range: [0, 1] })
  const mean = v.edf[j] / n
  return (
    <Figure
      title="Generalised cross-validation against leave-one-out"
      state={state}
      hoverReadout={false}
      caption="A P-spline smoother (cubic B-splines on 15 intervals of [−1, 1], second-difference penalty λ) fitted to n points of y = sin(πx) + σε. Left: against the penalty, generalised cross-validation, exact leave-one-out by n refits, the training error and the true error of the fit on 2,000 fresh points from the same design. Right: the leverage Sᵢᵢ of each point at the penalty marked by the vertical line, and their average tr S / n (dashed), which GCV uses in place of each one. With evenly spaced inputs the leverages are nearly equal and GCV and leave-one-out agree. With a gap, points at its edges and at the ends of the range have high leverage; leave-one-out inflates their residuals and GCV does not, and the two curves part where the penalty is light. The error axis is logarithmic. Drag the vertical line to move the penalty."
      readouts={
        <>
          <Readout label="effective degrees of freedom" value={fmt(v.edf[j])} />
          <Readout
            label="GCV / LOO / true error"
            value={`${fmt(v.gcv[j])} / ${fmt(v.loo[j])} / ${fmt(v.trueError[j])}`}
          />
          <Readout label="log₁₀ λ chosen by GCV / LOO" value={`${LOG_LAMBDAS[best.gcv]} / ${LOG_LAMBDAS[best.loo]}`} />
          <Readout label="log₁₀ λ with the smallest true error" value={`${LOG_LAMBDAS[best.truth]}`} />
          <Readout
            label="true error at the GCV / LOO choice"
            value={`${fmt(v.trueError[best.gcv])} / ${fmt(v.trueError[best.loo])}`}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={lx} y={ey} title="error estimates against the penalty">
          <Curve name="GCV" x={LOG_LAMBDAS} y={v.gcv} slot={0} stale={run.stale} />
          <Curve name="leave-one-out" x={LOG_LAMBDAS} y={v.loo} slot={1} stale={run.stale} />
          <Curve name="training error" x={LOG_LAMBDAS} y={v.train} slot={2} stale={run.stale} />
          <Curve name="true error" x={LOG_LAMBDAS} y={v.trueError} emphasis dashed stale={run.stale} />
          <Handle
            kind="x"
            at={LOG_LAMBDAS[j]}
            label="λ"
            onDrag={(x) => state.set('logLambda', Math.max(-5, Math.min(2, Math.round(x * 4) / 4)))}
          />
        </Plot>
        <Plot x={hx} y={hy} title="leverages at the marked penalty">
          <Points name="leverage Sᵢᵢ" x={v.x} y={v.leverage[j]} slot={1} stale={run.stale} />
          <Curve name="average tr S / n" x={[-1.05, 1.05]} y={[mean, mean]} emphasis dashed />
        </Plot>
      </div>
    </Figure>
  )
}
