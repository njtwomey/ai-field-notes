import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
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
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { crossValidate, leaveOneOut } from 'aifn-compute/learning/validate'
import { unitGrid } from 'aifn-methods/theory'
import { freshSample, psplineLearner, sineData, testError, truth } from '../_shared/problems'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const GRID = Array.from(unitGrid(201))
const NOISE = 0.3

/**
 * Leave-one-out for a P-spline smoother: one movable point, the fits with and without it, its ordinary and
 * leave-one-out residuals and its leverage; and the leave-one-out error of the whole sample beside the true error.
 */
export function LooShortcut() {
  const state = useFigureState({
    n: int(20, { ge: 8, le: 80, suggestions: [10, 20, 40], label: 'other points n' }),
    logLambda: float(-1, { min: -4, max: 2, step: 0.1, label: 'log₁₀ penalty λ' }),
    px: float(0.95, { min: -1, max: 1, step: 0.01, label: 'moved point, x' }),
    py: float(-0.6, { min: -2.5, max: 2.5, step: 0.01, label: 'moved point, y' }),
  })
  const { n, logLambda, px, py } = state
  const base = useMemo(() => sineData(child(stream('loo-shortcut'), 'data'), n, NOISE), [n])
  const fresh = useMemo(() => freshSample(stream('loo-shortcut/fresh'), NOISE, 1000), [])
  const run = useComputed(
    () => {
      const learner = psplineLearner(10 ** logLambda)
      const xs = Float64Array.from([...base.x, px])
      const ys = Float64Array.from([...base.y, py])
      const all = dataset(fromData(xs, [n + 1, 1]), fromData(ys))
      const full = learner.fit(all)
      const without = learner.fit(base.data)
      const cv = crossValidate(learner, all, leaveOneOut(), [meanSquaredError])
      const grid = fromData(Float64Array.from(GRID), [GRID.length, 1])
      const fitted = toFlat(full.fit.fitted)
      const leverage = Array.from(toFlat(full.fit.leverage))
      return {
        curveWith: Array.from(toFlat(full.decide(grid))),
        curveWithout: Array.from(toFlat(without.decide(grid))),
        fittedProbe: fitted[n],
        looProbe: toFlat(cv.outOfFold.decide!)[n],
        leverage,
        xs: Array.from(xs),
        loo: cv.mean.meanSquaredError,
        train: meanSquaredError(ys, fitted),
        trueError: testError(full, fresh),
        edf: full.fit.edf,
      }
    },
    [base, fresh, logLambda, px, py, n],
    { mode: 'frame' },
  )
  const v = run.value
  const ordinary = py - v.fittedProbe
  const loo = py - v.looProbe
  const h = v.leverage[n]
  const fx = useAxis({ label: 'x', range: [-1.05, 1.05], nice: false })
  const fy = useAxis({ label: 'y', range: [-2.5, 2.5] })
  const lx = useAxis({ label: 'x', range: [-1.05, 1.05], nice: false })
  const ly = useAxis({ label: 'leverage Sᵢᵢ', range: [0, 1] })
  return (
    <Figure
      title="Leave-one-out residuals from one fit"
      state={state}
      caption={`A P-spline smoother (cubic B-splines on 15 equal intervals of [−1, 1], second-difference penalty λ) is fitted to n points of y = sin(πx) + 0.3ε and one more point that can be dragged. Left: the fit with the moved point (solid) and the fit refitted without it (dashed); the ink segment is the moved point's ordinary residual, the red one its leave-one-out residual, to the dashed fit. Right: the leverage Sᵢᵢ of every point, the moved one in ink. The leave-one-out residual equals the ordinary residual divided by 1 − Sᵢᵢ, so a point with high leverage, at the edge of the data or where the penalty is light, is predicted much worse without itself than its ordinary residual suggests. The readouts set the leave-one-out error of all n + 1 points beside the training error and the true error, measured on 1,000 fresh points.`}
      readouts={
        <>
          <Readout label="ordinary residual" value={fmt(ordinary)} />
          <Readout label="leave-one-out residual (refit)" value={fmt(loo)} />
          <Readout label="leverage Sᵢᵢ" value={fmt(h)} />
          <Readout label="ordinary ÷ (1 − Sᵢᵢ)" value={fmt(ordinary / (1 - h))} />
          <Readout label="training error" value={fmt(v.train)} />
          <Readout label="leave-one-out error" value={fmt(v.loo)} />
          <Readout label="true error" value={fmt(v.trueError)} />
          <Readout label="effective degrees of freedom" value={fmt(v.edf)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={fx} y={fy} title="fits with and without the moved point">
          <Curve name="sin(πx)" x={GRID} y={GRID.map(truth)} muted dashed />
          <Points name="data" x={Array.from(base.x)} y={Array.from(base.y)} slot={0} />
          <Curve name="fit with the point" x={GRID} y={v.curveWith} slot={1} stale={run.stale} />
          <Curve name="fit without the point" x={GRID} y={v.curveWithout} slot={2} dashed stale={run.stale} />
          <Segments
            name="leave-one-out residual"
            segments={[{ from: [px + 0.02, py], to: [px + 0.02, v.looProbe] }]}
            width={3}
            tone="destructive"
          />
          <Segments
            name="ordinary residual"
            segments={[{ from: [px - 0.02, py], to: [px - 0.02, v.fittedProbe] }]}
            width={3}
            emphasis
          />
          <Handle
            kind="point"
            at={[px, py]}
            label="moved point"
            onDrag={([x, y]) => {
              state.set('px', Math.max(-1, Math.min(1, x)))
              state.set('py', Math.max(-2.5, Math.min(2.5, y)))
            }}
          />
        </Plot>
        <Plot x={lx} y={ly} title="leverage of each point">
          <Points name="leverage" x={v.xs.slice(0, n)} y={v.leverage.slice(0, n)} slot={0} />
          <Points name="moved point" x={[px]} y={[h]} emphasis size={9} />
        </Plot>
      </div>
    </Figure>
  )
}
