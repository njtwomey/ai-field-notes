import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normalCdf } from 'aifn-compute/numerics/special'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const X = toFlat(linspace(-3, 5, 241))
const D_GRID = toFlat(linspace(0.005, 0.5, 100))
const logit = (p: number) => Math.log(p / (1 - p))

/**
 * Chow's reject option on a binormal problem: x is N(0, 1) for negatives and N(sep, 1) for positives, equal priors, so
 * the posterior is p(x) = σ(sep·x − sep²/2). A case is rejected when max(p, 1 − p) < 1 − d. The left chart shows the
 * posterior with the reject band; the right chart is the error-reject curve traced by varying d.
 */
export function RejectBand() {
  const state = useFigureState({
    d: float(0.2, { min: 0.01, max: 0.5, step: 0.005, label: 'rejection cost d' }),
    sep: float(2, { min: 0.5, max: 4, step: 0.1, label: 'class separation' }),
  })

  // Scores where the posterior equals d and 1 − d.
  const band = (dd: number, s: number) => {
    const lo = (logit(dd) + (s * s) / 2) / s
    const hi = (logit(1 - dd) + (s * s) / 2) / s
    return { lo, hi }
  }
  const rates = (dd: number, s: number) => {
    const { lo, hi } = band(dd, s)
    const error = 0.5 * (1 - normalCdf(hi)) + 0.5 * normalCdf(lo - s)
    const reject = 0.5 * (normalCdf(hi) - normalCdf(lo)) + 0.5 * (normalCdf(hi - s) - normalCdf(lo - s))
    return { error, reject, lo, hi }
  }

  const post = useMemo(
    () => X.map((x) => 1 / (1 + Math.exp(-(state.sep * x - (state.sep * state.sep) / 2)))),
    [state.sep],
  )
  const curve = useMemo(() => {
    const pts = D_GRID.map((dd) => rates(dd, state.sep))
    return { reject: pts.map((q) => q.reject), error: pts.map((q) => q.error) }
  }, [state.sep])
  const at = rates(state.d, state.sep)

  const xAxis = useAxis({ label: 'score x', range: [-3, 5] })
  const yAxis = useAxis({ label: 'P(y = 1 | x)', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'reject rate', range: [0, 1] })
  const yAxis2 = useAxis({ label: 'error rate', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Rejecting the least certain cases"
      state={state}
      caption="Left: the posterior P(y = 1 | x) for two unit-variance Gaussian classes with equal priors. Cases whose larger class probability is below 1 − d fall in the shaded band and are rejected. Drag the horizontal line at 1 − d, or use the slider. Right: the error rate against the reject rate as d varies from 0.5 (no rejection) towards 0. The marked point is the current d; the curve's slope there equals −d."

      readouts={
        <>
          <Readout label="reject band on x" value={`[${formatNumber(at.lo)}, ${formatNumber(at.hi)}]`} />
          <Readout label="error rate" value={formatNumber(at.error)} />
          <Readout label="reject rate" value={formatNumber(at.reject)} />
          <Readout label="expected loss" value={formatNumber(at.error + state.d * at.reject)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve name="posterior" x={X} y={post} slot={0} />
          <Area name="reject band" x={[at.lo, at.lo, at.hi, at.hi]} y={[0, 1, 1, 0]} slot={2} />
          <Curve name="d" x={[-3, 5]} y={[state.d, state.d]} dashed muted />
          <Handle kind="y" at={1 - state.d} label="accept above 1 − d" onDrag={(y) => state.set('d', 1 - y)} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={280}>
          <Curve name="error-reject curve" x={curve.reject} y={curve.error} slot={1} />
          <Points name="current d" x={[at.reject]} y={[at.error]} emphasis />
        </Plot>
      </div>
    </Figure>
  )
}
