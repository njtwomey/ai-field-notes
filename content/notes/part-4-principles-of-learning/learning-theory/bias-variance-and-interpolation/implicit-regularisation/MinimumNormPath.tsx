import { useMemo, useState } from 'react'
import { linearSgdSteps } from 'aifn-methods/learning/linear'
import { leastSquaresWeights } from 'aifn-methods/theory'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'

type Vec = [number, number]
const R = 4
const coord = (initial: number) => slider(-R, R, initial, { step: 0.1, onChart: true })
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const fmtVec = (v: Vec) => `(${fmt(v[0])}, ${fmt(v[1])})`

/** Gradient descent on one squared-error equation in two unknowns, from the origin and from a chosen start. */
function descend(x: Vec, y: number, start: Vec, eta: number, steps: number): Vec[] {
  const alg = linearSgdSteps({
    x: tensor([x]),
    y: tensor([y]),
    alpha: 0,
    schedule: 'constant',
    eta0: eta,
    intercept: false,
    shuffle: false,
  })
  const tr = trace(alg, { weights: tensor(start) }, steps, { record: { w: (s) => toFlat(s.weights) } })
  const flat = tr.series.w.data as Float64Array
  const path: Vec[] = []
  for (let i = 0; i + 1 < flat.length; i += 2) path.push([flat[i], flat[i + 1]])
  return path
}

/**
 * Gradient descent on the single equation xᵀw = y in two unknowns. Every gradient is a multiple of x, so the iterates
 * move along x and keep the start's component along the line of solutions. From the origin they reach the
 * minimum-norm solution; from any other start they reach the start's projection onto the line.
 */
export function MinimumNormPath() {
  const state = useFigureState({
    x1: coord(1),
    x2: coord(2),
    w01: coord(2),
    w02: coord(0),
    y: slider(0.5, 6, 3, { step: 0.1, label: 'target y' }),
    eta: float(0.1, { gt: 0, le: 2, suggestions: [0.02, 0.1, 0.3], label: 'step size η' }),
    steps: int(20, { ge: 1, le: 500, suggestions: [10, 20, 100], label: 'steps' }),
  })
  const x: Vec = useMemo(() => [state.x1, state.x2], [state.x1, state.x2])
  const start: Vec = useMemo(() => [state.w01, state.w02], [state.w01, state.w02])
  const [step, setStep] = useState(0)

  const xx = x[0] * x[0] + x[1] * x[1]
  const ok = xx > 1e-6
  const minNorm: Vec = useMemo(() => {
    if (!ok) return [0, 0]
    const w = leastSquaresWeights(new Float64Array(x), 1, 2, new Float64Array([state.y]))
    return [w[0], w[1]]
  }, [x, state.y, ok])
  const fromZero = useMemo(
    () => (ok ? descend(x, state.y, [0, 0], state.eta, state.steps) : [[0, 0] as Vec]),
    [x, state.y, state.eta, state.steps, ok],
  )
  const fromStart = useMemo(
    () => (ok ? descend(x, state.y, start, state.eta, state.steps) : [start]),
    [x, state.y, start, state.eta, state.steps, ok],
  )
  const t = Math.min(step, fromZero.length - 1, fromStart.length - 1)
  const a = fromZero[t]
  const b = fromStart[t]
  const stable = state.eta * xx < 2

  // The line of solutions {w : xᵀw = y}: through the minimum-norm point, perpendicular to x.
  const n = Math.sqrt(xx) || 1
  const along: Vec = [-x[1] / n, x[0] / n]
  const L = 4 * R
  const lineX = [minNorm[0] - L * along[0], minNorm[0] + L * along[0]]
  const lineY = [minNorm[1] - L * along[1], minNorm[1] + L * along[1]]
  const residual = (w: Vec) => state.y - (x[0] * w[0] + x[1] * w[1])

  const wAxis = useAxis({ label: 'w₁', range: [-R, R] })
  const vAxis = useAxis({ label: 'w₂', range: [-R, R], equal: wAxis })
  return (
    <Figure
      title="Gradient descent finds the minimum-norm solution"
      state={state}
      caption="One training example x (arrow) and target y: every weight vector on the solid line fits it exactly. Gradient descent on ½(y − xᵀw)² moves only along x, so it keeps whatever component along the line it started with. From the origin (first colour) it stops at the minimum-norm solution, the point of the line nearest the origin. From the start w₀ (second colour) it stops at the projection of w₀ onto the line. Drag the tip of x, the start w₀, and scrub the steps. With η‖x‖² ≥ 2 the iterates diverge."
      controls={<Player value={t} onChange={setStep} count={Math.max(fromZero.length, 1)} label="step" />}
      readouts={
        <>
          <Readout label="step t" value={String(t)} />
          <Readout label="minimum-norm w*" value={fmtVec(minNorm)} />
          <Readout label="‖w*‖" value={fmt(Math.hypot(...minNorm))} />
          <Readout label="from 0: ‖wₜ‖" value={fmt(Math.hypot(...a))} />
          <Readout label="from w₀: ‖wₜ‖" value={fmt(Math.hypot(...b))} />
          <Readout label="residual from w₀" value={fmt(residual(b))} />
          <Readout label="contraction 1 − η‖x‖²" value={stable ? fmt(1 - state.eta * xx) : 'unstable'} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={wAxis} y={vAxis}>
          <Curve name="solutions xᵀw = y" x={lineX} y={lineY} muted />
          <Vectors vectors={[{ from: [0, 0], to: x }]} />
          <Curve
            name="path from 0"
            slot={0}
            x={fromZero.slice(0, t + 1).map((p) => p[0])}
            y={fromZero.slice(0, t + 1).map((p) => p[1])}
            showPoints
          />
          <Curve
            name="path from w₀"
            slot={1}
            x={fromStart.slice(0, t + 1).map((p) => p[0])}
            y={fromStart.slice(0, t + 1).map((p) => p[1])}
            showPoints
          />
          <Points name="minimum-norm solution" x={[minNorm[0]]} y={[minNorm[1]]} emphasis size={9} />
          <Handle {...state.handle(['x1', 'x2'], { label: 'x' })} />
          <Handle {...state.handle(['w01', 'w02'], { label: 'w₀' })} />
        </Plot>
      </div>
    </Figure>
  )
}
