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
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { sigmoid } from 'aifn/numerics/special'

// A small nonlinear model with an interaction term: f(x) = σ(2 x₁ + x₂ + 1.5 x₁ x₂ − 1).
const score = (x1: number, x2: number) => 2 * x1 + x2 + 1.5 * x1 * x2 - 1
const f = (x1: number, x2: number) => sigmoid(score(x1, x2))
function grad(x1: number, x2: number): [number, number] {
  const p = f(x1, x2)
  const s = p * (1 - p)
  return [s * (2 + 1.5 * x2), s * (1 + 1.5 * x1)]
}

const LO = -2
const HI = 2
const AXIS = toFlat(linspace(LO, HI, 41))
const GRID = AXIS.map((y) => AXIS.map((x) => f(x, y)))
const RANGE: [number, number] = [0, 1]

/**
 * Integrated gradients along the straight line from a baseline to an input, approximated by a midpoint Riemann sum.
 * The input and the baseline are draggable; the readouts compare the attributions' sum with f(x) − f(baseline).
 */
export function IntegratedGradients() {
  const state = useFigureState({
    steps: int(8, { min: 1, max: 64, step: 1, label: 'Riemann steps m' }),
    x1: float(1.5, { min: LO, max: HI, step: 0.05, label: 'input x₁' }),
    x2: float(1, { min: LO, max: HI, step: 0.05, label: 'input x₂' }),
    b1: float(0, { min: LO, max: HI, step: 0.05, label: 'baseline x′₁' }),
    b2: float(0, { min: LO, max: HI, step: 0.05, label: 'baseline x′₂' }),
  })

  const ig = useMemo(() => {
    const d1 = state.x1 - state.b1
    const d2 = state.x2 - state.b2
    let s1 = 0
    let s2 = 0
    const px: number[] = []
    const py: number[] = []
    for (let k = 0; k < state.steps; k++) {
      const a = (k + 0.5) / state.steps
      const p1 = state.b1 + a * d1
      const p2 = state.b2 + a * d2
      const [g1, g2] = grad(p1, p2)
      s1 += g1
      s2 += g2
      px.push(p1)
      py.push(p2)
    }
    return { a1: (d1 * s1) / state.steps, a2: (d2 * s2) / state.steps, px, py }
  }, [state.x1, state.x2, state.b1, state.b2, state.steps])

  const [g1, g2] = grad(state.x1, state.x2)
  const gxi1 = (state.x1 - state.b1) * g1
  const gxi2 = (state.x2 - state.b2) * g2
  const delta = f(state.x1, state.x2) - f(state.b1, state.b2)

  const overlay = [
    { name: 'path', x: [state.b1, state.x1], y: [state.b2, state.x2] },
    { name: 'Riemann points', x: ig.px, y: ig.py },
  ] as const

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="Integrated gradients on a two-feature model"
      purpose="Drag the input and the baseline and change the number of Riemann steps to see how integrated gradients split the change in output between the two features."
      state={state}
      caption="The background is the model output f(x) = σ(2x₁ + x₂ + 1.5x₁x₂ − 1). Drag the input and the baseline. Integrated gradients average the gradient at the Riemann points along the straight path and multiply by the displacement. Their sum approaches f(x) − f(baseline) as the number of steps grows. Gradient × (x − baseline) uses only the gradient at the input, so it misses most of the change when the input sits where the sigmoid has flattened."

      readouts={
        <>
          <Readout label="f(x) − f(x′)" value={formatNumber(delta)} />
          <Readout label="IG₁, IG₂" value={`${formatNumber(ig.a1)}, ${formatNumber(ig.a2)}`} />
          <Readout label="IG₁ + IG₂" value={formatNumber(ig.a1 + ig.a2)} />
          <Readout label="gradient × (x − x′), summed" value={formatNumber(gxi1 + gxi2)} />
        </>
      }
    >
      <Plot
        x={xAxis}
        y={yAxis}
        ariaLabel={'Heatmap of a two-feature model with the straight path from a baseline to an input'}
      >
        <Raster x={AXIS} y={AXIS} z={GRID} scale={'sequential'} range={RANGE} valueLabel={'f(x)'} />
        <Curve {...overlay[0]} live />
        <Points {...overlay[1]} live />
        <Handle
          kind="point"
          at={[state.x1, state.x2]}
          label="input x"
          onDrag={([a, b]) => (state.set('x1', a), state.set('x2', b))}
        />
        <Handle
          kind="point"
          at={[state.b1, state.b2]}
          label="baseline"
          onDrag={([a, b]) => (state.set('b1', a), state.set('b2', b))}
        />
      </Plot>
    </Figure>
  )
}
