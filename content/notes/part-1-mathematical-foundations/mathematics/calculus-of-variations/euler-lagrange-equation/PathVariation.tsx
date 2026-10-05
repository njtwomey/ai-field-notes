import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type Functional = 'length' | 'energy'
type Direction = 'sine' | 'double' | 'bump' | 'endpoint'

/** Perturbations η and their derivatives. All but the last vanish at both ends, as the fixed end points require. */
const DIRECTIONS: Record<Direction, { label: string; eta: (x: number) => number; deta: (x: number) => number }> = {
  sine: { label: 'sin πx', eta: (x) => Math.sin(Math.PI * x), deta: (x) => Math.PI * Math.cos(Math.PI * x) },
  double: {
    label: 'sin 2πx',
    eta: (x) => Math.sin(2 * Math.PI * x),
    deta: (x) => 2 * Math.PI * Math.cos(2 * Math.PI * x),
  },
  bump: { label: '4x(1 − x)', eta: (x) => 4 * x * (1 - x), deta: (x) => 4 - 8 * x },
  endpoint: { label: 'x (moves the end)', eta: (x) => x, deta: () => 1 },
}

const INTEGRANDS: Record<Functional, (slope: number) => number> = {
  length: (s) => Math.sqrt(1 + s * s),
  energy: (s) => s * s,
}

const EPS = { min: -0.6, max: 0.6, step: 0.01 }
const GRID = toFlat(linspace(0, 1, 201))
const EPS_GRID = toFlat(linspace(EPS.min, EPS.max, 121))

/** Simpson's rule on the fixed grid of 201 points. */
function simpson(values: number[]): number {
  const h = 1 / (values.length - 1)
  let s = values[0] + values[values.length - 1]
  for (let i = 1; i < values.length - 1; i++) s += (i % 2 ? 4 : 2) * values[i]
  return (s * h) / 3
}

/** J[x + εη] for the straight line y = x from (0, 0) to (1, 1), whose slope is 1. */
function value(functional: Functional, direction: Direction, eps: number): number {
  const { deta } = DIRECTIONS[direction]
  return simpson(GRID.map((x) => INTEGRANDS[functional](1 + eps * deta(x))))
}

/**
 * The straight line from (0, 0) to (1, 1) and a perturbed path y = x + εη. Along any perturbation that keeps the end
 * points fixed, the functional has zero slope at ε = 0: the straight line is stationary.
 */
export function PathVariation() {
  const state = useFigureState({
    functional: choice<Functional>(
      [
        { value: 'length', label: 'length ∫√(1 + y′²)' },
        { value: 'energy', label: 'energy ∫y′²' },
      ],
      'length',
      { label: 'functional J' },
    ),
    direction: choice<Direction>(
      (Object.keys(DIRECTIONS) as Direction[]).map((d) => ({ value: d, label: DIRECTIONS[d].label })),
      'sine',
      { label: 'perturbation η' },
    ),
    eps: slider(EPS.min, EPS.max, 0.3, { step: EPS.step, label: 'ε' }),
  })
  const eps = state.eps
  const { eta } = DIRECTIONS[state.direction]

  const curve = useMemo(
    () => EPS_GRID.map((e) => value(state.functional, state.direction, e)),
    [state.functional, state.direction],
  )
  const current = value(state.functional, state.direction, eps)
  const h = 1e-4
  const slopeAtZero =
    (value(state.functional, state.direction, h) - value(state.functional, state.direction, -h)) / (2 * h)

  const paths = useMemo(
    () =>
      [
        { name: 'straight line', x: GRID, y: GRID, slot: 0, dashed: true },
        { name: 'x + εη', x: GRID, y: GRID.map((x) => x + eps * eta(x)), slot: 1 },
      ] as const,
    [eps, eta],
  )
  const series = useMemo(
    () =>
      [
        { name: 'J(ε)', x: EPS_GRID, y: curve, slot: 0 },
        { name: 'current ε', x: [eps], y: [current], emphasis: true },
      ] as const,
    [curve, eps, current],
  )

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-0.6, 1.6] })
  const xAxis2 = useAxis({ label: 'ε', hold: 'union' })
  const yAxis2 = useAxis({ label: 'J(ε)', hold: 'union' })
  return (
    <Figure
      title="Perturbing the shortest path"
      state={state}
      caption="Left: the straight line between the fixed end points and a perturbed path x + εη. Right: the functional along that family of paths. For every η that vanishes at both ends, J(ε) has zero slope at ε = 0, so the straight line is stationary, and here it is the minimum. Choose 'x (moves the end)' to see the slope at ε = 0 become nonzero: that perturbation leaves the admissible class, which is why the end-point terms matter. Drag the ε line or use the slider."
      readouts={
        <>
          <Readout label="J(ε)" value={formatNumber(current)} />
          <Readout label="J(0), the straight line" value={formatNumber(value(state.functional, state.direction, 0))} />
          <Readout label="dJ/dε at ε = 0" value={formatNumber(Math.abs(slopeAtZero) < 1e-9 ? 0 : slopeAtZero)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...paths[0]} />
          <Curve {...paths[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...series[0]} />
          <Points {...series[1]} />
          <Handle {...state.handle('eps', { label: 'ε' })} />
        </Plot>
      </div>
    </Figure>
  )
}
