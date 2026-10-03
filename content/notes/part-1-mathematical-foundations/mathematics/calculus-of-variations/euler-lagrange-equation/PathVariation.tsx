import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

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
const GRID = linspace(0, 1, 201)
const EPS_GRID = linspace(EPS.min, EPS.max, 121)

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
  const [functional, setFunctional] = useState<Functional>('length')
  const [direction, setDirection] = useState<Direction>('sine')
  const eps = useParam(0.3, EPS)
  const { eta } = DIRECTIONS[direction]

  const curve = useMemo(() => EPS_GRID.map((e) => value(functional, direction, e)), [functional, direction])
  const current = value(functional, direction, eps.value)
  const h = 1e-4
  const slopeAtZero = (value(functional, direction, h) - value(functional, direction, -h)) / (2 * h)

  const paths: XYSeries[] = useMemo(
    () => [
      { name: 'straight line', type: 'line', x: GRID, y: GRID, slot: 0, dashed: true },
      { name: 'x + εη', type: 'line', x: GRID, y: GRID.map((x) => x + eps.value * eta(x)), slot: 1 },
    ],
    [eps.value, eta],
  )
  const series: XYSeries[] = useMemo(
    () => [
      { name: 'J(ε)', type: 'line', x: EPS_GRID, y: curve, slot: 0 },
      { name: 'current ε', type: 'scatter', x: [eps.value], y: [current], emphasis: true },
    ],
    [curve, eps.value, current],
  )
  const handles: Handle[] = [{ kind: 'x', at: eps.value, label: 'ε', onDrag: eps.set }]

  return (
    <Interactive
      title="Perturbing the shortest path"
      caption="Left: the straight line between the fixed end points and a perturbed path x + εη. Right: the functional along that family of paths. For every η that vanishes at both ends, J(ε) has zero slope at ε = 0, so the straight line is stationary, and here it is the minimum. Choose 'x (moves the end)' to see the slope at ε = 0 become nonzero: that perturbation leaves the admissible class, which is why the end-point terms matter. Drag the ε line or use the slider."
      controls={
        <>
          <ParamChoice
            label="functional J"
            value={functional}
            onChange={setFunctional}
            options={[
              { value: 'length', label: 'length ∫√(1 + y′²)' },
              { value: 'energy', label: 'energy ∫y′²' },
            ]}
          />
          <ParamChoice
            label="perturbation η"
            value={direction}
            onChange={setDirection}
            options={(Object.keys(DIRECTIONS) as Direction[]).map((d) => ({ value: d, label: DIRECTIONS[d].label }))}
          />
          <ParamSlider label="ε" param={eps} />
        </>
      }
      readout={
        <>
          <Readout label="J(ε)" value={formatNumber(current)} />
          <Readout label="J(0), the straight line" value={formatNumber(value(functional, direction, 0))} />
          <Readout label="dJ/dε at ε = 0" value={formatNumber(Math.abs(slopeAtZero) < 1e-9 ? 0 : slopeAtZero)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={paths} xLabel="x" yLabel="y" xRange={[0, 1]} yRange={[-0.6, 1.6]} height={300} />
        <XYChart series={series} xLabel="ε" yLabel="J(ε)" handles={handles} height={300} />
      </div>
    </Interactive>
  )
}
