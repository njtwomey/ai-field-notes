import { useMemo, useState } from 'react'
import {
  Area,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type Rule = 'left' | 'midpoint' | 'right'
type Fn = { label: string; f: (x: number) => number; F: (x: number) => number; range: [number, number] }

const FUNCTIONS: Record<'square' | 'sine' | 'bump', Fn> = {
  square: { label: 'x²', f: (x) => x * x, F: (x) => (x * x * x) / 3, range: [0, 2] },
  sine: { label: 'sin x', f: Math.sin, F: (x) => 1 - Math.cos(x), range: [0, Math.PI] },
  bump: {
    label: '1 + x e⁻ˣ',
    f: (x) => 1 + x * Math.exp(-x),
    // ∫₀ˣ t e⁻ᵗ dt = 1 − (1 + x) e⁻ˣ, by parts.
    F: (x) => x + 1 - (1 + x) * Math.exp(-x),
    range: [0, 4],
  },
}

/**
 * Riemann sums converging to the integral (left), and the accumulated area F(x) whose slope at the dragged point equals
 * f(x) there (right): the two halves of the fundamental theorem.
 */
export function RiemannSums() {
  const state = useFigureState({
    name: choice<keyof typeof FUNCTIONS>(
      [
        { value: 'square', label: 'x²' },
        { value: 'sine', label: 'sin x' },
        { value: 'bump', label: '1 + x e⁻ˣ' },
      ],
      'square',
      { label: 'function' },
    ),
    rule: choice<Rule>(
      [
        { value: 'left', label: 'left' },
        { value: 'midpoint', label: 'midpoint' },
        { value: 'right', label: 'right' },
      ],
      'left',
      { label: 'rule' },
    ),
    n: int(6, { min: 1, max: 60, step: 1, label: 'pieces n' }),
  })
  const fn = FUNCTIONS[state.name]
  const [a, b] = fn.range
  const [x0, setX0] = useState(1)
  const x = Math.min(Math.max(x0, a), b)

  const r = useMemo(() => {
    const h = (b - a) / state.n
    const offset = state.rule === 'left' ? 0 : state.rule === 'right' ? 1 : 0.5
    // Each rectangle traced as one closed outline, so the whole set draws as a single shaded line.
    const rx: number[] = []
    const ry: number[] = []
    let sum = 0
    for (let i = 0; i < state.n; i++) {
      const lo = a + i * h
      const height = fn.f(lo + offset * h)
      sum += height * h
      rx.push(lo, lo, lo + h, lo + h)
      ry.push(0, height, height, 0)
    }
    const xs = toFlat(linspace(a, b, 300))
    return { h, sum, rx, ry, xs, exact: fn.F(b) - fn.F(a) }
  }, [fn, a, b, state.n, state.rule])

  const left = [
    { name: `${state.rule} sum`, x: r.rx, y: r.ry, slot: 1 },
    { name: `f(x) = ${fn.label}`, x: r.xs, y: r.xs.map(fn.f), slot: 0 },
  ] as const
  // The tangent to F at x has slope f(x): the first part of the theorem.
  const slope = fn.f(x)
  const span = (b - a) / 4
  const right = [
    { name: 'F(x) = area from a to x', x: r.xs, y: r.xs.map((t) => fn.F(t) - fn.F(a)), slot: 0 },
    {
      name: 'tangent, slope f(x)',
      x: [x - span, x + span],
      y: [fn.F(x) - fn.F(a) - slope * span, fn.F(x) - fn.F(a) + slope * span],
      slot: 1,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [a, b] })
  const yAxis = useAxis({ label: 'f(x)', hold: 'union' })
  const xAxis2 = useAxis({ label: 'x', range: [a, b] })
  const yAxis2 = useAxis({ label: 'F(x)', hold: 'union' })
  return (
    <Figure
      title="Area as a limit, and its rate of change"
      state={state}
      caption="Left: rectangles whose heights are taken at the left end, the midpoint or the right end of each piece. Their total area approaches the integral as n grows; the midpoint rule gets there fastest. Right: the accumulated area F(x). Drag the line labelled x; the dashed tangent has slope exactly f(x), the height of the curve on the left."

      readouts={
        <>
          <Readout label="Riemann sum" value={formatNumber(r.sum)} />
          <Readout label="integral F(b) − F(a)" value={formatNumber(r.exact)} />
          <Readout label="error" value={formatNumber(r.sum - r.exact)} />
          <Readout label="f(x)" value={formatNumber(slope)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          <Area {...left[0]} />
          <Curve {...left[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...right[0]} />
          <Curve {...right[1]} />
          <Handle kind="x" at={x} label="x" onDrag={(v) => setX0(Math.min(Math.max(v, a), b))} />
        </Plot>
      </div>
    </Figure>
  )
}
