import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

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
  const [name, setName] = useState<keyof typeof FUNCTIONS>('square')
  const [rule, setRule] = useState<Rule>('left')
  const [n, setN] = useState(6)
  const fn = FUNCTIONS[name]
  const [a, b] = fn.range
  const [x0, setX0] = useState(1)
  const x = Math.min(Math.max(x0, a), b)

  const r = useMemo(() => {
    const h = (b - a) / n
    const offset = rule === 'left' ? 0 : rule === 'right' ? 1 : 0.5
    // Each rectangle traced as one closed outline, so the whole set draws as a single shaded line.
    const rx: number[] = []
    const ry: number[] = []
    let sum = 0
    for (let i = 0; i < n; i++) {
      const lo = a + i * h
      const height = fn.f(lo + offset * h)
      sum += height * h
      rx.push(lo, lo, lo + h, lo + h)
      ry.push(0, height, height, 0)
    }
    const xs = linspace(a, b, 300)
    return { h, sum, rx, ry, xs, exact: fn.F(b) - fn.F(a) }
  }, [fn, a, b, n, rule])

  const left: XYSeries[] = [
    { name: `${rule} sum`, type: 'line', x: r.rx, y: r.ry, area: true, slot: 1 },
    { name: `f(x) = ${fn.label}`, type: 'line', x: r.xs, y: r.xs.map(fn.f), slot: 0 },
  ]
  // The tangent to F at x has slope f(x): the first part of the theorem.
  const slope = fn.f(x)
  const span = (b - a) / 4
  const right: XYSeries[] = [
    { name: 'F(x) = area from a to x', type: 'line', x: r.xs, y: r.xs.map((t) => fn.F(t) - fn.F(a)), slot: 0 },
    {
      name: 'tangent, slope f(x)',
      type: 'line',
      x: [x - span, x + span],
      y: [fn.F(x) - fn.F(a) - slope * span, fn.F(x) - fn.F(a) + slope * span],
      slot: 1,
      dashed: true,
    },
  ]
  const handles: Handle[] = [{ kind: 'x', at: x, label: 'x', onDrag: (v) => setX0(Math.min(Math.max(v, a), b)) }]

  return (
    <Interactive
      title="Area as a limit, and its rate of change"
      caption="Left: rectangles whose heights are taken at the left end, the midpoint or the right end of each piece. Their total area approaches the integral as n grows; the midpoint rule gets there fastest. Right: the accumulated area F(x). Drag the line labelled x; the dashed tangent has slope exactly f(x), the height of the curve on the left."
      controls={
        <>
          <ParamChoice
            label="function"
            value={name}
            onChange={setName}
            options={[
              { value: 'square', label: 'x²' },
              { value: 'sine', label: 'sin x' },
              { value: 'bump', label: '1 + x e⁻ˣ' },
            ]}
          />
          <ParamChoice
            label="rule"
            value={rule}
            onChange={setRule}
            options={[
              { value: 'left', label: 'left' },
              { value: 'midpoint', label: 'midpoint' },
              { value: 'right', label: 'right' },
            ]}
          />
          <ParamSlider label="pieces n" value={n} onChange={setN} min={1} max={60} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="Riemann sum" value={formatNumber(r.sum)} />
          <Readout label="integral F(b) − F(a)" value={formatNumber(r.exact)} />
          <Readout label="error" value={formatNumber(r.sum - r.exact)} />
          <Readout label="f(x)" value={formatNumber(slope)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={left} xLabel="x" yLabel="f(x)" xRange={[a, b]} />
        <XYChart series={right} xLabel="x" yLabel="F(x)" xRange={[a, b]} handles={handles} />
      </div>
    </Interactive>
  )
}
