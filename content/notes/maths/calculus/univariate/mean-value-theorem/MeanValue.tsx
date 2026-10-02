import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { linspace } from '@/lib/math'

type Fn = { label: string; f: (x: number) => number; df: (x: number) => number; range: [number, number] }

const FUNCTIONS: Record<'cubic' | 'sin' | 'bump', Fn> = {
  cubic: { label: 'x³ − 3x', f: (x) => x ** 3 - 3 * x, df: (x) => 3 * x * x - 3, range: [-4, 4] },
  sin: {
    label: 'sin 2x + x/2',
    f: (x) => Math.sin(2 * x) + x / 2,
    df: (x) => 2 * Math.cos(2 * x) + 0.5,
    range: [-2.5, 2.5],
  },
  bump: { label: 'e^(−x²)', f: (x) => Math.exp(-x * x), df: (x) => -2 * x * Math.exp(-x * x), range: [-0.4, 1.4] },
}
const DOMAIN: [number, number] = [-2.2, 2.2]
const GRID = linspace(DOMAIN[0], DOMAIN[1], 2001)
/** Half-width of the tangent segments drawn at each mean-value point. */
const HALF = 0.6

/** A secant over [a, b] and every point c in between where the tangent is parallel to it. */
export function MeanValue() {
  const [name, setName] = useState<keyof typeof FUNCTIONS>('cubic')
  const [a, setA] = useState(-1.8)
  const [b, setB] = useState(1.5)
  const fn = FUNCTIONS[name]
  const [lo, hi] = a < b ? [a, b] : [b, a]

  const r = useMemo(() => {
    const slope = (fn.f(hi) - fn.f(lo)) / (hi - lo || 1e-9)
    // The points c in (lo, hi) with f'(c) = slope: sign changes of f' − slope on a fine grid, refined linearly.
    const cs: number[] = []
    for (let i = 1; i < GRID.length; i++) {
      const [x0, x1] = [GRID[i - 1], GRID[i]]
      if (x0 <= lo || x1 >= hi) continue
      const [g0, g1] = [fn.df(x0) - slope, fn.df(x1) - slope]
      if (g0 === 0 || g0 * g1 < 0) cs.push(x0 - (g0 * (x1 - x0)) / (g1 - g0))
    }
    const xs = linspace(DOMAIN[0], DOMAIN[1], 300)
    const series: XYSeries[] = [
      { name: `f(x) = ${fn.label}`, type: 'line', x: xs, y: xs.map(fn.f), slot: 0 },
      { name: 'secant over [a, b]', type: 'line', x: [lo, hi], y: [fn.f(lo), fn.f(hi)], slot: 1 },
      { name: 'mean-value points c', type: 'scatter', x: cs, y: cs.map(fn.f), emphasis: true },
    ]
    const tangents: Segment[] = cs.map((c) => ({
      from: [c - HALF, fn.f(c) - HALF * slope],
      to: [c + HALF, fn.f(c) + HALF * slope],
    }))
    return { series, slope, cs, tangents }
  }, [fn, lo, hi])

  const clamp = (x: number) => Math.round(Math.min(Math.max(x, DOMAIN[0]), DOMAIN[1]) * 100) / 100
  const handles: Handle[] = [
    { kind: 'x', at: a, label: 'a', onDrag: (x) => setA(clamp(x)) },
    { kind: 'x', at: b, label: 'b', onDrag: (x) => setB(clamp(x)) },
  ]

  return (
    <Interactive
      title="A tangent parallel to every secant"
      caption="Drag the lines labelled a and b. The secant joins the curve's values at the two ends; its slope is the average rate of change over [a, b]. The marked points c are where the curve's own slope equals that average. The mean value theorem guarantees at least one such point for every interval."
      controls={
        <ParamChoice
          label="function"
          value={name}
          onChange={setName}
          options={Object.entries(FUNCTIONS).map(([value, f]) => ({
            value: value as keyof typeof FUNCTIONS,
            label: f.label,
          }))}
        />
      }
      readout={
        <>
          <Readout label="average slope over [a, b]" value={formatNumber(r.slope)} />
          <Readout label="points c" value={r.cs.map(formatNumber).join(', ') || '—'} />
        </>
      }
    >
      <XYChart
        series={r.series}
        segments={r.tangents}
        xRange={DOMAIN}
        yRange={fn.range}
        xLabel="x"
        yLabel="y"
        handles={handles}
        height={340}
      />
    </Interactive>
  )
}
