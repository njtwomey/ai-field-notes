import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

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
const GRID = toFlat(linspace(DOMAIN[0], DOMAIN[1], 2001))
/** Half-width of the tangent segments drawn at each mean-value point. */
const HALF = 0.6

/** A secant over [a, b] and every point c in between where the tangent is parallel to it. */
export function MeanValue() {
  const state = useFigureState({
    name: choice<keyof typeof FUNCTIONS>(
      Object.entries(FUNCTIONS).map(([value, f]) => ({
        value: value as keyof typeof FUNCTIONS,
        label: f.label,
      })),
      'cubic',
      { label: 'function' },
    ),
  })
  const [a, setA] = useState(-1.8)
  const [b, setB] = useState(1.5)
  const fn = FUNCTIONS[state.name]
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
    const xs = toFlat(linspace(DOMAIN[0], DOMAIN[1], 300))
    const series = [
      { name: `f(x) = ${fn.label}`, x: xs, y: xs.map(fn.f), slot: 0 },
      { name: 'secant over [a, b]', x: [lo, hi], y: [fn.f(lo), fn.f(hi)], slot: 1 },
      { name: 'mean-value points c', x: cs, y: cs.map(fn.f), emphasis: true },
    ] as const
    const tangents: Segment[] = cs.map((c) => ({
      from: [c - HALF, fn.f(c) - HALF * slope],
      to: [c + HALF, fn.f(c) + HALF * slope],
    }))
    return { series, slope, cs, tangents }
  }, [fn, lo, hi])

  const clamp = (x: number) => Math.round(Math.min(Math.max(x, DOMAIN[0]), DOMAIN[1]) * 100) / 100

  const xAxis = useAxis({ label: 'x', range: DOMAIN })
  const yAxis = useAxis({ label: 'y', range: fn.range })
  return (
    <Figure
      title="A tangent parallel to every secant"
      state={state}
      caption="Drag the lines labelled a and b. The secant joins the curve's values at the two ends; its slope is the average rate of change over [a, b]. The marked points c are where the curve's own slope equals that average. The mean value theorem guarantees at least one such point for every interval."

      readouts={
        <>
          <Readout label="average slope over [a, b]" value={formatNumber(r.slope)} />
          <Readout label="points c" value={r.cs.map(formatNumber).join(', ') || '—'} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Points {...r.series[2]} />
        <Segments segments={r.tangents} />
        <Handle kind="x" at={a} label="a" onDrag={(x) => setA(clamp(x))} />
        <Handle kind="x" at={b} label="b" onDrag={(x) => setB(clamp(x))} />
      </Plot>
    </Figure>
  )
}
