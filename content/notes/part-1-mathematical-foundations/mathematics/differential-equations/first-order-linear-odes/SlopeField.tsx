import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
  when,
} from 'aifn-render'
import { directionField, integrate } from '../_shared/ode'

type Eq = 'linear' | 'logistic' | 'forced'
type Pt = [number, number]

const SPEC: Record<Eq, { t: [number, number]; x: [number, number]; starts: [Pt, Pt]; label: string }> = {
  linear: {
    t: [0, 4],
    x: [-3, 3],
    starts: [
      [0, 2],
      [0, -1],
    ],
    label: 'ẋ = a x',
  },
  logistic: {
    t: [0, 8],
    x: [-0.5, 1.5],
    starts: [
      [0, 0.1],
      [1, 1.4],
    ],
    label: 'ẋ = x(1 − x)',
  },
  forced: {
    t: [0, 10],
    x: [-2, 2],
    starts: [
      [0, 0],
      [0, 1.8],
    ],
    label: 'ẋ = −x + sin t',
  },
}

/** Right-hand side g(t, x) of ẋ = g(t, x) for each equation. */
function rate(eq: Eq, a: number): (t: number, x: number) => number {
  if (eq === 'linear') return (_t, x) => a * x
  if (eq === 'logistic') return (_t, x) => x * (1 - x)
  return (t, x) => -x + Math.sin(t)
}

/** Slope field of a first-order ODE with two draggable initial conditions and the solution through each. */
export function SlopeField() {
  const state = useFigureState({
    eq: choice<Eq>(
      (Object.keys(SPEC) as Eq[]).map((k) => ({ value: k, label: SPEC[k].label })),
      'linear',
      { label: 'equation' },
    ),
    a: float(-0.8, { min: -1.5, max: 1.5, step: 0.1, label: 'rate a', when: when('eq', 'linear') }),
  })
  const eq = state.eq
  // Dragged starts belong to the equation they were dragged on; another equation opens at its own defaults.
  const [picked, setPicked] = useState<{ eq: Eq; starts: [Pt, Pt] }>({ eq: 'linear', starts: SPEC.linear.starts })
  const starts = picked.eq === eq ? picked.starts : SPEC[eq].starts
  const spec = SPEC[eq]

  const field = useMemo(() => {
    const g = rate(eq, state.a)
    return directionField((t, x) => [1, g(t, x)], SPEC[eq].t, SPEC[eq].x, 22, 14, { arrows: false })
  }, [eq, state.a])

  const curves = useMemo(() => {
    const g = rate(eq, state.a)
    const f = (t: number, x: number[]) => [g(t, x[0])]
    const spec = SPEC[eq]
    return starts.map(([t0, x0]) => {
      const back = integrate(f, [x0], t0, spec.t[0], 200, 50)
      const fwd = integrate(f, [x0], t0, spec.t[1], 400, 50)
      const ts = [...back.ts.slice(1).reverse(), ...fwd.ts]
      const xs = [...back.xs.slice(1).reverse(), ...fwd.xs].map((v) => v[0])
      const complete = fwd.ts[fwd.ts.length - 1] >= spec.t[1] - 1e-9
      return { ts, xs, end: complete ? fwd.xs[fwd.xs.length - 1][0] : NaN }
    })
  }, [eq, state.a, starts])

  const series = useMemo<SeriesSpec[]>(
    () => [
      ...curves.map((c, i) => ({ name: `solution ${i + 1}`, type: 'line' as const, x: c.ts, y: c.xs, slot: i })),
      {
        name: 'initial conditions',
        type: 'scatter',
        x: starts.map((s) => s[0]),
        y: starts.map((s) => s[1]),
        emphasis: true,
      },
    ],
    [curves, starts],
  )

  const handles = useMemo<Handle[]>(
    () =>
      starts.map((s, i) => ({
        kind: 'point',
        at: s,
        label: `start ${i + 1}`,
        onDrag: ([t, x]) =>
          setPicked((prev) => {
            const base = prev.eq === eq ? prev.starts : SPEC[eq].starts
            const next: [Pt, Pt] = [base[0], base[1]]
            const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v))
            next[i] = [clamp(t, spec.t), clamp(x, spec.x)]
            return { eq, starts: next }
          }),
      })),
    [starts, spec, eq],
  )

  const xAxis = useAxis({ label: 't', range: spec.t })
  const yAxis = useAxis({ label: 'x', range: spec.x })
  return (
    <Figure
      title="Slope field and solutions"
      state={state}
      caption="Each short segment has the slope ẋ that the equation assigns to that point (t, x). A solution is a curve that is tangent to every segment it passes. Drag either black point to move an initial condition; the curve through it is the unique solution with that starting value. Solutions never cross."
      readouts={curves.map((c, i) => (
        <Readout
          key={i}
          label={`solution ${i + 1} at t = ${spec.t[1]}`}
          value={Number.isFinite(c.end) ? formatNumber(c.end) : 'left the plot'}
        />
      ))}
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(series)}
        <Segments segments={field} />
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
