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
  const [eq, setEq] = useState<Eq>('linear')
  const [a, setA] = useState(-0.8)
  const [starts, setStarts] = useState<[Pt, Pt]>(SPEC.linear.starts)
  const spec = SPEC[eq]

  const field = useMemo(() => {
    const g = rate(eq, a)
    return directionField((t, x) => [1, g(t, x)], SPEC[eq].t, SPEC[eq].x, 22, 14, { arrows: false })
  }, [eq, a])

  const curves = useMemo(() => {
    const g = rate(eq, a)
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
  }, [eq, a, starts])

  const series = useMemo<XYSeries[]>(
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
          setStarts((prev) => {
            const next: [Pt, Pt] = [prev[0], prev[1]]
            const clamp = (v: number, [lo, hi]: [number, number]) => Math.min(hi, Math.max(lo, v))
            next[i] = [clamp(t, spec.t), clamp(x, spec.x)]
            return next
          }),
      })),
    [starts, spec],
  )

  const choose = (e: Eq) => {
    setEq(e)
    setStarts(SPEC[e].starts)
  }

  return (
    <Interactive
      title="Slope field and solutions"
      caption="Each short segment has the slope ẋ that the equation assigns to that point (t, x). A solution is a curve that is tangent to every segment it passes. Drag either black point to move an initial condition; the curve through it is the unique solution with that starting value. Solutions never cross."
      controls={
        <>
          <ParamChoice
            label="equation"
            value={eq}
            onChange={choose}
            options={(Object.keys(SPEC) as Eq[]).map((k) => ({ value: k, label: SPEC[k].label }))}
          />
          {eq === 'linear' && <ParamSlider label="rate a" value={a} onChange={setA} min={-1.5} max={1.5} step={0.1} />}
        </>
      }
      readout={curves.map((c, i) => (
        <Readout
          key={i}
          label={`solution ${i + 1} at t = ${spec.t[1]}`}
          value={Number.isFinite(c.end) ? formatNumber(c.end) : 'left the plot'}
        />
      ))}
    >
      <XYChart
        height={340}
        xLabel="t"
        yLabel="x"
        series={series}
        segments={field}
        handles={handles}
        xRange={spec.t}
        yRange={spec.x}
      />
    </Interactive>
  )
}
