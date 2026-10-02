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

type Fn = {
  label: string
  f: (x: number) => number
  /** The derivative, or undefined where it does not exist. */
  df: (x: number) => number | undefined
  domain: [number, number]
  range: [number, number]
}

const FUNCTIONS: Record<'square' | 'sin' | 'exp' | 'abs', Fn> = {
  square: { label: 'x²', f: (x) => x * x, df: (x) => 2 * x, domain: [-2.5, 2.5], range: [-1.5, 6.5] },
  sin: { label: 'sin x', f: Math.sin, df: Math.cos, domain: [-4, 4], range: [-2, 2] },
  exp: { label: 'eˣ', f: Math.exp, df: Math.exp, domain: [-2.5, 2], range: [-1, 7.5] },
  abs: {
    label: '|x|',
    f: Math.abs,
    df: (x) => (Math.abs(x) < 1e-9 ? undefined : Math.sign(x)),
    domain: [-2.5, 2.5],
    range: [-1, 2.5],
  },
}

/** The secant through (a, f(a)) and (a + h, f(a + h)) turning into the tangent as h shrinks. */
export function SecantTangent() {
  const [name, setName] = useState<keyof typeof FUNCTIONS>('square')
  const fn = FUNCTIONS[name]
  const a = useParam(1, { min: -2, max: 2, step: 0.05 })
  const h = useParam(1, { min: -1.5, max: 1.5, step: 0.01 })
  const step = Math.abs(h.value) < 0.005 ? 0.005 : h.value

  const r = useMemo(() => {
    const xs = linspace(fn.domain[0], fn.domain[1], 300)
    const fa = fn.f(a.value)
    const slope = (fn.f(a.value + step) - fa) / step
    const derivative = fn.df(a.value)
    const [x0, x1] = fn.domain
    const series: XYSeries[] = [
      { name: `f(x) = ${fn.label}`, type: 'line', x: xs, y: xs.map(fn.f), slot: 0 },
      {
        name: 'secant',
        type: 'line',
        x: [x0, x1],
        y: [fa + slope * (x0 - a.value), fa + slope * (x1 - a.value)],
        slot: 1,
      },
      {
        name: 'points',
        type: 'scatter',
        x: [a.value, a.value + step],
        y: [fa, fn.f(a.value + step)],
        emphasis: true,
      },
    ]
    if (derivative !== undefined) {
      series.splice(2, 0, {
        name: 'tangent',
        type: 'line',
        x: [x0, x1],
        y: [fa + derivative * (x0 - a.value), fa + derivative * (x1 - a.value)],
        slot: 2,
        dashed: true,
      })
    }
    return { series, slope, derivative }
  }, [fn, a.value, step])

  const handles: Handle[] = [{ kind: 'x', at: a.value, label: 'a', onDrag: a.set }]

  return (
    <Interactive
      title="From secant to tangent"
      caption="The secant joins (a, f(a)) and (a + h, f(a + h)); its slope is the difference quotient. As h shrinks towards 0 from either side, the secant turns into the tangent (dashed), whose slope is f′(a). Drag the line labelled a to move the point. For |x| at a = 0 the quotient is +1 for h > 0 and −1 for h < 0, so it has no limit and there is no tangent."
      controls={
        <>
          <ParamChoice
            label="function"
            value={name}
            onChange={setName}
            options={Object.entries(FUNCTIONS).map(([value, f]) => ({
              value: value as keyof typeof FUNCTIONS,
              label: f.label,
            }))}
          />
          <ParamSlider label="point a" param={a} />
          <ParamSlider label="step h" param={h} />
        </>
      }
      readout={
        <>
          <Readout label="difference quotient" value={formatNumber(r.slope)} />
          <Readout label="f′(a)" value={r.derivative === undefined ? 'does not exist' : formatNumber(r.derivative)} />
          <Readout
            label="gap"
            value={r.derivative === undefined ? '—' : formatNumber(Math.abs(r.slope - r.derivative))}
          />
        </>
      }
    >
      <XYChart
        series={r.series}
        xRange={fn.domain}
        yRange={fn.range}
        xLabel="x"
        yLabel="y"
        handles={handles}
        height={340}
      />
    </Interactive>
  )
}
