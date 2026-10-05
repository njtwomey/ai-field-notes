import { useMemo } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

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
  const state = useFigureState({
    name: choice<keyof typeof FUNCTIONS>(
      Object.entries(FUNCTIONS).map(([value, f]) => ({
        value: value as keyof typeof FUNCTIONS,
        label: f.label,
      })),
      'square',
      { label: 'function' },
    ),
    a: float(1, { min: -2, max: 2, step: 0.05, label: 'point a' }),
    h: float(1, { min: -1.5, max: 1.5, step: 0.01, label: 'step h' }),
  })
  const fn = FUNCTIONS[state.name]
  const step = Math.abs(state.h) < 0.005 ? 0.005 : state.h

  const r = useMemo(() => {
    const xs = toFlat(linspace(fn.domain[0], fn.domain[1], 300))
    const fa = fn.f(state.a)
    const slope = (fn.f(state.a + step) - fa) / step
    const derivative = fn.df(state.a)
    const [x0, x1] = fn.domain
    const series: SeriesSpec[] = [
      { name: `f(x) = ${fn.label}`, type: 'line', x: xs, y: xs.map(fn.f), slot: 0 },
      {
        name: 'secant',
        type: 'line',
        x: [x0, x1],
        y: [fa + slope * (x0 - state.a), fa + slope * (x1 - state.a)],
        slot: 1,
      },
      {
        name: 'points',
        type: 'scatter',
        x: [state.a, state.a + step],
        y: [fa, fn.f(state.a + step)],
        emphasis: true,
      },
    ]
    if (derivative !== undefined) {
      series.splice(2, 0, {
        name: 'tangent',
        type: 'line',
        x: [x0, x1],
        y: [fa + derivative * (x0 - state.a), fa + derivative * (x1 - state.a)],
        slot: 2,
        dashed: true,
      })
    }
    return { series, slope, derivative }
  }, [fn, state.a, step])

  const xAxis = useAxis({ label: 'x', range: fn.domain })
  const yAxis = useAxis({ label: 'y', range: fn.range })
  return (
    <Figure
      title="From secant to tangent"
      state={state}
      caption="The secant joins (a, f(a)) and (a + h, f(a + h)); its slope is the difference quotient. As h shrinks towards 0 from either side, the secant turns into the tangent (dashed), whose slope is f′(a). Drag the line labelled a to move the point. For |x| at a = 0 the quotient is +1 for h > 0 and −1 for h < 0, so it has no limit and there is no tangent."

      readouts={
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
      <Plot x={xAxis} y={yAxis} height={340}>
        {seriesLayers(r.series)}
        <Handle {...state.handle('a', { label: 'a' })} />
      </Plot>
    </Figure>
  )
}
