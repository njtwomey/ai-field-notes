import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { logFactorial } from 'aifn-compute/numerics/special'

type Fn = {
  label: string
  f: (x: number) => number
  /** k-th derivative at a. */
  derivative: (k: number, a: number) => number
  domain: [number, number]
  a: [number, number]
}

type FnId = 'sin' | 'exp' | 'log' | 'geometric'

const FUNCTIONS: Record<FnId, Fn> = {
  sin: {
    label: 'sin x',
    f: Math.sin,
    derivative: (k, a) => Math.sin(a + (k * Math.PI) / 2),
    domain: [-8, 8],
    a: [-4, 4],
  },
  exp: { label: 'eˣ', f: Math.exp, derivative: (_, a) => Math.exp(a), domain: [-4, 4], a: [-2, 2] },
  log: {
    label: 'log(1 + x)',
    f: (x) => (x > -1 ? Math.log1p(x) : NaN),
    // d^k/dx^k log(1 + x) = (−1)^{k−1} (k − 1)! / (1 + x)^k for k ≥ 1.
    derivative: (k, a) => (k === 0 ? Math.log1p(a) : ((-1) ** (k - 1) * Math.exp(logFactorial(k - 1))) / (1 + a) ** k),
    domain: [-0.99, 3],
    a: [-0.5, 1.5],
  },
  geometric: {
    label: '1 / (1 − x)',
    f: (x) => 1 / (1 - x),
    // d^k/dx^k (1 − x)^{−1} = k! / (1 − x)^{k+1}.
    derivative: (k, a) => Math.exp(logFactorial(k)) / (1 - a) ** (k + 1),
    domain: [-2.5, 0.95],
    a: [-1.5, 0.5],
  },
}

/** A function and its Taylor polynomial of order n about a. */
export function TaylorExplorer() {
  const state = useFigureState({
    name: variants(
      Object.fromEntries(
        (Object.keys(FUNCTIONS) as FnId[]).map((k) => {
          const f = FUNCTIONS[k]
          const a = slider(f.a[0], f.a[1], 0, { step: 0.05, label: 'expansion point a' })
          return [k, { label: f.label, params: { a } }]
        }),
      ) as Record<FnId, { label: string; params: { a: ReturnType<typeof slider> } }>,
      { choiceLabel: 'function' },
    ),
    order: int(3, { min: 0, max: 15, step: 1, label: 'order n', format: (v) => String(v) }),
  })
  const fn = FUNCTIONS[state.name.key]
  const centre = state.name.values.a

  const r = useMemo(() => {
    const coef = Array.from({ length: state.order + 1 }, (_, k) => fn.derivative(k, centre) / Math.exp(logFactorial(k)))
    const taylor = (x: number) => coef.reduce((s, c, k) => s + c * (x - centre) ** k, 0)
    const xs = toFlat(linspace(fn.domain[0], fn.domain[1], 400))
    const fy = xs.map(fn.f)
    const finite = fy.filter(Number.isFinite)
    const lo = Math.min(...finite)
    const hi = Math.max(...finite)
    const pad = 0.3 * (hi - lo || 1)
    const series = [
      { name: fn.label, x: xs, y: fy, slot: 0 },
      { name: `order ${state.order} Taylor polynomial`, x: xs, y: xs.map(taylor), slot: 1, dashed: true },
      { name: 'expansion point', x: [centre], y: [fn.f(centre)], emphasis: true },
    ] as const
    const at = centre + 0.5
    return {
      series,
      yRange: [lo - pad, hi + pad] as [number, number],
      errorAt: at,
      error: Math.abs(fn.f(at) - taylor(at)),
    }
  }, [fn, state.order, centre])

  const xAxis = useAxis({ label: 'x', range: fn.domain })
  const yAxis = useAxis({ label: 'y', range: r.yRange })
  return (
    <Figure
      title="Polynomials that hug a function"
      state={state}
      caption="The dashed curve matches the function's value and first n derivatives at the expansion point. Raise the order and the match extends further. For log(1 + x) and 1/(1 − x) it never extends past the nearest singularity, however high the order. Drag the vertical line labelled a, or use the slider, to move the expansion point."
      readouts={
        <>
          <Readout label={`error at a + 0.5 = ${formatNumber(r.errorAt)}`} value={formatNumber(r.error)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Points {...r.series[2]} />
        <Handle {...state.handle('name.a', { label: 'a' })} />
      </Plot>
    </Figure>
  )
}
