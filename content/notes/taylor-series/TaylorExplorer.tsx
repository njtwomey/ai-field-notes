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
import { logFactorial } from '@/lib/math/special'

type Fn = {
  label: string
  f: (x: number) => number
  /** k-th derivative at a. */
  derivative: (k: number, a: number) => number
  domain: [number, number]
  a: [number, number]
}

const FUNCTIONS: Record<'sin' | 'exp' | 'log' | 'geometric', Fn> = {
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
  const [name, setName] = useState<keyof typeof FUNCTIONS>('sin')
  const [order, setOrder] = useState(3)
  const [a, setA] = useState(0)
  const fn = FUNCTIONS[name]
  const centre = Math.min(Math.max(a, fn.a[0]), fn.a[1])
  // The expansion point as a vertical line on the chart, bound to the same state and step as its slider.
  const handles: Handle[] = [
    {
      kind: 'x',
      at: centre,
      label: 'a',
      onDrag: (x) => setA(Number((Math.round(Math.min(Math.max(x, fn.a[0]), fn.a[1]) / 0.05) * 0.05).toFixed(2))),
    },
  ]

  const r = useMemo(() => {
    const coef = Array.from({ length: order + 1 }, (_, k) => fn.derivative(k, centre) / Math.exp(logFactorial(k)))
    const taylor = (x: number) => coef.reduce((s, c, k) => s + c * (x - centre) ** k, 0)
    const xs = linspace(fn.domain[0], fn.domain[1], 400)
    const fy = xs.map(fn.f)
    const finite = fy.filter(Number.isFinite)
    const lo = Math.min(...finite)
    const hi = Math.max(...finite)
    const pad = 0.3 * (hi - lo || 1)
    const series: XYSeries[] = [
      { name: fn.label, type: 'line', x: xs, y: fy, slot: 0 },
      { name: `order ${order} Taylor polynomial`, type: 'line', x: xs, y: xs.map(taylor), slot: 1, dashed: true },
      { name: 'expansion point', type: 'scatter', x: [centre], y: [fn.f(centre)], emphasis: true },
    ]
    const at = centre + 0.5
    return {
      series,
      yRange: [lo - pad, hi + pad] as [number, number],
      errorAt: at,
      error: Math.abs(fn.f(at) - taylor(at)),
    }
  }, [fn, order, centre])

  return (
    <Interactive
      title="Polynomials that hug a function"
      caption="The dashed curve matches the function's value and first n derivatives at the expansion point. Raise the order and the match extends further. For log(1 + x) and 1/(1 − x) it never extends past the nearest singularity, however high the order. Drag the vertical line labelled a, or use the slider, to move the expansion point."
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
          <ParamSlider
            label="order n"
            value={order}
            onChange={setOrder}
            min={0}
            max={15}
            step={1}
            format={(v) => String(v)}
          />
          <ParamSlider
            label="expansion point a"
            value={centre}
            onChange={setA}
            min={fn.a[0]}
            max={fn.a[1]}
            step={0.05}
          />
        </>
      }
      readout={
        <>
          <Readout label={`error at a + 0.5 = ${formatNumber(r.errorAt)}`} value={formatNumber(r.error)} />
        </>
      }
    >
      <XYChart
        height={320}
        series={r.series}
        xRange={fn.domain}
        yRange={r.yRange}
        xLabel="x"
        yLabel="y"
        handles={handles}
      />
    </Interactive>
  )
}
