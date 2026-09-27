import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, sigmoid } from '@/lib/math'
import { normalCdf, normalPdf } from '@/lib/math/special'

type Activation = { name: string; f: (x: number) => number; df: (x: number) => number }

const LEAK = 0.1

/** Fixed order, so each activation keeps its colour slot when others are toggled. */
const ACTIVATIONS: Activation[] = [
  { name: 'sigmoid', f: sigmoid, df: (x) => sigmoid(x) * (1 - sigmoid(x)) },
  { name: 'tanh', f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2 },
  { name: 'ReLU', f: (x) => Math.max(0, x), df: (x) => (x > 0 ? 1 : 0) },
  { name: `leaky ReLU (α = ${LEAK})`, f: (x) => (x > 0 ? x : LEAK * x), df: (x) => (x > 0 ? 1 : LEAK) },
  { name: 'ELU', f: (x) => (x > 0 ? x : Math.expm1(x)), df: (x) => (x > 0 ? 1 : Math.exp(x)) },
  { name: 'GELU', f: (x) => x * normalCdf(x), df: (x) => normalCdf(x) + x * normalPdf(x) },
  { name: 'SiLU', f: (x) => x * sigmoid(x), df: (x) => sigmoid(x) * (1 + x * (1 - sigmoid(x))) },
]

const XS = linspace(-5, 5, 401)
const X_RANGE: [number, number] = [-5, 5]
const F_RANGE: [number, number] = [-1.5, 3]
const DF_RANGE: [number, number] = [-0.25, 1.25]

export function ActivationExplorer() {
  const [shown, setShown] = useState<boolean[]>([true, true, true, false, false, true, false])
  const at = useParam(-2, { min: -5, max: 5, step: 0.05 })

  const curves = useMemo(() => {
    const pick = ACTIVATIONS.flatMap((a, slot) => (shown[slot] ? [{ a, slot }] : []))
    const values: XYSeries[] = pick.map(({ a, slot }) => ({ name: a.name, type: 'line', x: XS, y: XS.map(a.f), slot }))
    const slopes: XYSeries[] = pick.map(({ a, slot }) => ({ name: a.name, type: 'line', x: XS, y: XS.map(a.df), slot }))
    return { values, slopes }
  }, [shown])

  const handles: Handle[] = [{ kind: 'x', at: at.value, label: 'x', onDrag: at.set }]
  const x = at.value

  return (
    <Interactive
      title="Activation functions and their derivatives"
      caption="Left: the activation φ(x). Right: its derivative φ′(x), the factor by which the activation passes a gradient back. Drag the vertical line on either chart, or use the slider, to read both at a point. Sigmoid and tanh have derivatives near 0 once |x| exceeds about 3; ReLU passes gradient 1 for x > 0 and nothing for x < 0; GELU and SiLU are smooth versions of ReLU that dip slightly below zero."
      controls={
        <>
          <ParamSlider label="input x" param={at} />
          <div className="col-span-full flex flex-wrap gap-x-5 gap-y-2">
            {ACTIVATIONS.map((a, i) => (
              <ParamSwitch
                key={a.name}
                label={a.name}
                checked={shown[i]}
                onChange={(v) => setShown((s) => s.map((old, j) => (j === i ? v : old)))}
              />
            ))}
          </div>
        </>
      }
      readout={ACTIVATIONS.flatMap((a, i) =>
        shown[i]
          ? [
              <Readout
                key={a.name}
                label={`${a.name}: φ, φ′ =`}
                value={`${formatNumber(a.f(x))}, ${formatNumber(a.df(x))}`}
              />,
            ]
          : [],
      )}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={curves.values} xRange={X_RANGE} yRange={F_RANGE} xLabel="x" yLabel="φ(x)" handles={handles} />
        <XYChart
          series={curves.slopes}
          xRange={X_RANGE}
          yRange={DF_RANGE}
          xLabel="x"
          yLabel="φ′(x)"
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
