import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  row,
  seriesLayers,
  type SeriesSpec,
  setting,
  type SwitchDef,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalPdf, sigmoid } from 'aifn/numerics/special'

type Activation = { id: string; name: string; shown: boolean; f: (x: number) => number; df: (x: number) => number }

const LEAK = 0.1

/** Fixed order, so each activation keeps its colour slot when others are toggled. */
const ACTIVATIONS: Activation[] = [
  {
    id: 'sigmoid',
    name: 'sigmoid',
    shown: true,
    f: (v: number) => sigmoid(v),
    df: (x) => sigmoid(x) * (1 - sigmoid(x)),
  },
  { id: 'tanh', name: 'tanh', shown: true, f: Math.tanh, df: (x) => 1 - Math.tanh(x) ** 2 },
  { id: 'relu', name: 'ReLU', shown: true, f: (x) => Math.max(0, x), df: (x) => (x > 0 ? 1 : 0) },
  {
    id: 'leaky',
    shown: false,
    name: `leaky ReLU (α = ${LEAK})`,
    f: (x) => (x > 0 ? x : LEAK * x),
    df: (x) => (x > 0 ? 1 : LEAK),
  },
  { id: 'elu', name: 'ELU', shown: false, f: (x) => (x > 0 ? x : Math.expm1(x)), df: (x) => (x > 0 ? 1 : Math.exp(x)) },
  { id: 'gelu', name: 'GELU', shown: true, f: (x) => x * normalCdf(x), df: (x) => normalCdf(x) + x * normalPdf(x) },
  {
    id: 'silu',
    name: 'SiLU',
    shown: false,
    f: (x) => x * sigmoid(x),
    df: (x) => sigmoid(x) * (1 + x * (1 - sigmoid(x))),
  },
]

/** One switch per activation, in a row of its own. */
const SHOW = row(
  'show',
  Object.fromEntries(ACTIVATIONS.map((a) => [a.id, setting(a.shown, a.name)])) as Record<string, SwitchDef>,
)

const XS = toFlat(linspace(-5, 5, 401))
const X_RANGE: [number, number] = [-5, 5]
const F_RANGE: [number, number] = [-1.5, 3]
const DF_RANGE: [number, number] = [-0.25, 1.25]

export function ActivationExplorer() {
  const state = useFigureState({
    show: SHOW,
    at: float(-2, { min: -5, max: 5, step: 0.05, label: 'input x' }),
  })

  const show = state.show
  const shown = useMemo(() => ACTIVATIONS.map((a) => show[a.id]), [show])
  const curves = useMemo(() => {
    const pick = ACTIVATIONS.flatMap((a, slot) => (shown[slot] ? [{ a, slot }] : []))
    const values: SeriesSpec[] = pick.map(({ a, slot }) => ({
      name: a.name,
      type: 'line',
      x: XS,
      y: XS.map(a.f),
      slot,
    }))
    const slopes: SeriesSpec[] = pick.map(({ a, slot }) => ({
      name: a.name,
      type: 'line',
      x: XS,
      y: XS.map(a.df),
      slot,
    }))
    return { values, slopes }
  }, [shown])

  const x = state.at

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'φ(x)', range: F_RANGE })
  const xAxis2 = useAxis({ label: 'x', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'φ′(x)', range: DF_RANGE })
  return (
    <Figure
      title="Activation functions and their derivatives"
      state={state}
      caption="Left: the activation φ(x). Right: its derivative φ′(x), the factor by which the activation passes a gradient back. Drag the vertical line on either chart, or use the slider, to read both at a point. Sigmoid and tanh have derivatives near 0 once |x| exceeds about 3; ReLU passes gradient 1 for x > 0 and nothing for x < 0; GELU and SiLU are smooth versions of ReLU that dip slightly below zero."
      readouts={ACTIVATIONS.flatMap((a, i) =>
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
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(curves.values)}
          <Handle {...state.handle('at', { label: 'x' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(curves.slopes)}
          <Handle {...state.handle('at', { label: 'x' })} />
        </Plot>
      </div>
    </Figure>
  )
}
