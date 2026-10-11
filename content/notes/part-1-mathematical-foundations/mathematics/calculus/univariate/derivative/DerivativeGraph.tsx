import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

type Fn = {
  label: string
  f: (x: number) => number
  /** The derivative, or NaN where it does not exist (which breaks the curve). */
  df: (x: number) => number
  range: [number, number]
  dRange: [number, number]
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))

const FUNCTIONS = {
  cubic: { label: 'x³ − 3x', f: (x) => x ** 3 - 3 * x, df: (x) => 3 * x * x - 3, range: [-5, 5], dRange: [-4, 12] },
  sin: { label: 'sin x', f: Math.sin, df: Math.cos, range: [-1.5, 1.5], dRange: [-1.5, 1.5] },
  bump: {
    label: 'x² e⁻ˣ',
    f: (x) => x * x * Math.exp(-x),
    df: (x) => x * Math.exp(-x) * (2 - x),
    range: [-0.5, 3],
    dRange: [-2, 2],
  },
  sigmoid: {
    label: 'σ(x) = 1 / (1 + e⁻ˣ)',
    f: sigmoid,
    df: (x) => sigmoid(x) * (1 - sigmoid(x)),
    range: [-0.2, 1.2],
    dRange: [-0.1, 0.4],
  },
  abs: {
    label: '|x|',
    f: Math.abs,
    df: (x) => (Math.abs(x) < 1e-9 ? NaN : Math.sign(x)),
    range: [-0.5, 3.5],
    dRange: [-1.5, 1.5],
  },
} satisfies Record<string, Fn>

type Name = keyof typeof FUNCTIONS

const DOMAIN: [number, number] = [-3, 3]
const XS = Array.from({ length: 401 }, (_, i) => DOMAIN[0] + ((DOMAIN[1] - DOMAIN[0]) * i) / 400)
const ZERO = XS.map(() => 0)
/** Half-width of the tangent segment drawn at a. */
const REACH = 0.8

/** The slope of f at each x, plotted as a height: the graph of f' under the graph of f. */
export function DerivativeGraph() {
  const state = useFigureState({
    name: choice<Name>(
      (Object.keys(FUNCTIONS) as Name[]).map((value) => ({ value, label: FUNCTIONS[value].label })),
      'cubic',
      { label: 'function f(x)' },
    ),
    a: slider(-2.8, 2.8, -1.5, { step: 0.02, label: 'point a', onChart: true }),
  })
  const fn: Fn = FUNCTIONS[state.name]
  const { a } = state
  const setA = state.bind('a').set

  const fx = useMemo(() => XS.map(fn.f), [fn])
  const dfx = useMemo(() => XS.map(fn.df), [fn])
  const fa = fn.f(a)
  const da = fn.df(a)
  const tangent = Number.isFinite(da) ? { x: [a - REACH, a + REACH], y: [fa - da * REACH, fa + da * REACH] } : undefined

  const x = useAxis({ label: 'x', range: DOMAIN })
  const y = useAxis({ label: 'f(x)', range: fn.range })
  const dy = useAxis({ label: "f'(x)", range: fn.dRange })

  const trend = !Number.isFinite(da)
    ? 'kink: no derivative'
    : Math.abs(da) < 0.02
      ? 'flat (stationary point)'
      : da > 0
        ? 'increasing'
        : 'decreasing'

  return (
    <Figure
      title="The derivative as a function"
      state={state}
      caption="Top: f with its tangent at a. Bottom: f', whose height at each x is the slope of f there. Drag the point on either panel to move a. Where f rises, f' is above zero; where f falls, f' is below zero; at a peak or a trough of f the tangent is flat and f' crosses zero. For |x| the slope jumps from −1 to +1 at 0, so f' has a gap there."
      readouts={
        <>
          <Readout label="f(a)" value={formatNumber(fa)} />
          <Readout label="f'(a), the tangent's slope" value={Number.isFinite(da) ? formatNumber(da) : '—'} />
          <Readout label="f at a" value={trend} />
        </>
      }
    >
      <Plots rows={2} hoverGroup>
        <Plot x={x} y={y}>
          <Curve name={`f(x) = ${fn.label}`} x={XS} y={fx} emphasis />
          {tangent && <Curve name="tangent at a" x={tangent.x} y={tangent.y} slot={1} width={3} live />}
          <Handle kind="point" at={[a, fa]} label="a" onDrag={([px]) => setA(Math.min(2.8, Math.max(-2.8, px)))} />
        </Plot>
        <Plot x={x} y={dy}>
          <Curve name="zero" x={XS} y={ZERO} muted dashed silent />
          <Curve name="f'(x)" x={XS} y={dfx} slot={0} />
          {Number.isFinite(da) && <Points name="f'(a)" x={[a]} y={[da]} slot={1} size={9} live />}
          <Handle kind="x" at={a} label="a" onDrag={(px) => setA(Math.min(2.8, Math.max(-2.8, px)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
