import { grad, gradCheck, hessian, hvp, jvp } from 'aifn/autodiff'
import { logGamma, sigmoid, softplus } from 'aifn/special'
import {
  diag,
  dot,
  linspace,
  matmul,
  mul,
  sin,
  square,
  sum,
  tanh,
  tensor,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn/tensor'
import { useMemo, useState } from 'react'
import { Select, Slider, Switch } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type Handle, type XYSeries } from '@lab/viz'
import { formatValue, TensorView } from '@lab/views'

// ── A function and its derivatives ───────────────────────────────────────────────────────────────────────────────────

// Each function's x window is chosen so that, with equal units, the top panel is neither a sliver nor taller than the
// page: the whole curve's shape matters more than a wide range.
const FUNCTIONS = {
  'x sin x': { f: (x: Value) => mul(x, sin(x)), range: [-6, 6] },
  softplus: { f: (x: Value) => softplus(x), range: [-6, 6] },
  sigmoid: { f: (x: Value) => sigmoid(x), range: [-4, 4] },
  'tanh(x)²': { f: (x: Value) => square(tanh(x)), range: [-2.5, 2.5] },
  'log Γ(x)': { f: (x: Value) => logGamma(x), range: [0.3, 5] },
} satisfies Record<string, { f: (x: Value) => Value; range: [number, number] }>
type FunctionName = keyof typeof FUNCTIONS

/** Below this |κ| the curve is treated as straight (an inflection) and no circle is drawn. */
const FLAT = 1e-3
/** Points on the drawn osculating circle. */
const CIRCLE_POINTS = 128

export function DerivativeSpecimen() {
  const [name, setName] = useState<FunctionName>('x sin x')
  const [x0, setX0] = useState(1)
  const [equalUnits, setEqualUnits] = useState(true)
  const { f, range } = FUNCTIONS[name]
  const { xs, ys, d1, d2, yRange } = useMemo(() => {
    const x = linspace(range[0], range[1], 241)
    // For an elementwise f, the gradient of Σ f(x) is f′ at every element; nesting grad gives f″.
    const first = grad((v: Value) => sum(f(v)))
    const second = grad((v: Value) => sum(first(v)))
    const ys = toFlat(f(x) as Tensor)
    const [lo, hi] = [Math.min(...ys), Math.max(...ys)]
    const pad = 0.1 * (hi - lo || 1)
    return {
      xs: toFlat(x),
      ys,
      d1: toFlat(first(x) as Tensor),
      d2: toFlat(second(x) as Tensor),
      // Fixed to the function's view, so neither the circle nor the tangent rescales the axes.
      yRange: [lo - pad, hi + pad] as [number, number],
    }
  }, [f, range])
  const at = Math.min(Math.max(x0, range[0]), range[1])
  const y0 = f(at) as number
  const p = grad(f)(at) as number
  const q = grad(grad(f))(at) as number
  const check = gradCheck(f, at)
  // Curvature of the graph y = f(x): κ = f″ / (1 + f′²)^{3/2}; the osculating circle has radius 1/|κ| and its centre
  // lies on the normal, on the concave side.
  const lift = 1 + p * p
  const kappa = q / lift ** 1.5
  const flat = Math.abs(kappa) < FLAT
  const radius = 1 / Math.abs(kappa)
  const centre: [number, number] = [at - (p * lift) / q, y0 + lift / q]

  const series = useMemo((): XYSeries[] => [{ name, type: 'line', x: xs, y: ys, slot: 0 }], [name, xs, ys])
  const derivatives = useMemo(
    (): XYSeries[] => [
      { name: 'f′ = grad(f)', type: 'line', x: xs, y: d1, slot: 1 },
      { name: 'f″ = grad(grad(f))', type: 'line', x: xs, y: d2, slot: 2, dashed: true },
    ],
    [xs, d1, d2],
  )
  // The tangent, circle, radius and centre follow x₀; they go to the chart as live series (a patch).
  const span = range[1] - range[0]
  const t = Array.from({ length: CIRCLE_POINTS + 1 }, (_, i) => (2 * Math.PI * i) / CIRCLE_POINTS)
  const live: XYSeries[] = [
    {
      name: 'tangent at x₀',
      type: 'line',
      x: [at - span, at + span],
      y: [y0 - p * span, y0 + p * span],
      emphasis: true,
      thin: true,
    },
    {
      name: 'osculating circle',
      type: 'line',
      x: flat ? [] : t.map((a) => centre[0] + radius * Math.cos(a)),
      y: flat ? [] : t.map((a) => centre[1] + radius * Math.sin(a)),
      slot: 3,
      thin: true,
    },
    {
      name: 'osculating circle',
      type: 'line',
      x: flat ? [] : [at, centre[0]],
      y: flat ? [] : [y0, centre[1]],
      slot: 3,
      thin: true,
      dashed: true,
    },
    { name: 'osculating circle', type: 'scatter', x: flat ? [] : [centre[0]], y: flat ? [] : [centre[1]], slot: 3 },
  ]
  const handles: Handle[] = [{ kind: 'x', at, onDrag: setX0, label: 'x₀' }]
  return (
    <Figure
      title="f with its tangent and osculating circle, and f′, f″ from grad"
      defaultSize="L"
      controls={
        <>
          <Select label="f" value={name} onChange={setName} options={Object.keys(FUNCTIONS) as FunctionName[]} />
          <Slider label="x₀" value={at} min={range[0]} max={range[1]} onChange={setX0} />
          <Switch label="equal units" checked={equalUnits} onChange={setEqualUnits} />
        </>
      }
      readouts={
        <>
          <Readout label="f(x₀)" value={formatValue(y0)} />
          <Readout label="f′(x₀)" value={formatValue(p)} />
          <Readout label="f″(x₀)" value={formatValue(q)} />
          <Readout label="κ" value={flat ? 'inflection: κ ≈ 0' : formatValue(kappa)} />
          <Readout label="R = 1/|κ|" value={flat ? '∞' : formatValue(radius)} />
          <Readout label="gradCheck relative error" value={formatValue(check.maxRelError)} />
        </>
      }
      caption="Drag the vertical line on either panel to move x₀. The osculating circle has radius 1/|κ| and matches the curve's value, slope and second derivative at x₀; it is hidden near an inflection, where κ ≈ 0. With equal units (the default) the top panel's height follows from f's range, so the circle is round and both axes still fit the data. Switch equal units off and the axes fit the panel: the circle is then drawn as an ellipse, which still has second-order contact with the curve stretched the same way."
    >
      <Subplots rows={2} sharex heightRatios={[2, 1]} hoverGroup>
        <Panel aspect={equalUnits ? 'equal' : 'fit'}>
          <XYChart
            series={series}
            live={live}
            xLabel="x"
            yLabel="f(x)"
            xRange={range}
            yRange={yRange}
            handles={handles}
          />
        </Panel>
        <Panel>
          <XYChart series={derivatives} xLabel="x" xRange={range} handles={handles} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ── Hessian-vector product on a quadratic ────────────────────────────────────────────────────────────────────────────

export function HvpSpecimen() {
  const [angle, setAngle] = useState(0.5)
  const [condition, setCondition] = useState(4)
  const [v, setV] = useState<[number, number]>([1, 0.4])
  const [x, setX] = useState<[number, number]>([0.3, -0.2])
  // Q = R diag(κ, 1) Rᵀ, and q(x) = ½ xᵀQx, whose Hessian is Q at every x.
  const Q = useMemo(() => {
    const [c, s] = [Math.cos(angle), Math.sin(angle)]
    const R = tensor([
      [c, -s],
      [s, c],
    ])
    return matmul(matmul(R, diag(tensor([condition, 1]))), transpose(R)) as Tensor
  }, [angle, condition])
  const q = useMemo(() => (z: Value) => mul(0.5, dot(z, matmul(Q, z))), [Q])
  const hv = toFlat(hvp(q, tensor(x), tensor(v)) as Tensor)
  const qv = toFlat(matmul(Q, tensor(v)))
  const H = hessian(q)(tensor(x)) as Tensor
  const directional = jvp(grad(q) as (z: Value) => Value, tensor(x), tensor(v)).tangent as Tensor
  const handles: Handle[] = [{ kind: 'point', at: v, onDrag: (p) => setV(p), label: 'v' }]
  const ring = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
  // The unit circle's image under Q, for scale.
  const image = ring.map((t) => toFlat(matmul(Q, tensor([Math.cos(t), Math.sin(t)]))))
  const series: XYSeries[] = [
    {
      name: 'Q·(unit circle)',
      type: 'line',
      x: image.map((p) => p[0]),
      y: image.map((p) => p[1]),
      slot: 0,
      thin: true,
    },
  ]
  return (
    <>
      <Figure
        title="v and H·v for q(x) = ½ xᵀQx"
        controls={
          <>
            <Slider label="rotation" value={angle} min={0} max={Math.PI} onChange={setAngle} />
            <Slider label="condition κ" value={condition} min={1} max={10} onChange={setCondition} />
            <Slider label="x₁" value={x[0]} min={-2} max={2} onChange={(a) => setX([a, x[1]])} />
            <Slider label="x₂" value={x[1]} min={-2} max={2} onChange={(b) => setX([x[0], b])} />
          </>
        }
        readouts={
          <>
            <Readout label="hvp(q, x, v)" value={hv.map(formatValue).join(', ')} />
            <Readout label="Q·v" value={qv.map(formatValue).join(', ')} />
            <Readout label="jvp of grad q along v" value={toFlat(directional).map(formatValue).join(', ')} />
          </>
        }
        caption="Drag the tip of v. The Hessian of a quadratic is Q at every x, so moving x changes nothing."
      >
        <XYChart
          series={series}
          vectors={[
            { from: [0, 0], to: v, slot: 1, label: 'v' },
            { from: [0, 0], to: [hv[0], hv[1]], slot: 2, label: 'H·v' },
          ]}
          handles={handles}
          xRange={[-11, 11]}
          yRange={[-11, 11]}
          equalAspect
        />
      </Figure>
      <TensorView tensor={H} title="hessian(q)(x), equal to Q" initialMode="table" defaultSize="S" />
    </>
  )
}
