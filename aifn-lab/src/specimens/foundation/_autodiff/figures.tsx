import { grad, gradCheck, hessian, hvp, jvp } from 'aifn/foundation/autodiff'
import { logGamma, sigmoid, softplus } from 'aifn/numerics/special'
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
} from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Equation, Figure, live, tex } from '@lab/layout'
import { choice, row, setting, slider, useFigureState, useProbe } from '@lab/state'
import { Curve, Handle, Plot, Plots, Points, Probe, Readout, useAxis, Vectors } from '@lab/viz'
import { formatValue, TensorModePanel } from '@lab/views'

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

const FUNCTION_NAMES = Object.keys(FUNCTIONS) as FunctionName[]

export function DerivativeSpecimen() {
  const state = useFigureState({
    name: choice(FUNCTION_NAMES, 'x sin x', { label: 'f' }),
    x0: slider(-6, 6, 2, { label: 'x₀' }),
    equalUnits: setting(true, 'equal units'),
  })
  const name = state.name as FunctionName
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
  const probe = useProbe({ x: state.bind('x0'), label: 'x₀' })
  const at = Math.min(Math.max(probe.x ?? state.x0, range[0]), range[1])
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

  // The tangent, circle, radius and centre follow x₀; they are live layers (a patch).
  const span = range[1] - range[0]
  const t = Array.from({ length: CIRCLE_POINTS + 1 }, (_, i) => (2 * Math.PI * i) / CIRCLE_POINTS)
  const x = useAxis({ label: 'x', range, key: name })
  const yTop = useAxis({ label: 'f(x)', range: yRange, key: name, equal: state.equalUnits ? x : undefined })
  const yBottom = useAxis({ label: 'derivatives', hold: 'initial', key: name })
  return (
    <Figure
      title="f with its tangent and osculating circle, and f′, f″ from grad"
      purpose="grad(f) is f′ and grad(grad(f)) is f″, from each primitive's own derivative rule; at x₀ they fix the tangent's slope and the osculating circle's curvature."
      defaultSize="L"
      state={state}
      equation={
        <Equation>
          {flat
            ? tex`\kappa = \frac{f''(x_0)}{(1 + f'(x_0)^2)^{3/2}} = \frac{${live(q, { digits: 3 })}}{(1 + ${live(p, { digits: 3 })}^2)^{3/2}} \approx 0 \quad \text{(inflection)}`
            : tex`\kappa = \frac{f''(x_0)}{(1 + f'(x_0)^2)^{3/2}} = \frac{${live(q, { digits: 3 })}}{(1 + ${live(p, { digits: 3 })}^2)^{3/2}} = ${live(kappa, { digits: 3, strong: true })}, \quad R = 1/|\kappa| = ${live(radius, { digits: 3 })}`}
        </Equation>
      }
      readouts={{
        'at x₀': (
          <>
            <Readout label="x₀" value={formatValue(at)} />
            <Readout label="f(x₀)" value={formatValue(y0)} />
            <Readout label="f′(x₀)" value={formatValue(p)} />
            <Readout label="f″(x₀)" value={formatValue(q)} />
          </>
        ),
        check: <Readout label="gradCheck relative error" value={formatValue(check.maxRelError)} />,
      }}
      caption="Drag the vertical line on either panel to move x₀. The osculating circle has radius 1/|κ| and matches the curve's value, slope and second derivative at x₀; it is hidden near an inflection, where κ ≈ 0. With equal units (the default) the circle is round. Switch equal units off and the axes fit the panel: the circle is then drawn as an ellipse, which still has second-order contact with the curve stretched the same way."
    >
      <Plots rows={2} heights={[2, 1]} hoverGroup>
        <Plot x={x} y={yTop}>
          <Curve name={name} x={xs} y={ys} slot={0} />
          <Curve
            name="tangent at x₀"
            x={[at - span, at + span]}
            y={[y0 - p * span, y0 + p * span]}
            emphasis
            thin
            live
          />
          <Curve
            name="osculating circle"
            x={flat ? [] : t.map((a) => centre[0] + radius * Math.cos(a))}
            y={flat ? [] : t.map((a) => centre[1] + radius * Math.sin(a))}
            slot={3}
            thin
            live
          />
          <Curve
            name="osculating circle"
            x={flat ? [] : [at, centre[0]]}
            y={flat ? [] : [y0, centre[1]]}
            slot={3}
            thin
            dashed
            live
          />
          <Points name="osculating circle" x={flat ? [] : [centre[0]]} y={flat ? [] : [centre[1]]} slot={3} live />
          <Probe probe={probe} at={y0} />
        </Plot>
        <Plot x={x} y={yBottom}>
          <Curve name="f′ = grad(f)" x={xs} y={d1} slot={1} />
          <Curve name="f″ = grad(grad(f))" x={xs} y={d2} slot={2} dashed />
          <Probe probe={probe} at={p} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── Hessian-vector product on a quadratic ────────────────────────────────────────────────────────────────────────────

const RING = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)

export function HvpSpecimen() {
  const state = useFigureState({
    Q: row('1 · the quadratic', {
      angle: slider(0, Math.PI, 0.5, { label: 'rotation' }),
      condition: slider(1, 10, 4, { label: 'condition κ' }),
    }),
    at: row('2 · the point x (changes nothing)', {
      x1: slider(-2, 2, 0.3, { label: 'x₁' }),
      x2: slider(-2, 2, -0.2, { label: 'x₂' }),
    }),
    vx: slider(-8, 8, 2, { onChart: true }),
    vy: slider(-8, 8, 0.8, { onChart: true }),
  })
  const { angle, condition } = state.Q
  const { x1, x2 } = state.at
  const v: [number, number] = [state.vx, state.vy]
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
  const x = tensor([x1, x2])
  const hv = toFlat(hvp(q, x, tensor(v)) as Tensor)
  const qv = toFlat(matmul(Q, tensor(v)))
  const H = useMemo(() => hessian(q)(tensor([x1, x2])) as Tensor, [q, x1, x2])
  const directional = jvp(grad(q) as (z: Value) => Value, x, tensor(v)).tangent as Tensor
  // The unit circle's image under Q, for scale.
  const image = useMemo(() => {
    const pts = RING.map((t) => toFlat(matmul(Q, tensor([Math.cos(t), Math.sin(t)]))))
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
  }, [Q])
  const xa = useAxis({ label: 'first coordinate', range: [-8, 8] })
  const ya = useAxis({ label: 'second coordinate', range: [-8, 8], equal: xa })
  return (
    <>
      <Figure
        title="v and H·v for q(x) = ½ xᵀQx"
        purpose="hvp(q, x, v) computes H·v without forming H; for a quadratic H = Q at every x, so H·v is Q·v wherever x is."
        state={state}
        equation={
          <Equation>
            {tex`\operatorname{hvp}(q, x, v) = (${live(hv[0], { digits: 4, strong: true })}, ${live(hv[1], { digits: 4, strong: true })}) \quad Q v = (${live(qv[0], { digits: 4 })}, ${live(qv[1], { digits: 4 })})`}
          </Equation>
        }
        readouts={<Readout label="jvp of grad q along v" value={toFlat(directional).map(formatValue).join(', ')} />}
        caption="Drag the tip of v. H·v stretches v most along Q's leading eigenvector (the long axis of Q·(unit circle)); raise κ to stretch it further. Moving x changes nothing, because a quadratic's Hessian is constant."
      >
        <Plot x={xa} y={ya}>
          <Curve name="Q·(unit circle)" x={image.x} y={image.y} slot={0} thin />
          <Vectors
            vectors={[
              { from: [0, 0], to: v, slot: 1, label: 'v' },
              { from: [0, 0], to: [hv[0], hv[1]], slot: 2, label: 'H·v' },
            ]}
            live
          />
          <Handle {...state.handle(['vx', 'vy'], { label: 'v' })} />
        </Plot>
      </Figure>
      <Figure
        title="hessian(q)(x), equal to Q"
        purpose="hessian(q) differentiates grad(q) once more; for the quadratic it returns Q itself, the same at every x."
        defaultSize="S"
        hoverReadout={false}
      >
        <TensorModePanel tensor={H} initialMode="table" />
      </Figure>
    </>
  )
}
