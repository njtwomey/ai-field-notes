import { useMemo, useState } from 'react'
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
import {
  ControlGroup,
  Curve,
  Figure,
  Handle,
  NumberSelector,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Switch,
  Vectors,
  useAxis,
} from 'aifn-render'

const fmt = (v: number | null | undefined, digits = 3) =>
  v !== null && v !== undefined && Number.isFinite(v) ? String(Number(v.toPrecision(digits))) : '—'

// ── 1. Hessian-Vector Product Explorer ─────────────────────────────────────────────────────────────────────────────

const RING = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)

export function HvpExplorer() {
  const [angle, setAngle] = useState(0.5)
  const [condition, setCondition] = useState(4.0)
  const [x1, setX1] = useState(0.3)
  const [x2, setX2] = useState(-0.2)
  const [v, setV] = useState<[number, number]>([2.0, 0.8])

  // Q = R diag(κ, 1) Rᵀ, and q(x) = ½ xᵀ Q x, whose Hessian is Q at every x.
  const Q = useMemo(() => {
    const [c, s] = [Math.cos(angle), Math.sin(angle)]
    const R = tensor([
      [c, -s],
      [s, c],
    ])
    return matmul(matmul(R, diag(tensor([condition, 1]))), transpose(R)) as Tensor
  }, [angle, condition])

  const q = useMemo(() => (z: Value) => mul(0.5, dot(z, matmul(Q, z))), [Q])
  const xTen = useMemo(() => tensor([x1, x2]), [x1, x2])
  const vTen = useMemo(() => tensor(v), [v])

  // Forward-over-reverse Hessian-vector product
  const hv = useMemo(() => toFlat(hvp(q, xTen, vTen) as Tensor), [q, xTen, vTen])
  // Direct matrix product Q v
  const qv = useMemo(() => toFlat(matmul(Q, vTen)), [Q, vTen])
  // Full Hessian matrix evaluation
  const H = useMemo(() => toFlat(hessian(q)(xTen) as Tensor), [q, xTen])
  // Forward-mode directional derivative of gradient (JVP of grad)
  const directional = useMemo(
    () => toFlat(jvp(grad(q) as (z: Value) => Value, xTen, vTen).tangent as Tensor),
    [q, xTen, vTen],
  )

  // Unit circle mapped through Q, showing stretching along eigenvectors
  const image = useMemo(() => {
    const pts = RING.map((t) => toFlat(matmul(Q, tensor([Math.cos(t), Math.sin(t)]))))
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
  }, [Q])

  const diffNorm = Math.hypot(hv[0] - qv[0], hv[1] - qv[1])

  const xa = useAxis({ label: 'First Coordinate', range: [-8, 8], nice: false })
  const ya = useAxis({ label: 'Second Coordinate', range: [-8, 8], equal: xa, nice: false })

  return (
    <Figure
      title="Hessian-Vector Product: Action of H without Materializing Matrices"
      purpose="hvp(q, x, v) computes the action of the Hessian H · v via a forward-over-reverse pass in O(n) operations, without computing or storing the O(n²) Hessian matrix."
      defaultSize="L"
      controls={
        <div className="flex flex-col gap-3">
          <ControlGroup title="1 · Quadratic Geometry (Q)">
            <NumberSelector
              label="Principal Rotation Angle θ"
              value={angle}
              onChange={setAngle}
              min={0}
              max={Math.PI}
              step={0.1}
              suggestions={[0, 0.5, 1.0, 1.57]}
            />
            <NumberSelector
              label="Condition Number κ (Eigenvalue Ratio)"
              value={condition}
              onChange={setCondition}
              min={1}
              max={10}
              step={0.5}
              suggestions={[1, 2, 4, 8]}
            />
          </ControlGroup>
          <ControlGroup title="2 · Base Point x and Direction Vector v">
            <NumberSelector
              label="Point x₁"
              value={x1}
              onChange={setX1}
              min={-2}
              max={2}
              step={0.1}
              suggestions={[-1, 0, 1]}
            />
            <NumberSelector
              label="Point x₂"
              value={x2}
              onChange={setX2}
              min={-2}
              max={2}
              step={0.1}
              suggestions={[-1, 0, 1]}
            />
            <NumberSelector
              label="Vector v₁"
              value={v[0]}
              onChange={(val) => setV([val, v[1]])}
              min={-6}
              max={6}
              step={0.2}
              suggestions={[-2, 0, 2]}
            />
            <NumberSelector
              label="Vector v₂"
              value={v[1]}
              onChange={(val) => setV([v[0], val])}
              min={-6}
              max={6}
              step={0.2}
              suggestions={[-1, 0.8, 2]}
            />
          </ControlGroup>
        </div>
      }
      readouts={{
        'Hessian-Vector Product Comparisons': (
          <>
            <Readout label="Input Vector v" value={`(${fmt(v[0])}, ${fmt(v[1])})`} />
            <Readout label="hvp(q, x, v)" value={`(${fmt(hv[0])}, ${fmt(hv[1])})`} />
            <Readout label="Direct Q · v" value={`(${fmt(qv[0])}, ${fmt(qv[1])})`} />
            <Readout label="jvp(grad(q), x, v)" value={`(${fmt(directional[0])}, ${fmt(directional[1])})`} />
            <Readout label="Absolute Difference ‖hvp − Qv‖" value={fmt(diffNorm, 2)} />
          </>
        ),
        'Hessian Matrix Entries H = Q': (
          <>
            <Readout label="H₁₁" value={fmt(H[0])} />
            <Readout label="H₁₂ = H₂₁" value={fmt(H[1])} />
            <Readout label="H₂₂" value={fmt(H[3])} />
          </>
        ),
      }}
      caption="Drag the tip of vector v on the chart. hvp(q, x, v) stretches v predominantly along the dominant eigenvector of Q (the major axis of the ellipse Q · unit circle). Increasing the condition number κ magnifies this directional elongation. Notice that shifting x has zero effect on the result, as the Hessian of a quadratic function is constant everywhere."
    >
      <Plot x={xa} y={ya}>
        <Curve name="Q · (Unit Circle)" x={image.x} y={image.y} slot={0} thin />
        <Vectors
          vectors={[
            { from: [0, 0], to: v, slot: 1, label: 'v' },
            { from: [0, 0], to: [hv[0], hv[1]], slot: 2, label: 'H · v' },
          ]}
        />
        <Handle kind="point" at={v} onDrag={setV} label="v" />
      </Plot>
    </Figure>
  )
}

// ── 2. Higher-Order Derivatives & Curvature Explorer ───────────────────────────────────────────────────────────────

const FUNCTIONS = {
  'x sin x': { f: (x: Value) => mul(x, sin(x)), range: [-6, 6] as [number, number] },
  softplus: { f: (x: Value) => softplus(x), range: [-6, 6] as [number, number] },
  sigmoid: { f: (x: Value) => sigmoid(x), range: [-4, 4] as [number, number] },
  'tanh(x)²': { f: (x: Value) => square(tanh(x)), range: [-2.5, 2.5] as [number, number] },
  'log Γ(x)': { f: (x: Value) => logGamma(x), range: [0.3, 5] as [number, number] },
}
type FunctionName = keyof typeof FUNCTIONS

const FUNCTION_OPTIONS = [
  { value: 'x sin x', label: 'x · sin(x)' },
  { value: 'softplus', label: 'Softplus ln(1 + eˣ)' },
  { value: 'sigmoid', label: 'Sigmoid σ(x)' },
  { value: 'tanh(x)²', label: 'tanh(x)²' },
  { value: 'log Γ(x)', label: 'Log-Gamma ln Γ(x)' },
]

const FLAT_THRESHOLD = 1e-3
const CIRCLE_POINTS = 120

export function HigherOrderDerivativesExplorer() {
  const [funcName, setFuncName] = useState<FunctionName>('x sin x')
  const [x0, setX0] = useState(2.0)
  const [equalUnits, setEqualUnits] = useState(true)

  const { f, range } = FUNCTIONS[funcName]

  const { xs, ys, d1, d2, yRange } = useMemo(() => {
    const x = linspace(range[0], range[1], 241)
    const first = grad((v: Value) => sum(f(v)))
    const second = grad((v: Value) => sum(first(v)))
    const yVals = toFlat(f(x) as Tensor)
    const [lo, hi] = [Math.min(...yVals), Math.max(...yVals)]
    const pad = 0.1 * (hi - lo || 1)
    return {
      xs: toFlat(x),
      ys: yVals,
      d1: toFlat(first(x) as Tensor),
      d2: toFlat(second(x) as Tensor),
      yRange: [lo - pad, hi + pad] as [number, number],
    }
  }, [f, range])

  const at = Math.min(Math.max(x0, range[0]), range[1])
  const y0 = f(at) as number
  const p = grad(f)(at) as number
  const q = grad(grad(f))(at) as number
  const check = useMemo(() => gradCheck(f, at), [f, at])

  // Curvature: κ = f″ / (1 + (f′)²)^(3/2)
  const lift = 1 + p * p
  const kappa = q / lift ** 1.5
  const isFlat = Math.abs(kappa) < FLAT_THRESHOLD
  const radius = 1 / Math.max(Math.abs(kappa), 1e-6)
  const centre: [number, number] = [at - (p * lift) / q, y0 + lift / q]

  const span = (range[1] - range[0]) * 0.3
  const t = Array.from({ length: CIRCLE_POINTS + 1 }, (_, i) => (2 * Math.PI * i) / CIRCLE_POINTS)

  const xAx = useAxis({ label: 'x', range, nice: false })
  const yTopAx = useAxis({ label: 'f(x)', range: yRange, equal: equalUnits ? xAx : undefined, nice: false })
  const yBottomAx = useAxis({ label: 'Derivatives', nice: true })

  return (
    <Figure
      title="Higher-Order Autodiff: Tangents, Curvature, and Osculating Circles"
      purpose="grad(f) computes the first derivative f′, and nesting grad(grad(f)) computes the second derivative f″. Together they determine the tangent slope and the osculating circle of curvature 1/|κ|."
      defaultSize="L"
      controls={
        <div className="flex flex-col gap-3">
          <ControlGroup title="1 · Smooth Function Selection">
            <Select
              label="Function f(x)"
              value={funcName}
              onChange={(v) => {
                const next = v as FunctionName
                setFuncName(next)
                setX0((FUNCTIONS[next].range[0] + FUNCTIONS[next].range[1]) / 2)
              }}
              options={FUNCTION_OPTIONS}
            />
            <NumberSelector
              label="Evaluation Coordinate x₀"
              value={at}
              onChange={setX0}
              min={range[0]}
              max={range[1]}
              step={0.1}
              suggestions={[-2, 0, 1.5, 2]}
            />
            <Switch label="Equal Unit Aspect Ratio" checked={equalUnits} onChange={setEqualUnits} />
          </ControlGroup>
        </div>
      }
      readouts={{
        'Autodiff Derivative Values at x₀': (
          <>
            <Readout label="x₀" value={fmt(at)} />
            <Readout label="f(x₀)" value={fmt(y0)} />
            <Readout label="f′(x₀) = grad(f)" value={fmt(p)} />
            <Readout label="f″(x₀) = grad(grad(f))" value={fmt(q)} />
          </>
        ),
        'Differential Geometry (Curvature & Radius)': (
          <>
            <Readout label="Curvature κ" value={isFlat ? '≈ 0 (Inflection)' : fmt(kappa)} />
            <Readout label="Osculating Radius R = 1/|κ|" value={isFlat ? '∞' : fmt(radius)} />
            <Readout label="gradCheck Relative Error" value={fmt(check.maxRelError, 2)} />
          </>
        ),
      }}
      caption="Drag the handle at x₀ along the curve. The osculating circle of radius R = 1/|κ| matches f in value, tangent slope, and second-order curvature. At inflection points where f″ ≈ 0, curvature vanishes and radius diverges to infinity. The lower plot displays f′ and f″ computed purely via algorithmic differentiation."
    >
      <Plots rows={2} heights={[2, 1]} hoverGroup>
        <Plot x={xAx} y={yTopAx} title="f(x) with Tangent Line and Osculating Circle">
          <Curve name={funcName} x={xs} y={ys} slot={0} width={2} />
          <Curve
            name="Tangent"
            x={[at - span, at + span]}
            y={[y0 - p * span, y0 + p * span]}
            emphasis
            width={1.5}
          />
          {!isFlat && (
            <Curve
              name="Osculating Circle"
              x={t.map((a) => centre[0] + radius * Math.cos(a))}
              y={t.map((a) => centre[1] + radius * Math.sin(a))}
              slot={3}
              width={1.5}
            />
          )}
          {!isFlat && (
            <Curve
              name="Radius Vector"
              x={[at, centre[0]]}
              y={[y0, centre[1]]}
              slot={3}
              dashed
              width={1}
            />
          )}
          <Points name="Center of Curvature" x={isFlat ? [] : [centre[0]]} y={isFlat ? [] : [centre[1]]} slot={3} size={8} />
          <Points name="Point (x₀, f(x₀))" x={[at]} y={[y0]} emphasis size={11} />
          <Handle kind="x" at={at} onDrag={setX0} label="x₀" />
        </Plot>
        <Plot x={xAx} y={yBottomAx} title="First & Second Autodiff Derivatives">
          <Curve name="f′ = grad(f)" x={xs} y={d1} slot={1} width={2} />
          <Curve name="f″ = grad(grad(f))" x={xs} y={d2} slot={2} dashed width={2} />
          <Points name="Current f′" x={[at]} y={[p]} slot={1} size={9} />
          <Points name="Current f″" x={[at]} y={[q]} slot={2} size={9} />
        </Plot>
      </Plots>
    </Figure>
  )
}
