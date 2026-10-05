import { grad } from 'aifn-compute/foundation/autodiff'
import { softplus } from 'aifn-compute/numerics/special'
import {
  add,
  div,
  exp,
  linspace,
  mul,
  sin,
  square,
  sub,
  sum,
  toFlat,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { choice, defineVariants, slider, toggle, useParam, useVariants, VariantControls } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { Curve, formatNumber, Handle, Plot, Points, Readout, useAxis } from 'aifn-render/viz'

/**
 * The function family. Each entry is one function with its own parameters; adding a function is adding an entry.
 * `amplitude` is shared: every function has it, with one value. `f` is written once with aifn primitives, so the
 * derivative comes from `grad`.
 */
const FUNCTIONS = defineVariants(
  {
    quadratic: {
      label: 'Quadratic a x² + b x + c',
      params: { a: slider(-2, 2, 0.5), b: slider(-2, 2, 0), c: slider(-2, 2, -1) },
      f: (x: Value, p): Value => mul(p.amplitude, add(add(mul(p.a, square(x)), mul(p.b, x)), p.c)),
    },
    bump: {
      label: 'Gaussian bump',
      params: { mu: slider(-2, 2, 0, { label: 'mean μ' }), sigma: slider(0.1, 2, 0.6, { label: 'width σ' }) },
      f: (x: Value, p): Value => mul(p.amplitude, exp(div(square(sub(x, p.mu)), -2 * p.sigma ** 2))),
    },
    softplus: {
      label: 'Softplus',
      params: {
        beta: slider(0.2, 5, 1, { label: 'sharpness β' }),
        shape: choice(['plain', 'shifted']),
        // Shown only when the shape is 'shifted'.
        shift: slider(-2, 2, 1, { label: 'shift', when: (p) => p.shape === 'shifted' }),
      },
      f: (x: Value, p): Value => {
        const at = p.shape === 'shifted' ? sub(x, p.shift) : x
        return mul(p.amplitude / p.beta, softplus(mul(p.beta, at)))
      },
    },
    wave: {
      label: 'Damped wave',
      params: {
        omega: slider(0.5, 6, 3, { label: 'frequency ω' }),
        decay: slider(0, 1, 0.3, { label: 'decay' }),
        squared: toggle(false, { label: 'square it' }),
      },
      f: (x: Value, p): Value => {
        const wave = mul(sin(mul(p.omega, x)), exp(mul(-p.decay, square(x))))
        return mul(p.amplitude, p.squared ? square(wave) : wave)
      },
    },
  },
  { amplitude: slider(0.2, 2, 1, { label: 'amplitude (shared)' }) },
)

const RANGE: [number, number] = [-3, 3]

/** f and f′ = grad(f) for the chosen function, with a draggable x₀ and the tangent line there. */
export function FunctionFamilyFigure() {
  const v = useVariants(FUNCTIONS)
  const x0 = useParam(0.8, { min: RANGE[0], max: RANGE[1] })
  const f = v.f!
  const { series, y0, slope } = useMemo(() => {
    const x = linspace(RANGE[0], RANGE[1], 301)
    // For an elementwise f, the gradient of Σ f(x) is f′ at every element.
    const df = grad((u: Value) => sum(f(u)))
    const xs = toFlat(x)
    const y0 = f(x0.value) as number
    const slope = grad(f)(x0.value) as number
    const series = {
      x: xs,
      f: toFlat(f(x) as Tensor),
      df: toFlat(df(x) as Tensor),
      tangent: { x: [x0.value - 1, x0.value + 1], y: [y0 - slope, y0 + slope] },
    }
    return { series, y0, slope }
  }, [f, x0.value])
  const xAxis = useAxis({ label: 'x', range: RANGE })
  const yAxis = useAxis({ label: 'value' })
  return (
    <Figure
      title="Function family"
      // TODO(5c): purpose taken from the description
      purpose="Pick a function; the controls change to its parameters, and each function keeps its own values when you switch away and back."
      defaultSize="L"
      controls={<VariantControls variants={v} />}
      readouts={
        <>
          <Readout label="f(x₀)" value={formatNumber(y0)} />
          <Readout label="f′(x₀)" value={formatNumber(slope)} />
          <Readout label="state" value={JSON.stringify(v.state)} />
        </>
      }
      caption="f′ comes from grad in aifn/autodiff, not a hand-written derivative. Drag the dashed x₀ line to move the tangent. Choose Softplus and set its shape to 'shifted' to reveal the shift slider. The state readout is the plain JSON a URL could hold."
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="f(x)" x={series.x} y={series.f} slot={0} />
        <Curve name="f′(x) = grad(f)" x={series.x} y={series.df} slot={1} />
        <Curve name="tangent at x₀" x={series.tangent.x} y={series.tangent.y} slot={2} dashed />
        <Points name="x₀" x={[x0.value]} y={[y0]} emphasis />
        <Handle kind="x" at={x0.value} label="x₀" onDrag={x0.set} />
      </Plot>
    </Figure>
  )
}
