import { traceGraph } from 'aifn/autodiff'
import { softplus } from 'aifn/special'
import { add, exp, log, mul, square, sub, type Value } from 'aifn/tensor'
import { useMemo } from 'react'
import { choice, defineVariants, slider, useVariants, VariantControls } from '@lab/controls'
import { ComputationGraphView, formatValue } from '@lab/views'
import { Readout } from '@lab/viz'

/** The traced inputs of a function, by name; the names become the symbols in the diagram. */
type Inputs = Record<string, Value>

/** The two points the MSE line is fitted to. */
const DATA = [
  { x: 1, t: 1.5 },
  { x: 2, t: 2.2 },
]

/**
 * The functions on offer. Each lists its parameters; those named in `INPUTS` are traced (they get adjoints), the rest
 * are constants of the function.
 */
const FUNCTIONS = defineVariants({
  logistic: {
    label: 'Logistic loss of one example',
    params: {
      w: slider(-3, 3, 0.8, { label: 'weight w' }),
      x: slider(-3, 3, 1.5, { label: 'input x' }),
      b: slider(-2, 2, -0.5, { label: 'bias b' }),
      y: choice(['+1', '-1'], '+1', { label: 'label y' }),
    },
    f: (v: Inputs, p): Value => log(add(1, exp(mul(add(mul(v.w, v.x), v.b), p.y === '+1' ? -1 : 1)))),
  },
  fanout: {
    label: 'Product and sum with fan-out',
    params: { a: slider(-2, 2, 1.5), b: slider(-2, 2, -0.8) },
    f: (v: Inputs): Value => mul(add(v.a, v.b), mul(v.a, v.b)),
  },
  softplus: {
    label: 'Softplus of an affine map',
    params: {
      w: slider(-3, 3, 1.2, { label: 'weight w' }),
      x: slider(-3, 3, 0.7, { label: 'input x' }),
      b: slider(-2, 2, -0.4, { label: 'bias b' }),
    },
    f: (v: Inputs): Value => softplus(add(mul(v.w, v.x), v.b)),
  },
  mse: {
    label: 'Squared error of a line on two points',
    params: { w: slider(-2, 3, 0.5, { label: 'slope w' }), b: slider(-2, 2, 0.2, { label: 'intercept b' }) },
    f: (v: Inputs): Value =>
      mul(0.5, add(...(DATA.map((d) => square(sub(add(mul(v.w, d.x), v.b), d.t))) as [Value, Value]))),
  },
})

type Key = keyof typeof FUNCTIONS.specs

/** For each function: its traced inputs, the output's symbol and the function as TeX. */
const META: Record<Key, { inputs: string[]; output: string; tex: (p: Record<string, unknown>) => string }> = {
  logistic: {
    inputs: ['w', 'x', 'b'],
    output: 'L',
    tex: (p) => `L = \\log\\bigl(1 + e^{-y(wx + b)}\\bigr), \\quad y = ${p.y === '+1' ? '+1' : '-1'}`,
  },
  fanout: { inputs: ['a', 'b'], output: 'f', tex: () => 'f = (a + b)\\,ab' },
  softplus: {
    inputs: ['w', 'x', 'b'],
    output: 'y',
    tex: () => 'y = \\operatorname{softplus}(wx + b) = \\log(1 + e^{wx + b})',
  },
  mse: {
    inputs: ['w', 'b'],
    output: 'L',
    tex: () =>
      `L = \\tfrac12 \\sum_{i=1}^{2} (w x_i + b - t_i)^2, \\quad (x_i, t_i) \\in \\{${DATA.map((d) => `(${d.x}, ${d.t})`).join(', ')}\\}`,
  },
}

/** traceGraph on the chosen function, drawn as a two-band factor graph and stepped through forward and backward. */
export function BackpropSpecimen() {
  const v = useVariants(FUNCTIONS)
  const meta = META[v.key]
  const f = v.f!
  const point = Object.fromEntries(meta.inputs.map((name) => [name, v.values[name] as number]))
  const key = JSON.stringify(point)
  // Recomputed only when the function or a value changes; the graph is small, so sliders update it live.
  // oxlint-disable-next-line react-hooks/exhaustive-deps -- `key` stands for `point`; `f` changes with the values
  const graph = useMemo(() => traceGraph((x: Inputs) => f(x), point), [v.key, key, f])
  const grad = graph.grad as Record<string, number>
  return (
    <ComputationGraphView
      key={v.key}
      title={FUNCTIONS.specs[v.key].label}
      graph={graph}
      expression={meta.tex(v.values)}
      outputSymbol={meta.output}
      controls={<VariantControls variants={v} />}
      readouts={
        <>
          <Readout label={meta.output} value={formatValue(graph.value as number)} />
          {meta.inputs.map((name) => (
            <Readout key={name} label={`∂${meta.output}/∂${name}`} value={formatValue(grad[name])} />
          ))}
        </>
      }
      caption="Step with the player: first the forward pass computes each value from the inputs (top band, left to right), then the backward pass seeds the output's adjoint with 1 and pulls it back through each operation in reverse (bottom band, right to left). Each backward edge carries the local partial the adjoint is multiplied by; an input used twice adds the messages from both uses. Values not yet computed show as a dot. Move a slider to recompute everything live."
    />
  )
}
