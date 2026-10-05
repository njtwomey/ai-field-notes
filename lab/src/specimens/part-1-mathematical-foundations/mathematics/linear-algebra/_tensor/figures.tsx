import { useMemo } from 'react'
import {
  add,
  arange,
  cos,
  elementwise,
  einsum,
  exp,
  linspace,
  logsumexp,
  matmul,
  max,
  mul,
  permute,
  reshape,
  sin,
  slice,
  sub,
  sum,
  toFlat,
  transpose,
  type Tensor,
  type Value,
} from 'aifn-compute/foundation/tensor'
import { grad } from 'aifn-compute/foundation/autodiff'
import { sigmoid } from 'aifn-compute/numerics/special'
import { Equation, Figure, live, tex } from 'aifn-render/layout'
import { choice, slider, useFigureState } from 'aifn-render/state'
import { Bars, Curve, Plot, Readout, useAxis } from 'aifn-render/viz'
import { formatValue, TensorModePanel } from '@lab/views'

/** Broadcasting: a column [m, 1] and a row [1, n] combine into an m×n grid. */
export function BroadcastSpecimen() {
  const state = useFigureState({
    m: slider(2, 30, 12, { step: 1, label: 'rows m' }),
    n: slider(2, 40, 20, { step: 1, label: 'columns n' }),
  })
  const { m, n } = state
  const grid = useMemo(() => {
    const y = reshape(linspace(-1, 1, m), [m, 1])
    const x = reshape(linspace(-2, 2, n), [1, n])
    return mul(sin(mul(3, x)), cos(mul(2, y))) as Tensor
  }, [m, n])
  return (
    <Figure
      title="sin(3x) cos(2y)"
      purpose="A column of shape [m, 1] times a row of shape [1, n] is an [m, n] grid: each operand is repeated along its size-1 axis."
      description="sin(3x) · cos(2y) with x of shape [1, n] on [−2, 2] and y of shape [m, 1] on [−1, 1]."
      state={state}
      readouts={<Readout label="shape" value={`[${m}, 1] ⊙ [1, ${n}] → [${grid.shape.join(', ')}]`} />}
    >
      <TensorModePanel tensor={grid} />
    </Figure>
  )
}

const VIEWS = [
  { value: 'base', label: 'base' },
  { value: 'slice', label: '[1:5, ::2]' },
  { value: 'transpose', label: 'transpose' },
  { value: 'reversed', label: '[::-1, ::-1]' },
  { value: 'rank3', label: 'reshape [2,3,8], permute' },
] as const

/** Views share data: slicing and permuting change only shape, strides and offset. */
export function ViewsSpecimen() {
  const state = useFigureState({ view: choice(VIEWS, 'slice', { label: 'view' }) })
  const which = state.view
  const base = useMemo(() => reshape(arange(48), [6, 8]), [])
  const shown = useMemo(() => {
    if (which === 'base') return base
    if (which === 'slice') return slice(base, [1, 5], [null, null, 2])
    if (which === 'transpose') return transpose(base)
    if (which === 'reversed') return slice(base, [null, null, -1], [null, null, -1])
    return permute(reshape(base, [2, 3, 8]), [1, 0, 2])
  }, [base, which])
  return (
    <Figure
      title="A view of arange(48) reshaped to [6, 8]"
      purpose="A slice, transpose or permutation is a view: the same 48 numbers read with another shape, strides and offset, nothing copied."
      state={state}
      readouts={
        <>
          <Readout label="shape" value={`[${shown.shape.join(', ')}]`} />
          <Readout label="offset" value={shown.offset} />
          <Readout label="strides" value={`[${shown.strides.join(', ')}]`} />
          <Readout label="shares data with base" value={shown.data === base.data ? 'yes' : 'no'} />
        </>
      }
      caption="Element (i, j) of a view sits at offset + i·stride₀ + j·stride₁ in the base's data. A negative stride walks backwards; the transpose swaps the strides."
    >
      <TensorModePanel tensor={shown} initialMode="table" />
    </Figure>
  )
}

const INDEX = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅', 'x₆', 'x₇']

/** logsumexp against the naive log Σ exp, which overflows once the values pass about 709. */
export function LogSumExpSpecimen() {
  const state = useFigureState({
    shift: slider(-800, 800, 750, { step: 10, label: 'shift c (values c − 3 … c + 3)' }),
  })
  const shift = state.shift
  const x = useMemo(() => add(linspace(-3, 3, 7), shift) as Tensor, [shift])
  const stable = logsumexp(x)
  const naive = Math.log(sum(exp(x)))
  const m = max(x) as number
  const bars = useMemo(() => {
    const p = toFlat(exp(sub(x, stable)) as Tensor)
    return { x: p.map((_, i) => i), y: p }
  }, [x, stable])
  const xa = useAxis({ label: 'entry', categories: INDEX })
  const ya = useAxis({ label: 'softmax', range: [0, 1] })
  return (
    <Figure
      title="softmax = exp(x − logsumexp x)"
      purpose="Subtracting the maximum before exponentiating keeps logsumexp finite where log Σ exp overflows; softmax is unchanged by the shift."
      state={state}
      equation={
        <Equation>
          {tex`\log \sum_i e^{x_i} = m + \log \sum_i e^{x_i - m} = ${live(m, { digits: 4 })} + ${live(stable - m, { digits: 4 })} = ${live(stable, { digits: 6, strong: true })}`}
        </Equation>
      }
      readouts={
        <>
          <Readout label="logsumexp" value={formatValue(stable)} />
          <Readout label="log Σ exp (naive)" value={formatValue(naive)} />
        </>
      }
      caption="With the default shift the values pass 709, exp overflows to ∞ and the naive form returns ∞; logsumexp subtracts m = max x first, so every exponent is at most 0. Move the shift: the bars never change, because softmax depends only on the differences x_i − x_j."
    >
      <Plot x={xa} y={ya}>
        <Bars name="softmax" x={bars.x} y={bars.y} />
      </Plot>
    </Figure>
  )
}

/** einsum and batched matmul agree; the batch axis broadcasts. */
export function EinsumSpecimen() {
  const { batched, check } = useMemo(() => {
    const a = reshape(sin(arange(2 * 4 * 3)), [2, 4, 3])
    const b = reshape(cos(arange(3 * 5)), [3, 5])
    const viaMatmul = matmul(a, b)
    const viaEinsum = einsum('bij,jk->bik', a, b)
    const diff = max(sub(viaMatmul, viaEinsum)) as number
    return { batched: viaEinsum, check: diff }
  }, [])
  return (
    <Figure
      title="einsum('bij,jk->bik', a, b)"
      purpose="einsum sums over the index j that is missing from the output; with a batch index b it equals matmul broadcasting b over the [3, 5] matrix."
      description="a has shape [2, 4, 3], b has shape [3, 5]; the result has shape [2, 4, 5], one [4, 5] matrix per batch entry."
      readouts={<Readout label="max |matmul − einsum|" value={formatValue(check)} />}
    >
      <TensorModePanel tensor={batched} />
    </Figure>
  )
}

// One scalar rule and one derivative written with primitives, so softplus is differentiable to any order.
const softplus = elementwise({
  id: 'lab/softplus',
  f: (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))),
  derivative: [(x) => sigmoid(x)],
})
const dSoftplus = grad((x: Value) => softplus(x))

/** A primitive defined once works on numbers and tensors alike. */
export function PrimitiveSpecimen() {
  const x = useMemo(() => linspace(-6, 6, 241), [])
  const y = useMemo(() => toFlat(softplus(x)), [x])
  const xs = useMemo(() => toFlat(x), [x])
  const dy = useMemo(() => xs.map((v) => dSoftplus(v) as number), [xs])
  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'value' })
  return (
    <Figure
      title="softplus on a tensor"
      purpose="One scalar rule and its derivative define softplus once: it maps a number to a number, a tensor to a tensor, and grad differentiates it."
      readouts={
        <>
          <Readout label="softplus(0) (a number)" value={formatValue(softplus(0))} />
          <Readout label="softplus(tensor) shape" value={`[${softplus(x).shape.join(', ')}]`} />
          <Readout label="grad(softplus)(0)" value={formatValue(dSoftplus(0) as number)} />
        </>
      }
      caption="The curve is softplus applied once to a tensor of 241 points; its derivative is grad of the same function, evaluated point by point, which is the declared derivative σ(x)."
    >
      <Plot x={xa} y={ya}>
        <Curve name="softplus(x)" x={xs} y={y} />
        <Curve name="grad softplus = σ(x)" x={xs} y={dy} dashed />
      </Plot>
    </Figure>
  )
}
