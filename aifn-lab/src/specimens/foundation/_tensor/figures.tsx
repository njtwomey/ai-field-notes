import { useMemo, useState } from 'react'
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
} from 'aifn/foundation/tensor'
import { sigmoid } from 'aifn/numerics/special'
import { Select, Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Readout, XYChart } from '@lab/viz'
import { formatValue, TensorView } from '@lab/views'

/** Broadcasting: a column [m, 1] and a row [1, n] combine into an m×n grid. */
export function BroadcastSpecimen() {
  const [m, setM] = useState(12)
  const [n, setN] = useState(20)
  const grid = useMemo(() => {
    const y = reshape(linspace(-1, 1, m), [m, 1])
    const x = reshape(linspace(-2, 2, n), [1, n])
    return mul(sin(mul(3, x)), cos(mul(2, y))) as Tensor
  }, [m, n])
  return (
    <TensorView
      title="sin(3x) cos(2y)"
      description="sin(3x) · cos(2y) with x of shape [1, n] and y of shape [m, 1]."
      tensor={grid}
      controls={
        <>
          <Slider label="rows m" value={m} onChange={setM} min={2} max={30} step={1} />
          <Slider label="columns n" value={n} onChange={setN} min={2} max={40} step={1} />
        </>
      }
    />
  )
}

/** Views share data: slicing and permuting change only shape, strides and offset. */
export function ViewsSpecimen() {
  const [which, setWhich] = useState<'base' | 'slice' | 'transpose' | 'reversed' | 'rank3'>('slice')
  const base = useMemo(() => reshape(arange(48), [6, 8]), [])
  const shown = useMemo(() => {
    if (which === 'base') return base
    if (which === 'slice') return slice(base, [1, 5], [null, null, 2])
    if (which === 'transpose') return transpose(base)
    if (which === 'reversed') return slice(base, [null, null, -1], [null, null, -1])
    return permute(reshape(base, [2, 3, 8]), [1, 0, 2])
  }, [base, which])
  return (
    <TensorView
      title="A view of arange(48) reshaped to [6, 8]"
      tensor={shown}
      initialMode="table"
      controls={
        <Select
          label="view"
          value={which}
          onChange={setWhich}
          options={[
            { value: 'base', label: 'base' },
            { value: 'slice', label: '[1:5, ::2]' },
            { value: 'transpose', label: 'transpose' },
            { value: 'reversed', label: '[::-1, ::-1]' },
            { value: 'rank3', label: 'reshape [2,3,8], permute' },
          ]}
        />
      }
      readouts={
        <>
          <Readout label="offset" value={shown.offset} />
          <Readout label="strides" value={`[${shown.strides.join(', ')}]`} />
          <Readout label="shares data with base" value={shown.data === base.data ? 'yes' : 'no'} />
        </>
      }
    />
  )
}

/** logsumexp against the naive log Σ exp, which overflows once the values pass about 709. */
export function LogSumExpSpecimen() {
  const [shift, setShift] = useState(700)
  const x = useMemo(() => add(linspace(-3, 3, 7), shift) as Tensor, [shift])
  const stable = logsumexp(x)
  const naive = Math.log(sum(exp(x)))
  const bars = useMemo(() => {
    const p = toFlat(exp(sub(x, stable)) as Tensor)
    return { x: p.map((_, i) => i), y: p }
  }, [x, stable])
  return (
    <Figure
      title="softmax = exp(x − logsumexp x)"
      controls={
        <Slider
          label="shift c (values c − 3 … c + 3)"
          value={shift}
          onChange={setShift}
          min={-800}
          max={800}
          step={10}
        />
      }
      readouts={
        <>
          <Readout label="logsumexp" value={formatValue(stable)} />
          <Readout label="log Σ exp (naive)" value={formatValue(naive)} />
          <Readout label="max" value={formatValue(max(x))} />
        </>
      }
    >
      <XYChart
        xLabel="index"
        yLabel="softmax"
        integerX
        series={[{ name: 'softmax', type: 'bar', x: bars.x, y: bars.y }]}
      />
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
    <TensorView
      title="einsum('bij,jk->bik', a, b)"
      tensor={batched}
      readouts={<Readout label="max |matmul − einsum|" value={formatValue(check)} />}
    />
  )
}

// One scalar rule and one derivative written with primitives, so softplus is differentiable to any order.
const softplus = elementwise({
  id: 'lab/softplus',
  f: (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))),
  derivative: [(x) => sigmoid(x)],
})

/** A primitive defined once works on numbers and tensors alike. */
export function PrimitiveSpecimen() {
  const x = useMemo(() => linspace(-6, 6, 241), [])
  const y = useMemo(() => toFlat(softplus(x)), [x])
  const xs = useMemo(() => toFlat(x), [x])
  return (
    <Figure
      title="softplus on a tensor"
      readouts={
        <>
          <Readout label="softplus(0) (a number)" value={formatValue(softplus(0))} />
          <Readout label="softplus(tensor) shape" value={`[${softplus(x).shape.join(', ')}]`} />
        </>
      }
    >
      <XYChart xLabel="x" yLabel="softplus(x)" series={[{ name: 'softplus', type: 'line', x: xs, y }]} />
    </Figure>
  )
}
