/**
 * Products: batched matrix multiplication, dot and outer products, and a small `einsum`. Each is a primitive (or a
 * composition of primitives) with its derivative rule: for C = AB the cotangents are Ḡ Bᵀ and Aᵀ Ḡ (Giles, 2008,
 * "Collected matrix derivative results for forward and reverse mode algorithmic differentiation", §2.2).
 */

import { AifnError, NotDifferentiableError, ShapeError } from 'aifn/foundation/errors'
import { allocate, fromData, isTensor, promote, showShape, sizeOf, type Tensor } from './core'
import { mul } from './elementwise'
import { matmulKernel } from './kernels'
import { defineOp, sumLike, type Op, type Raw, type Result2 } from './primitive'
import { sum } from './reduce'
import { broadcastTo, expandDims, permute, reshape, shapeOfValue, squeeze } from './structure'
import { isTraced, type Value } from './tape'

/** Swap the last two axes. */
function swapLast(x: Value): Value {
  const rank = shapeOfValue(x).length
  const axes = Array.from({ length: rank }, (_, k) => k)
  ;[axes[rank - 2], axes[rank - 1]] = [axes[rank - 1], axes[rank - 2]]
  return permute(x, axes)
}

function tensorInput(x: Raw, where: string): Tensor {
  if (!isTensor(x)) throw new AifnError(where, `${where}: expected a tensor, got a number`)
  return x
}

const matmulOp: Op<undefined> = defineOp<undefined>(
  'foundation/tensor/matmul',
  ([a, b]) => matmulKernel(tensorInput(a, 'matmul'), tensorInput(b, 'matmul')),
  (g, [a, b]) => [sumLike(matmulOp([g, swapLast(b)], undefined), a), sumLike(matmulOp([swapLast(a), g], undefined), b)],
  {
    arity: 2,
    doc: { summary: 'Batched matrix product with broadcast batch axes.' },
    test: {
      secondOrder: true,
      cases: (draw) => [{ inputs: [draw([2, 3]), draw([3, 4])] }, { inputs: [draw([2, 2, 3]), draw([3, 2])] }],
    },
  },
)

/**
 * Matrix product with NumPy's `matmul` rules. Rank-2 operands multiply as matrices ([m, k] × [k, n] → [m, n]). Higher
 * ranks are stacks of matrices whose leading (batch) axes broadcast. A rank-1 left operand is a row vector and a
 * rank-1 right operand a column vector, and the added axis is removed from the result (vector · vector gives a rank-0
 * tensor; use `dot` for a number).
 */
export function matmul<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, Tensor> {
  const ra = shapeOfValue(a).length
  const rb = shapeOfValue(b).length
  if (ra === 0 || rb === 0) throw new ShapeError('matmul', 'matmul: operands must have rank ≥ 1 (use mul for scalars)')
  const left = ra === 1 ? expandDims(a, 0) : a
  const right = rb === 1 ? expandDims(b, 1) : b
  let out = matmulOp([left, right], undefined)
  if (rb === 1) out = squeeze(out, -1)
  if (ra === 1) out = squeeze(out, -2 + (rb === 1 ? 1 : 0))
  return out as Result2<A, B, Tensor>
}

/** The dot product Σᵢ aᵢbᵢ of two vectors of equal length, as a number. */
export function dot<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, number> {
  const sa = shapeOfValue(a)
  const sb = shapeOfValue(b)
  if (sa.length !== 1 || sb.length !== 1 || sa[0] !== sb[0]) {
    throw new ShapeError(
      'dot',
      `dot: expected two vectors of equal length, got ${showShape(sa)} and ${showShape(sb)}`,
      [sa, sb],
    )
  }
  return sum(mul(a, b)) as Result2<A, B, number>
}

/** The outer product abᵀ of two vectors (other ranks are flattened first), shape [len a, len b]. */
export function outer<A extends Value, B extends Value>(a: A, b: B): Result2<A, B, Tensor> {
  return mul(reshape(a, [-1, 1]), reshape(b, [1, -1])) as Result2<A, B, Tensor>
}

// ── einsum ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A parsed einsum specification: the labels of each operand and of the output. */
type EinsumSpec = { inputs: string[]; output: string }

/** Parse "ij,jk->ik" (explicit) or "ij,jk" (implicit: labels used once, in alphabetical order). */
function parseEinsum(spec: string, count: number): EinsumSpec {
  const clean = spec.replace(/\s+/g, '')
  if (clean.includes('.')) throw new AifnError('einsum', `einsum: ellipsis ("...") is not supported in "${spec}"`)
  const [lhs, rhs] = clean.split('->')
  if (clean.split('->').length > 2) throw new AifnError('einsum', `einsum: more than one "->" in "${spec}"`)
  const inputs = lhs.split(',')
  if (inputs.length !== count)
    throw new ShapeError('einsum', `einsum: "${spec}" names ${inputs.length} operands, got ${count}`)
  for (const labels of [...inputs, rhs ?? '']) {
    if (!/^[a-zA-Z]*$/.test(labels))
      throw new AifnError('einsum', `einsum: labels must be letters, got "${labels}" in "${spec}"`)
  }
  let output = rhs
  if (output === undefined) {
    const counts = new Map<string, number>()
    for (const c of inputs.join('')) counts.set(c, (counts.get(c) ?? 0) + 1)
    output = [...counts.keys()]
      .filter((c) => counts.get(c) === 1)
      .sort()
      .join('')
  }
  if (new Set(output).size !== output.length)
    throw new AifnError('einsum', `einsum: output "${output}" repeats a label`)
  for (const c of output) {
    if (!inputs.some((labels) => labels.includes(c)))
      throw new ShapeError('einsum', `einsum: output label "${c}" is in no operand`)
  }
  return { inputs, output }
}

/** Forward einsum: a loop over every combination of labels, output labels outermost. */
function einsumKernel(spec: EinsumSpec, ts: Tensor[]): Tensor {
  const sizes = new Map<string, number>()
  spec.inputs.forEach((labels, i) => {
    const t = ts[i]
    if (labels.length !== t.shape.length) {
      throw new ShapeError('einsum', `einsum: operand ${i} has rank ${t.shape.length} but labels "${labels}"`)
    }
    ;[...labels].forEach((c, k) => {
      const d = t.shape[k]
      const known = sizes.get(c)
      if (known !== undefined && known !== d)
        throw new ShapeError('einsum', `einsum: label "${c}" has lengths ${known} and ${d}`)
      sizes.set(c, d)
    })
  })
  const summed = [...sizes.keys()].filter((c) => !spec.output.includes(c))
  const order = [...spec.output, ...summed]
  const shape = order.map((c) => sizes.get(c)!)
  // Each operand's stride per label; a label repeated within an operand (a diagonal) adds its strides.
  const strides = ts.map((t, i) =>
    order.map((c) => [...spec.inputs[i]].reduce((s, l, k) => (l === c ? s + t.strides[k] : s), 0)),
  )
  const outShape = [...spec.output].map((c) => sizes.get(c)!)
  const out = allocate(ts.map((t) => t.dtype).reduce(promote), sizeOf(outShape))
  const inner = sizeOf(summed.map((c) => sizes.get(c)!))
  const index = new Array<number>(order.length).fill(0)
  const offsets = ts.map((t) => t.offset)
  const total = sizeOf(shape)
  // Walk all label combinations in row-major order of `order`; each run of `inner` steps accumulates one output.
  for (let k = 0; k < total; k++) {
    let term = 1
    for (let i = 0; i < ts.length; i++) term *= ts[i].data[offsets[i]]
    out[Math.floor(k / inner)] += term
    for (let a = order.length - 1; a >= 0; a--) {
      index[a]++
      for (let i = 0; i < ts.length; i++) offsets[i] += strides[i][a]
      if (index[a] < shape[a]) break
      for (let i = 0; i < ts.length; i++) offsets[i] -= strides[i][a] * shape[a]
      index[a] = 0
    }
  }
  return fromData(out, outShape)
}

const einsumOp: Op<EinsumSpec> = defineOp<EinsumSpec>(
  'foundation/tensor/einsum',
  (xs, spec) =>
    einsumKernel(
      spec,
      xs.map((x) => tensorInput(x, 'einsum')),
    ),
  (g, xs, _y, spec) =>
    xs.map((x, i) => {
      if (!isTraced(x)) return null
      const labels = spec.inputs[i]
      if (new Set(labels).size !== labels.length) {
        throw new NotDifferentiableError(
          'einsum',
          `einsum: no derivative for operand "${labels}", which repeats a label (a diagonal)`,
        )
      }
      // The cotangent of operand i contracts the output cotangent with the other operands. Labels that occur only in
      // operand i were summed away; its cotangent is constant along them, so broadcast back.
      const others = spec.inputs.filter((_, j) => j !== i)
      const reached = [...labels].filter((c) => spec.output.includes(c) || others.some((o) => o.includes(c))).join('')
      const operands = [g, ...xs.filter((_, j) => j !== i)]
      const contracted: Value = einsumOp(operands, { inputs: [spec.output, ...others], output: reached })
      if (reached.length === labels.length) return contracted
      const shape = shapeOfValue(x)
      const partial = [...labels].map((c, k) => (reached.includes(c) ? shape[k] : 1))
      return broadcastTo(reshape(contracted, partial), shape)
    }),
  {
    doc: { summary: 'Einstein summation over letter-labelled axes.' },
    test: {
      secondOrder: true,
      cases: (draw) => [
        { inputs: [draw([2, 3]), draw([3, 4])], params: { inputs: ['ij', 'jk'], output: 'ik' } },
        { inputs: [draw([2, 3])], params: { inputs: ['ij'], output: 'ji' } },
        { inputs: [draw([3]), draw([3])], params: { inputs: ['i', 'i'], output: '' } },
      ],
    },
  },
)

/**
 * Einstein summation over letter-labelled axes, e.g. `einsum('ij,jk->ik', a, b)` (matrix product),
 * `einsum('ij->ji', a)`, `einsum('i,i->', a, b)`, `einsum('bij,bjk->bik', a, b)`, `einsum('ii->', a)` (trace) or
 * `einsum('ii->i', a)` (diagonal). Without `->` the output is the labels used once, in alphabetical order. Labels
 * must have equal lengths wherever they occur (no broadcasting) and ellipses are not supported. The loop runs over
 * every label combination, so it suits small contractions; use `matmul` for large products.
 */
export function einsum(spec: string, ...operands: Tensor[]): Tensor
export function einsum(spec: string, ...operands: Value[]): Value
export function einsum(spec: string, ...operands: Value[]): Value {
  const parsed = parseEinsum(spec, operands.length)
  return einsumOp(operands, parsed)
}
