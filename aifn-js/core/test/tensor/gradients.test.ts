import { describe, expect, it } from 'vitest'
import {
  abs,
  add,
  broadcastTo,
  clip,
  concat,
  cos,
  defineBinary,
  defineOp,
  defineTernary,
  defineUnary,
  diag,
  diagonal,
  div,
  dot,
  einsum,
  exp,
  expandDims,
  expm1,
  flatten,
  get,
  greater,
  log,
  log1p,
  logsumexp,
  map,
  matmul,
  max,
  maximum,
  mean,
  min,
  minimum,
  mul,
  neg,
  norm,
  outer,
  permute,
  pow,
  prod,
  reshape,
  set,
  sign,
  sin,
  slice,
  sqrt,
  square,
  squeeze,
  stack,
  std,
  sub,
  sum,
  sumLike,
  sumTo,
  tanh,
  tensor,
  transpose,
  unwrap,
  variance,
  where,
  type Raw,
  type Tensor,
  type Traced,
  type Value,
} from 'aifn/foundation/tensor'
import { MockTape, checkGradient } from './mock-tape'

/** Deterministic test data of a shape, in (lo, hi), with no ties. */
function data(shape: number[], lo = -1.5, hi = 1.5, seed = 1): Tensor {
  const n = shape.reduce((a, b) => a * b, 1)
  return tensor(
    Array.from(
      { length: n },
      (_, k) => lo + (((hi - lo) * (((Math.sin(12.9898 * (k + seed)) * 43758.5453) % 1) + 1)) % 1),
    ),
    shape,
  )
}

const X = data([3, 4])
const P = data([3, 4], 0.2, 2)

describe('unary vjps match finite differences', () => {
  const cases: [string, (x: Value) => Value, Raw][] = [
    ['neg', neg, X],
    ['abs', abs, X],
    ['square', square, X],
    ['exp', exp, X],
    ['expm1', expm1, X],
    ['log', log, P],
    ['log1p', log1p, P],
    ['sqrt', sqrt, P],
    ['sin', sin, X],
    ['cos', cos, X],
    ['tanh', tanh, X],
    ['exp of a number', exp, 0.7],
  ]
  it.each(cases)('%s', (_name, f, x) => {
    checkGradient(f, [x])
  })
  it('sign has a zero cotangent', () => {
    const g = checkGradient((x) => sign(x), [X])
    expect(g[0].every((v) => v === 0)).toBe(true)
  })
})

describe('binary vjps match finite differences, with broadcasting', () => {
  const a = data([3, 1], -1, 1, 2)
  const b = data([1, 4], 0.5, 1.5, 3)
  const cases: [string, (x: Value, y: Value) => Value][] = [
    ['add', add],
    ['sub', sub],
    ['mul', mul],
    ['div', div],
    ['maximum', maximum],
    ['minimum', minimum],
  ]
  it.each(cases)('%s', (_name, f) => {
    checkGradient(f, [a, b])
    checkGradient(f, [X, 0.3])
    checkGradient(f, [0.3, X])
    checkGradient(f, [0.4, 0.9])
  })
  it('pow in base and exponent', () => {
    checkGradient(pow, [data([3, 1], 0.3, 2, 4), b])
    checkGradient((x) => pow(x, 3), [X])
  })
  it('where routes cotangents by the condition', () => {
    checkGradient((x, y) => where(greater(X, 0), x, y), [X, data([4], -1, 1, 5)])
  })
  it('clip', () => checkGradient((x) => clip(x, -0.5, 0.8), [X]))
})

describe('structural vjps match finite differences', () => {
  const T = data([2, 3, 4])
  const cases: [string, (x: Value) => Value][] = [
    ['reshape', (x) => reshape(x, [4, -1])],
    ['flatten', (x) => flatten(x)],
    ['permute', (x) => permute(x, [2, 0, 1])],
    ['transpose then reshape (a copy)', (x) => reshape(transpose(x), [6, 4])],
    ['broadcastTo', (x) => broadcastTo(slice(x, null, [0, 1]), [5, 2, 3, 4])],
    ['sumTo', (x) => sumTo(x, [3, 1])],
    ['squeeze and expandDims', (x) => squeeze(expandDims(x, 1), 1)],
    ['slice with steps', (x) => slice(x, [null, null, -1], [0, 3, 2], [1])],
    ['integer slice', (x) => slice(x, 1, -1)],
    ['get', (x) => get(x, 1, 2, -1)],
    ['set', (x) => set(x, [0, 1, 2], mul(get(x, 1, 1, 1), 3))],
    ['concat', (x) => concat([x, slice(x, null, [1])], 1)],
    ['stack', (x) => stack([slice(x, 0), slice(x, 1)], 2)],
  ]
  it.each(cases)('%s', (_name, f) => {
    checkGradient(f, [T])
  })
  it('diagonal and diag', () => {
    checkGradient((x) => diagonal(x), [X])
    checkGradient((x) => diagonal(transpose(x)), [X])
    checkGradient((x) => diag(x), [data([3])])
  })
})

describe('reduction vjps match finite differences', () => {
  const T = data([2, 3, 4])
  const axes = [undefined, 0, 1, -1, [0, 2]] as const
  it.each(axes)('axis %s', (axis) => {
    const ax = axis as number | number[] | undefined
    checkGradient((x) => sum(x, ax), [T])
    checkGradient((x) => mean(x, ax, true), [T])
    checkGradient((x) => max(x, ax), [T])
    checkGradient((x) => min(x, ax), [T])
    checkGradient((x) => prod(x, ax), [T])
    checkGradient((x) => logsumexp(x, ax), [T])
    checkGradient((x) => variance(x, ax, false, 1), [T])
    checkGradient((x) => std(x, ax), [T])
  })
  it('norms of each order', () => {
    for (const ord of [1, 2, 3, Infinity, -Infinity]) checkGradient((x) => norm(x, 1, false, ord), [T])
  })
  it('prod with zeros gives the product of the others', () => {
    const z = tensor([
      [2, 0, 3],
      [0, 0, 5],
      [1, 2, 4],
    ])
    const [g] = checkGradient((x) => prod(x, 1), [z])
    const w = [0.3 + 0.7 * Math.sin(0.4), 0.3 + 0.7 * Math.sin(2.1), 0.3 + 0.7 * Math.sin(3.8)]
    expect(g[1]).toBeCloseTo(6 * w[0], 12)
    expect(g[3]).toBe(0)
  })
  it('max splits the cotangent equally among ties', () => {
    const tape = new MockTape()
    const x = tape.variable(tensor([1, 3, 3, 2]))
    const [g] = tape.gradient(max(x) as Traced, [x])
    expect(Array.from((unwrap(g) as Tensor).data)).toEqual([0, 0.5, 0.5, 0])
  })
})

describe('product vjps match finite differences', () => {
  it('matmul: matrices, batches with broadcasting, and vectors', () => {
    checkGradient(matmul, [data([2, 3]), data([3, 4], -1, 1, 7)])
    checkGradient(matmul, [data([2, 1, 3, 4]), data([5, 4, 2], -1, 1, 8)])
    checkGradient(matmul, [data([3]), data([3, 4], -1, 1, 9)])
    checkGradient(matmul, [data([4, 2, 3]), data([3], -1, 1, 10)])
    checkGradient(matmul, [data([3]), data([3], -1, 1, 11)])
  })
  it('dot and outer', () => {
    checkGradient(dot, [data([4]), data([4], -1, 1, 12)])
    checkGradient(outer, [data([3]), data([4], -1, 1, 13)])
  })
  it.each([
    [
      'ij,jk->ik',
      [
        [2, 3],
        [3, 4],
      ],
    ],
    ['ij->ji', [[2, 3]]],
    ['i,i->', [[3], [3]]],
    ['ij->i', [[2, 3]]],
    [
      'bij,bjk->bik',
      [
        [2, 3, 4],
        [2, 4, 2],
      ],
    ],
    [
      'ij,jk,kl->il',
      [
        [2, 3],
        [3, 4],
        [4, 2],
      ],
    ],
    ['ij,k->ijk', [[2, 3], [2]]],
    ['ijk->', [[2, 3, 2]]],
  ] as [string, number[][]][])('einsum %s', (spec, shapes) => {
    checkGradient(
      (...xs) => einsum(spec, ...xs),
      shapes.map((s, i) => data(s, -1, 1, 20 + i)),
    )
  })
  it('einsum refuses to differentiate an operand with a repeated label', () => {
    expect(() => checkGradient((x) => einsum('ii->', x), [data([3, 3])])).toThrow(/repeats a label/)
  })
})

describe('higher derivatives', () => {
  it('the second derivative of exp(sin x) is traced through the vjps', () => {
    const x0 = 0.7
    const tape = new MockTape()
    const x = tape.variable(x0)
    const y = exp(sin(x)) as Traced
    const [dy] = tape.gradient(y, [x])
    expect(unwrap(dy)).toBeCloseTo(Math.cos(x0) * Math.exp(Math.sin(x0)), 14)
    const [d2y] = tape.gradient(dy as Traced, [x])
    const exact = Math.exp(Math.sin(x0)) * (Math.cos(x0) ** 2 - Math.sin(x0))
    expect(unwrap(d2y)).toBeCloseTo(exact, 14)
  })
  it('the Hessian of a matrix expression (logsumexp of Ax) column by column', () => {
    const A = data([3, 2], -1, 1, 30)
    const x0 = tensor([0.2, -0.4])
    const tape = new MockTape()
    const x = tape.variable(x0)
    const [g] = tape.gradient(logsumexp(matmul(A, x)) as Traced, [x])
    // H = Aᵀ (diag(p) − p pᵀ) A with p = softmax(Ax).
    const z = Array.from((matmul(A, x0) as Tensor).data)
    const m = Math.max(...z)
    const e = z.map((v) => Math.exp(v - m))
    const p = e.map((v) => v / e.reduce((s, u) => s + u, 0))
    const a = (i: number, j: number) => A.data[i * 2 + j]
    for (let r = 0; r < 2; r++) {
      const [row] = tape.gradient(get(g, r) as Traced, [x])
      for (let c = 0; c < 2; c++) {
        let h = 0
        for (let i = 0; i < 3; i++)
          for (let j = 0; j < 3; j++) h += a(i, r) * ((i === j ? p[i] : 0) - p[i] * p[j]) * a(j, c)
        expect((unwrap(row) as Tensor).data[c]).toBeCloseTo(h, 12)
      }
    }
  })
})

describe('defineUnary, defineBinary and defineOp', () => {
  const softplus = defineUnary(
    'softplus',
    (x) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))),
    (x) => 1 / (1 + Math.exp(-x)),
  )
  const hypot = defineBinary(
    'hypot',
    Math.hypot,
    (a, _b, y) => a / y,
    (_a, b, y) => b / y,
  )
  it('return numbers for numbers and tensors for tensors', () => {
    expect(softplus(0)).toBeCloseTo(Math.LN2, 15)
    expect(typeof softplus(0)).toBe('number')
    expect(softplus(tensor([0, 1])).shape).toEqual([2])
    expect(hypot(3, 4)).toBe(5)
    expect(Array.from(hypot(tensor([3, 5]), tensor([[4], [12]])).data)).toEqual([
      5,
      Math.hypot(5, 4),
      Math.hypot(3, 12),
      13,
    ])
  })
  it('have vjps matching finite differences', () => {
    checkGradient(softplus, [X])
    checkGradient(hypot, [data([3, 1], 0.5, 2, 3), data([1, 4], 0.5, 2, 4)])
  })
  it('a second derivative needs `options.derivative`; without it, it is an error, not a zero', () => {
    const tape = new MockTape()
    const x = tape.variable(0.3)
    const [d1] = tape.gradient(softplus(x), [x])
    expect(() => tape.gradient(d1 as Traced, [x])).toThrow(/no derivative: softplus′/)
    const sigmoid = defineUnary(
      'sigmoid',
      (v) => 1 / (1 + Math.exp(-v)),
      (_v, y) => y * (1 - y),
      { derivative: (_v, y) => mul(y, sub(1, y)) },
    )
    const smooth = defineUnary(
      'softplus',
      (v) => Math.log1p(Math.exp(v)),
      (v) => 1 / (1 + Math.exp(-v)),
      {
        derivative: (v) => sigmoid(v),
      },
    )
    const t2 = new MockTape()
    const z = t2.variable(0.3)
    const [e1] = t2.gradient(smooth(z), [z])
    const [e2] = t2.gradient(e1 as Traced, [z])
    const s = 1 / (1 + Math.exp(-0.3))
    expect(unwrap(e2)).toBeCloseTo(s * (1 - s), 14)
  })
  it('a null partial throws when that argument is traced, and is ignored when it is constant', () => {
    const scaled = defineBinary(
      'scaled',
      (a, k) => a * k,
      (_a, k) => k,
      null,
    )
    checkGradient((a) => scaled(a, 2), [X])
    expect(() => checkGradient((k) => scaled(X, k), [0.5])).toThrow(/no derivative with respect to argument 2/)
  })
  it('an op without a derivative is an error to differentiate through', () => {
    expect(() => checkGradient((x) => map(x, Math.floor), [X])).toThrow(/no derivative: map/)
  })
  it('defineOp: a general primitive with parameters', () => {
    const scaleRows = defineOp<number>(
      'scaleRows',
      ([m], k) => mul(m as Tensor, k),
      (g, _inputs, _y, k) => [mul(g, k)],
    )
    expect(Array.from((scaleRows([tensor([1, 2])], 3) as Tensor).data)).toEqual([3, 6])
    checkGradient((x) => scaleRows([x], 2.5), [X])
  })
})

describe('defineBinary options.derivative, defineTernary and sumLike', () => {
  const hypot = defineBinary(
    'hypot',
    Math.hypot,
    (a, _b, y) => a / y,
    (_a, b, y) => b / y,
    { derivative: [(a, _b, y) => div(a, y), (_a, b, y) => div(b, y)] },
  )
  it('defineBinary with options.derivative has correct first and second derivatives', () => {
    checkGradient(hypot, [data([3, 1], 0.5, 2, 3), data([1, 4], 0.5, 2, 4)])
    // ∂²/∂a² hypot(a, b) = b²/y³ and ∂²/∂a∂b = −ab/y³.
    const tape = new MockTape()
    const a = tape.variable(3)
    const b = tape.variable(4)
    const [ga] = tape.gradient(hypot(a, b) as Traced, [a])
    const [gaa, gab] = tape.gradient(ga as Traced, [a, b])
    expect(unwrap(gaa)).toBeCloseTo(16 / 125, 14)
    expect(unwrap(gab)).toBeCloseTo(-12 / 125, 14)
  })
  it('defineBinary without options.derivative: a second derivative is an error', () => {
    const plain = defineBinary(
      'hypot',
      Math.hypot,
      (a, _b, y) => a / y,
      (_a, b, y) => b / y,
    )
    const tape = new MockTape()
    const a = tape.variable(3)
    const [ga] = tape.gradient(plain(a, 4) as Traced, [a])
    expect(() => tape.gradient(ga as Traced, [a])).toThrow(/no derivative: hypot′a/)
  })
  const fma = defineTernary(
    'fma',
    (a, b, c) => a * b + c * c,
    (_a, b) => b,
    (a) => a,
    (_a, _b, c) => 2 * c,
    { derivative: [(_a, b) => b, (a) => a, (_a, _b, c) => mul(2, c)] },
  )
  it('defineTernary broadcasts numbers, tensors and mixed shapes', () => {
    expect(fma(2, 3, 1)).toBe(7)
    expect(typeof fma(2, 3, 1)).toBe('number')
    const y = fma(tensor([1, 2]), tensor([[1], [10]]), 1)
    expect(y.shape).toEqual([2, 2])
    expect(Array.from(y.data)).toEqual([2, 3, 11, 21])
    expect(fma(tensor([1, 2], [2], 'int32'), 3, 0).dtype).toBe('float64')
  })
  it('defineTernary vjps match finite differences in every argument, with broadcasting', () => {
    checkGradient(fma, [data([3, 1], -1, 1, 2), data([1, 4], -1, 1, 5), data([4], -1, 1, 6)])
    checkGradient((a, b, c) => fma(a, b, c), [0.3, data([2, 2], -1, 1, 7), 1.5])
  })
  it('defineTernary: second derivatives through options.derivative, null partials throw', () => {
    const tape = new MockTape()
    const c = tape.variable(0.7)
    const [gc] = tape.gradient(fma(1, 2, c) as Traced, [c])
    const [gcc] = tape.gradient(gc as Traced, [c])
    expect(unwrap(gcc)).toBe(2)
    const onlyLast = defineTernary(
      'onlyLast',
      (a, b, c) => a + b * c,
      null,
      null,
      (_a, b) => b,
    )
    checkGradient((c) => onlyLast(1, 2, c), [X])
    expect(() => checkGradient((a) => onlyLast(a, 2, 3), [0.5])).toThrow(/no derivative with respect to argument 1/)
  })
  it('sumLike reduces a broadcast cotangent to the input shape and kind', () => {
    const g = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(sumLike(g, 0)).toBe(21)
    expect(Array.from((sumLike(g, tensor([0, 0, 0])) as Tensor).data)).toEqual([5, 7, 9])
    expect(Array.from((sumLike(g, tensor([[0], [0]])) as Tensor).data)).toEqual([6, 15])
    expect(sumLike(g, g)).toBe(g)
    expect((sumLike(2, tensor([0, 0])) as Tensor).shape).toEqual([2])
  })
})
