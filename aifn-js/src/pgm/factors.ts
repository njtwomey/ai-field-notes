/**
 * Discrete factor graphs as plain data, and the factor algebra every discrete engine builds on: products,
 * marginalisation by sum or max, conditioning on evidence and normalisation (Koller & Friedman 2009, "Probabilistic
 * Graphical Models", §4.2 and §9.3; Kschischang, Frey & Loeliger 2001, "Factor graphs and the sum-product algorithm").
 *
 * A factor's table is a float64 tensor whose axes follow its scope: `table[a₀, a₁, …]` is the potential at
 * `x[scope[0]] = a₀, x[scope[1]] = a₁, …`. Potentials are non-negative (not logs).
 */

import { connectedComponents, fromEdges, type Graph } from 'aifn/graph'
import { fromData, toFlat, type Tensor } from 'aifn/tensor'

/** A potential over a few discrete variables; `table` has shape `scope.map((v) => cardinalities[v])`. */
export interface DiscreteFactor {
  scope: readonly number[]
  table: Tensor
  /** For display, e.g. `ψ₁₂`. */
  name?: string
}

/**
 * A discrete factor graph: variables 0 … V − 1 with the given cardinalities, and factors over them. The joint is
 * p(x) = (1/Z) Πₐ fₐ(x_{scope(a)}).
 */
export interface DiscreteFactorGraph {
  cardinalities: readonly number[]
  factors: readonly DiscreteFactor[]
  /** Variable names for display (default `x0`, `x1`, …). */
  names?: readonly string[]
}

/** One edge of a factor graph: between variable `variable` and factor `factor`, at `position` in its scope. */
export interface FactorGraphEdge {
  variable: number
  factor: number
  position: number
}

/** Row-major strides of a table with these axis sizes. */
export function stridesOf(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length)
  let s = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = s
    s *= shape[i]
  }
  return strides
}

/** The number of entries of a table with these axis sizes. */
export const tableSize = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1)

/** Contiguous float64 values of a table (a copy when the tensor is a strided view). */
export const valuesOf = (t: Tensor): Float64Array =>
  t.data instanceof Float64Array && t.offset === 0 && t.data.length === tableSize(t.shape)
    ? t.data
    : Float64Array.from(toFlat(t))

/** Visit every assignment of variables with these cardinalities in row-major order. */
export function forEachAssignment(shape: readonly number[], fn: (assignment: Int32Array, flat: number) => void): void {
  const n = tableSize(shape)
  const a = new Int32Array(shape.length)
  for (let flat = 0; flat < n; flat++) {
    fn(a, flat)
    for (let i = shape.length - 1; i >= 0; i--) {
      if (++a[i] < shape[i]) break
      a[i] = 0
    }
  }
}

/**
 * Build a factor from a scope, the graph's cardinalities and its values (a tensor, a flat array in row-major order, or
 * a function of the assignment). Throws on a negative or NaN potential.
 */
export function discreteFactor(
  scope: readonly number[],
  cardinalities: readonly number[],
  values: Tensor | ArrayLike<number> | ((assignment: Int32Array) => number),
  name?: string,
): DiscreteFactor {
  const shape = scope.map((v) => cardinalities[v])
  const n = tableSize(shape)
  let data: Float64Array
  if (typeof values === 'function') {
    data = new Float64Array(n)
    forEachAssignment(shape, (a, flat) => (data[flat] = values(a)))
  } else {
    data = 'shape' in values ? Float64Array.from(toFlat(values)) : Float64Array.from(values)
  }
  if (data.length !== n) throw new RangeError(`discreteFactor: ${data.length} values for a table of ${n}`)
  for (const v of data) if (!(v >= 0)) throw new RangeError(`discreteFactor: potential ${v} is not non-negative`)
  return { scope: [...scope], table: fromData(data, shape), ...(name === undefined ? {} : { name }) }
}

/** Check a factor graph's scopes and table shapes; returns it unchanged. */
export function discreteFactorGraph(
  cardinalities: readonly number[],
  factors: readonly DiscreteFactor[],
  names?: readonly string[],
): DiscreteFactorGraph {
  factors.forEach((f, k) => {
    f.scope.forEach((v, i) => {
      if (!(v >= 0 && v < cardinalities.length)) throw new RangeError(`factor ${k}: variable ${v} is out of range`)
      if (f.table.shape[i] !== cardinalities[v])
        throw new RangeError(
          `factor ${k}: axis ${i} has size ${f.table.shape[i]}, variable ${v} has ${cardinalities[v]}`,
        )
    })
    if (new Set(f.scope).size !== f.scope.length) throw new RangeError(`factor ${k}: repeated variable in scope`)
  })
  return { cardinalities: [...cardinalities], factors, ...(names ? { names: [...names] } : {}) }
}

/** The display name of variable v. */
export const variableName = (g: DiscreteFactorGraph, v: number): string => g.names?.[v] ?? `x${v}`

/** Every (variable, factor) edge, grouped by factor in scope order. */
export function factorGraphEdges(g: DiscreteFactorGraph): FactorGraphEdge[] {
  return g.factors.flatMap((f, factor) => f.scope.map((variable, position) => ({ variable, factor, position })))
}

/**
 * The factor graph as an undirected bipartite `aifn/graph` graph: nodes 0 … V − 1 are the variables and V … V + F − 1
 * the factors; edge k is `factorGraphEdges(g)[k]`.
 */
export function bipartiteGraph(g: DiscreteFactorGraph): Graph {
  const V = g.cardinalities.length
  const labels = [...g.cardinalities.map((_, v) => variableName(g, v)), ...g.factors.map((f, k) => f.name ?? `f${k}`)]
  return fromEdges(
    V + g.factors.length,
    factorGraphEdges(g).map((e) => [e.variable, V + e.factor] as const),
    { directed: false, labels },
  )
}

/** True when the factor graph has no cycles (it is a tree or a forest), so sum-product is exact on it. */
export function isTree(g: DiscreteFactorGraph): boolean {
  const b = bipartiteGraph(g)
  return b.edges.length === b.nodes - connectedComponents(b).count
}

/** The product of two factors, over the union of their scopes (a's variables first). */
export function factorProduct(a: DiscreteFactor, b: DiscreteFactor, cardinalities: readonly number[]): DiscreteFactor {
  const scope = [...a.scope, ...b.scope.filter((v) => !a.scope.includes(v))]
  const shape = scope.map((v) => cardinalities[v])
  const av = valuesOf(a.table)
  const bv = valuesOf(b.table)
  const as = stridesOf(a.table.shape)
  const bs = stridesOf(b.table.shape)
  const aAxis = scope.map((v) => a.scope.indexOf(v))
  const bAxis = scope.map((v) => b.scope.indexOf(v))
  const out = new Float64Array(tableSize(shape))
  forEachAssignment(shape, (x, flat) => {
    let ia = 0
    let ib = 0
    for (let i = 0; i < scope.length; i++) {
      if (aAxis[i] >= 0) ia += x[i] * as[aAxis[i]]
      if (bAxis[i] >= 0) ib += x[i] * bs[bAxis[i]]
    }
    out[flat] = av[ia] * bv[ib]
  })
  return { scope, table: fromData(out, shape) }
}

/** The product of several factors; an empty list gives the constant factor 1. */
export function factorProductAll(factors: readonly DiscreteFactor[], cardinalities: readonly number[]): DiscreteFactor {
  let out: DiscreteFactor = { scope: [], table: fromData(new Float64Array([1]), []) }
  for (const f of factors) out = factorProduct(out, f, cardinalities)
  return out
}

/**
 * Sum (or maximise) the given variables out of a factor. With `mode: 'max'` the result is a max-marginal
 * (max-product).
 */
export function factorMarginalise(
  f: DiscreteFactor,
  variables: readonly number[],
  mode: 'sum' | 'max' = 'sum',
): DiscreteFactor {
  const keep = f.scope.map((v, i) => (variables.includes(v) ? -1 : i)).filter((i) => i >= 0)
  const scope = keep.map((i) => f.scope[i])
  const shape = keep.map((i) => f.table.shape[i])
  const ks = stridesOf(shape)
  const out = new Float64Array(tableSize(shape)).fill(mode === 'sum' ? 0 : -Infinity)
  const values = valuesOf(f.table)
  forEachAssignment(f.table.shape, (x, flat) => {
    let j = 0
    for (let k = 0; k < keep.length; k++) j += x[keep[k]] * ks[k]
    out[j] = mode === 'sum' ? out[j] + values[flat] : Math.max(out[j], values[flat])
  })
  return { scope, table: fromData(out, shape) }
}

/** Condition a factor on evidence (variable → observed value): the observed variables leave its scope. */
export function factorReduce(f: DiscreteFactor, evidence: ReadonlyMap<number, number>): DiscreteFactor {
  const keep = f.scope.map((v, i) => (evidence.has(v) ? -1 : i)).filter((i) => i >= 0)
  if (keep.length === f.scope.length) return f
  const strides = stridesOf(f.table.shape)
  let base = 0
  f.scope.forEach((v, i) => {
    if (evidence.has(v)) base += evidence.get(v)! * strides[i]
  })
  const shape = keep.map((i) => f.table.shape[i])
  const values = valuesOf(f.table)
  const out = new Float64Array(tableSize(shape))
  forEachAssignment(shape, (x, flat) => {
    let j = base
    for (let k = 0; k < keep.length; k++) j += x[k] * strides[keep[k]]
    out[flat] = values[j]
  })
  return { scope: keep.map((i) => f.scope[i]), table: fromData(out, shape), ...(f.name ? { name: f.name } : {}) }
}

/** A factor scaled to sum to one, with the log of the sum it had (−∞ for an all-zero factor, which stays zero). */
export function normaliseFactor(f: DiscreteFactor): { factor: DiscreteFactor; logNormaliser: number } {
  const v = valuesOf(f.table)
  let z = 0
  for (const x of v) z += x
  if (!(z > 0)) return { factor: f, logNormaliser: -Infinity }
  return {
    factor: {
      ...f,
      table: fromData(
        v.map((x) => x / z),
        f.table.shape,
      ),
    },
    logNormaliser: Math.log(z),
  }
}

/** The variables sharing a factor with v (its Markov blanket in the factor graph), ascending. */
export function factorGraphNeighbours(g: DiscreteFactorGraph, v: number): number[] {
  const out = new Set<number>()
  for (const f of g.factors) if (f.scope.includes(v)) for (const u of f.scope) if (u !== v) out.add(u)
  return [...out].sort((a, b) => a - b)
}

/** The unnormalised log-probability Σₐ log fₐ(x) of a full assignment. */
export function logPotential(g: DiscreteFactorGraph, assignment: ArrayLike<number>): number {
  let total = 0
  for (const f of g.factors) {
    const strides = stridesOf(f.table.shape)
    let j = 0
    f.scope.forEach((v, i) => (j += assignment[v] * strides[i]))
    total += Math.log(valuesOf(f.table)[j])
  }
  return total
}

/**
 * A pairwise Ising model on a graph as a factor graph: spins xᵢ ∈ {−1, +1} (index 0 is −1, index 1 is +1), with
 * p(x) ∝ exp(Σ_{(i,j)} J xᵢxⱼ + Σᵢ hᵢ xᵢ). `coupling` and `field` may be numbers or per-edge / per-node arrays.
 */
export function isingModel(
  graph: Graph,
  coupling: number | ArrayLike<number>,
  field: number | ArrayLike<number>,
): DiscreteFactorGraph {
  const J = (k: number) => (typeof coupling === 'number' ? coupling : coupling[k])
  const h = (i: number) => (typeof field === 'number' ? field : field[i])
  const cards = new Array<number>(graph.nodes).fill(2)
  const spin = (a: number) => (a === 0 ? -1 : 1)
  const factors: DiscreteFactor[] = []
  for (let i = 0; i < graph.nodes; i++)
    factors.push(discreteFactor([i], cards, (a) => Math.exp(h(i) * spin(a[0])), `φ${i}`))
  graph.edges.forEach((e, k) =>
    factors.push(
      discreteFactor([e.from, e.to], cards, (a) => Math.exp(J(k) * spin(a[0]) * spin(a[1])), `ψ${e.from},${e.to}`),
    ),
  )
  return discreteFactorGraph(cards, factors, graph.labels)
}

/** The edges of a rows × cols grid (4-neighbour), node r·cols + c; for `isingModel` with `fromEdges`. */
export function gridEdges(rows: number, cols: number): [number, number][] {
  const edges: [number, number][] = []
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      if (c + 1 < cols) edges.push([r * cols + c, r * cols + c + 1])
      if (r + 1 < rows) edges.push([r * cols + c, (r + 1) * cols + c])
    }
  return edges
}
