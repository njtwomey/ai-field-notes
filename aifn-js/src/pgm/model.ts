/**
 * A small typed language for describing probabilistic models (plan §7.1): sizes, plates (nested, with fixed or
 * ragged sizes), constants, latent and observed variables whose conditional distributions come from
 * `aifn/distributions`, and deterministic nodes. The description is plain data; `expandModel` unrolls its plates
 * against sizes and data into instances, from which the factor graph, the Markov blankets, the log joint, ancestral
 * samples and the inference engines are built.
 *
 * ```ts
 * const lda = model('Latent Dirichlet allocation', (m) => {
 *   const K = m.size('K'), V = m.size('V')
 *   const topics = m.plate('topics', K), docs = m.plate('documents', 'D'), words = docs.plate('words', 'N')
 *   const alpha = m.constant('α'), beta = m.constant('β')
 *   const phi = topics.variable('φ', dist.Dirichlet(beta, V))
 *   const theta = docs.variable('θ', dist.Dirichlet(alpha, K))
 *   const z = words.variable('z', dist.Categorical(theta))
 *   words.observed('w', dist.Categorical(phi.at(z)))
 * })
 * ```
 *
 * `ref.at(selector)` indexes by the value of a discrete node: it picks one instance of a plate the referring node is
 * not in (φ_{z}), or one entry (row) of a vector- or matrix-valued node (means[z], A[y_{n−1}]).
 */

import {
  Bernoulli,
  Beta,
  Binomial,
  Categorical,
  Dirichlet,
  Gamma,
  Normal,
  Poisson,
  type Distribution,
} from 'aifn/distributions'
import type { Stream } from 'aifn/random'
import { normalCdf, sigmoid } from 'aifn/special'
import {
  add,
  exp,
  fromData,
  full,
  isTensor,
  mul,
  sub,
  sum,
  tensor,
  toArray,
  toFlat,
  type NestedArray,
  type Tensor,
} from 'aifn/tensor'

/** The value of a node instance: a number (scalars and discrete values) or a tensor (vectors). */
export type NodeValue = number | Tensor

/** Families the language supports, each realised by the `aifn/distributions` constructor of that name. */
export type Family = 'Normal' | 'Bernoulli' | 'Categorical' | 'Binomial' | 'Poisson' | 'Beta' | 'Gamma' | 'Dirichlet'

/** A reference to another node, optionally indexed by the value of a discrete `select` node (see `at`). */
export interface NodeRef {
  readonly kind: 'ref'
  readonly node: string
  readonly select?: string
}

/** A named size, bound when the model is expanded. */
export interface SizeRef {
  readonly kind: 'size'
  readonly name: string
}

/** An argument of a distribution or deterministic node. */
export type Arg = number | readonly number[] | Tensor | NodeRef | SizeRef

/** A conditional distribution: a family and its arguments, in the constructor's order. */
export interface DistSpec {
  family: Family
  args: readonly Arg[]
}

/**
 * Deterministic links: `sum` (Σ args), `difference` (a − b), `product` (Π args), `linear` (w · x + b for a constant
 * weight vector w, a vector node x and an optional bias), `probit` (Φ(a)), `logistic` (σ(a)), `exp`, and `index`
 * (table[i, j, …]: a conditional probability table indexed by the values of discrete parents).
 */
export type DeterministicOp = 'sum' | 'difference' | 'product' | 'linear' | 'probit' | 'logistic' | 'exp' | 'index'

/** One node of a model description. */
export interface ModelNode {
  name: string
  role: 'constant' | 'latent' | 'observed' | 'deterministic'
  /** The innermost plate holding the node (null for none). */
  plate: string | null
  dist?: DistSpec
  op?: DeterministicOp
  args?: readonly Arg[]
  /** A constant's default value (overridden by `constants` when expanding). */
  value?: NodeValue | NestedArray
  /** TeX for diagrams (default: the name). */
  label?: string
}

/** A plate: a named repetition of its nodes, of a fixed size or a named (possibly ragged) size. */
export interface Plate {
  name: string
  size: number | string
  parent: string | null
  label?: string
}

/** A model description: plain, serialisable data. */
export interface Model {
  name: string
  sizes: string[]
  plates: Plate[]
  nodes: ModelNode[]
}

/** A node handle in the builder: a reference, with `at` for indexing by a discrete node's value. */
export interface NodeHandle extends NodeRef {
  at(selector: NodeRef): NodeRef
}

/** Options for a node: its TeX label. */
export interface NodeOptions {
  label?: string
}

/** A place to declare nodes: the model itself or a plate. */
export interface ModelScope {
  constant(name: string, value?: NodeValue | NestedArray, options?: NodeOptions): NodeHandle
  variable(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  observed(name: string, distribution: DistSpec, options?: NodeOptions): NodeHandle
  deterministic(name: string, op: DeterministicOp, args: readonly Arg[], options?: NodeOptions): NodeHandle
  /** A plate nested here, of a fixed size or a named size (declared on first use). */
  plate(name: string, size: number | string | SizeRef, options?: NodeOptions): ModelScope
}

/** The builder passed to `model`. */
export interface ModelBuilder extends ModelScope {
  size(name: string): SizeRef
}

/** Distribution constructors for the language; each records its family and arguments. */
export const dist = {
  Normal: (mean: Arg, sd: Arg): DistSpec => ({ family: 'Normal', args: [mean, sd] }),
  Bernoulli: (p: Arg): DistSpec => ({ family: 'Bernoulli', args: [p] }),
  /** Categorical over 0 … K − 1 with probabilities `probs` (a vector, or a node holding one). */
  Categorical: (probs: Arg): DistSpec => ({ family: 'Categorical', args: [probs] }),
  Binomial: (n: Arg, p: Arg): DistSpec => ({ family: 'Binomial', args: [n, p] }),
  Poisson: (rate: Arg): DistSpec => ({ family: 'Poisson', args: [rate] }),
  Beta: (a: Arg, b: Arg): DistSpec => ({ family: 'Beta', args: [a, b] }),
  /** Gamma by shape and rate. */
  Gamma: (shape: Arg, rate: Arg): DistSpec => ({ family: 'Gamma', args: [shape, rate] }),
  /** Dirichlet with a concentration vector, or a symmetric concentration and a dimension. */
  Dirichlet: (concentration: Arg, dimension?: Arg): DistSpec => ({
    family: 'Dirichlet',
    args: dimension === undefined ? [concentration] : [concentration, dimension],
  }),
}

const isRef = (a: Arg): a is NodeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'ref'
const isSize = (a: Arg): a is SizeRef => typeof a === 'object' && a !== null && 'kind' in a && a.kind === 'size'

/** The nodes an argument list refers to (including selectors). */
export function argRefs(args: readonly Arg[] = []): string[] {
  return args.flatMap((a) => (isRef(a) ? (a.select ? [a.node, a.select] : [a.node]) : []))
}

/** Describe a model. Node names must be unique; nodes may refer only to nodes declared before them. */
export function model(name: string, build: (m: ModelBuilder) => void): Model {
  const out: Model = { name, sizes: [], plates: [], nodes: [] }
  const names = new Set<string>()
  const handle = (node: string): NodeHandle => ({
    kind: 'ref',
    node,
    at: (selector) => ({ kind: 'ref', node, select: selector.node }),
  })
  const add = (node: ModelNode): NodeHandle => {
    if (names.has(node.name)) throw new Error(`model: duplicate node ${node.name}`)
    for (const r of argRefs(node.dist?.args ?? node.args))
      if (!names.has(r)) throw new Error(`model: ${node.name} refers to undeclared node ${r}`)
    names.add(node.name)
    out.nodes.push(node)
    return handle(node.name)
  }
  const sizeName = (s: string) => {
    if (!out.sizes.includes(s)) out.sizes.push(s)
  }
  const scope = (plate: string | null): ModelScope => ({
    constant: (n, value, o = {}) =>
      add({ name: n, role: 'constant', plate, ...(value === undefined ? {} : { value }), ...o }),
    variable: (n, d, o = {}) => add({ name: n, role: 'latent', plate, dist: d, ...o }),
    observed: (n, d, o = {}) => add({ name: n, role: 'observed', plate, dist: d, ...o }),
    deterministic: (n, op, args, o = {}) => add({ name: n, role: 'deterministic', plate, op, args, ...o }),
    plate: (n, size, o = {}) => {
      if (out.plates.some((p) => p.name === n)) throw new Error(`model: duplicate plate ${n}`)
      const s = typeof size === 'object' ? size.name : size
      if (typeof s === 'string') sizeName(s)
      out.plates.push({ name: n, size: s, parent: plate, ...o })
      return scope(n)
    },
  })
  const root = scope(null)
  build({ ...root, size: (n) => (sizeName(n), { kind: 'size', name: n }) })
  return out
}

/** The plates holding a node, outermost first. */
export function plateChain(m: Model, plate: string | null): string[] {
  const chain: string[] = []
  for (let p = plate; p !== null;) {
    chain.unshift(p)
    p = m.plates.find((q) => q.name === p)!.parent
  }
  return chain
}

// ── Expansion ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** Nested data: a value per plate index (ragged arrays allowed), leaves numbers or vectors. */
export type Nested = NodeValue | NestedArray | readonly Nested[]

/** What a model is expanded against. */
export interface Bindings {
  /** Named sizes: a number, or for a nested plate one size per index of its parent plate. */
  sizes?: Readonly<Record<string, number | readonly number[]>>
  /** Values of constants (nested by plate), overriding their defaults. */
  constants?: Readonly<Record<string, Nested>>
  /** Values of observed nodes (nested by plate). */
  data?: Readonly<Record<string, Nested>>
}

/** One copy of a node: its key (`z[2,5]`), node, plate indices (outermost first) and plates. */
export interface Instance {
  key: string
  node: ModelNode
  index: number[]
  plates: string[]
}

/** A model unrolled against bindings. */
export interface ExpandedModel {
  model: Model
  bindings: Bindings
  /** Instances in declaration order (a topological order). */
  instances: Instance[]
  byKey: Map<string, Instance>
  byNode: Map<string, Instance[]>
  /** Values of constants and observed instances, by key. */
  fixed: Map<string, NodeValue>
}

export const instanceKey = (name: string, index: readonly number[]): string =>
  index.length ? `${name}[${index.join(',')}]` : name

function toValue(v: Nested): NodeValue {
  if (typeof v === 'number' || isTensor(v)) return v as NodeValue
  return tensor(v as NestedArray)
}

/** Pick a nested value by plate index. */
function pick(v: Nested | undefined, index: readonly number[], what: string): NodeValue | undefined {
  if (v === undefined) return undefined
  let cur: Nested = isTensor(v) && index.length ? (toArray(v as Tensor) as Nested) : v
  for (const i of index) {
    if (!Array.isArray(cur)) throw new Error(`expandModel: ${what} is not nested deeply enough`)
    cur = (cur as readonly Nested[])[i]
    if (cur === undefined) throw new Error(`expandModel: ${what} has no entry at index ${index.join(',')}`)
  }
  return toValue(cur)
}

/** The size of a plate at the given outer index. */
function plateSize(m: Model, b: Bindings, plate: Plate, outer: readonly number[]): number {
  if (typeof plate.size === 'number') return plate.size
  const s = b.sizes?.[plate.size]
  if (s === undefined) {
    // An unbound size is read from the data of an observed node in this plate or one nested in it.
    const inside = m.nodes.find(
      (n) => n.role === 'observed' && b.data?.[n.name] !== undefined && plateChain(m, n.plate).includes(plate.name),
    )
    if (inside) {
      let cur = b.data![inside.name] as Nested
      for (const i of outer)
        cur = isTensor(cur) ? (toArray(cur as Tensor) as Nested[])[i] : (cur as readonly Nested[])[i]
      if (Array.isArray(cur)) return cur.length
      if (isTensor(cur)) return (cur as Tensor).shape[0]
    }
    throw new Error(`expandModel: size ${plate.size} is not bound`)
  }
  return typeof s === 'number' ? s : s[outer[outer.length - 1] ?? 0]
}

/** Unroll a model's plates against sizes, constants and data. */
export function expandModel(m: Model, bindings: Bindings = {}): ExpandedModel {
  const instances: Instance[] = []
  const byKey = new Map<string, Instance>()
  const byNode = new Map<string, Instance[]>()
  const fixed = new Map<string, NodeValue>()
  for (const node of m.nodes) {
    const plates = plateChain(m, node.plate)
    const list: Instance[] = []
    const visit = (depth: number, index: number[]) => {
      if (depth === plates.length) {
        const inst = { key: instanceKey(node.name, index), node, index: [...index], plates }
        list.push(inst)
        return
      }
      const n = plateSize(
        m,
        bindings,
        m.plates.find((p) => p.name === plates[depth])!,
        index,
      )
      for (let i = 0; i < n; i++) visit(depth + 1, [...index, i])
    }
    visit(0, [])
    for (const inst of list) {
      instances.push(inst)
      byKey.set(inst.key, inst)
      if (node.role === 'constant') {
        const v =
          pick(bindings.constants?.[node.name], inst.index, node.name) ??
          pick(node.value as Nested, inst.index, node.name)
        if (v === undefined) throw new Error(`expandModel: constant ${node.name} has no value`)
        fixed.set(inst.key, v)
      } else if (node.role === 'observed') {
        const v = pick(bindings.data?.[node.name], inst.index, node.name)
        if (v !== undefined) fixed.set(inst.key, v)
      }
    }
    byNode.set(node.name, list)
  }
  return { model: m, bindings, instances, byKey, byNode, fixed }
}

// ── Resolving arguments ─────────────────────────────────────────────────────────────────────────────────────────────

/** A lookup of instance values by key (latent values, then fixed ones). */
export type Env = (key: string) => NodeValue

/** The instance keys a reference could point to from `inst`, and a function choosing one given the values. */
export function resolveRef(
  em: ExpandedModel,
  inst: Instance,
  ref: NodeRef,
): { candidates: string[]; selector: string | null; indexesValue: boolean; choose: (env: Env) => string } {
  const parent = em.model.nodes.find((n) => n.name === ref.node)!
  const pc = plateChain(em.model, parent.plate)
  const prefixOf = (chain: string[], of: string[]) => chain.every((p, i) => of[i] === p) && chain.length <= of.length
  const selector = ref.select ? resolveRef(em, inst, { kind: 'ref', node: ref.select }).candidates[0] : null
  if (prefixOf(pc, inst.plates)) {
    const key = instanceKey(parent.name, inst.index.slice(0, pc.length))
    return { candidates: [key], selector, indexesValue: selector !== null, choose: () => key }
  }
  const outer = pc.slice(0, -1)
  if (selector !== null && prefixOf(outer, inst.plates)) {
    const prefix = inst.index.slice(0, outer.length)
    const candidates = em.byNode
      .get(parent.name)!
      .filter((c) => c.index.slice(0, outer.length).every((v, i) => v === prefix[i]))
      .map((c) => c.key)
    return {
      candidates,
      selector,
      indexesValue: false,
      choose: (env) => candidates[Math.round(env(selector) as number)],
    }
  }
  throw new Error(`model: ${inst.node.name} cannot refer to ${ref.node} across plates without .at(selector)`)
}

/** Index the first axis of a value: an entry of a vector, a row of a matrix (e.g. a CPT row for a Categorical). */
function indexValue(v: NodeValue, i: number): NodeValue {
  if (typeof v === 'number') throw new Error('model: .at on a scalar node')
  const flat = toFlat(v)
  if (v.shape.length === 1) return flat[i]
  const rest = v.shape.slice(1)
  const n = rest.reduce((a, b) => a * b, 1)
  return fromData(
    Float64Array.from({ length: n }, (_, j) => flat[i * n + j]),
    rest,
  )
}

/** The value of an argument for an instance. */
export function argValue(em: ExpandedModel, inst: Instance, arg: Arg, env: Env): NodeValue {
  if (typeof arg === 'number') return arg
  if (isSize(arg)) {
    const s = em.bindings.sizes?.[arg.name]
    if (typeof s !== 'number') throw new Error(`model: size ${arg.name} must be a number here`)
    return s
  }
  if (isRef(arg)) {
    const r = resolveRef(em, inst, arg)
    const v = env(r.choose(env))
    return r.indexesValue ? indexValue(v, Math.round(env(r.selector!) as number)) : v
  }
  if (isTensor(arg)) return arg as Tensor
  return tensor(arg as number[])
}

/** Build the `aifn/distributions` object of a stochastic node from its argument values. */
export function realise(spec: DistSpec, values: readonly NodeValue[]): Distribution {
  const [a, b] = values
  switch (spec.family) {
    case 'Normal':
      return Normal(a, b)
    case 'Bernoulli':
      return Bernoulli(a)
    case 'Categorical':
      return Categorical(a)
    case 'Binomial':
      return Binomial(a, b)
    case 'Poisson':
      return Poisson(a)
    case 'Beta':
      return Beta(a, b)
    case 'Gamma':
      return Gamma(a, b)
    case 'Dirichlet':
      return Dirichlet(b === undefined ? a : full([b as number], a as number))
  }
}

/** The value of a deterministic node from its argument values. */
export function evaluateOp(op: DeterministicOp, values: readonly NodeValue[]): NodeValue {
  switch (op) {
    case 'sum':
      return values.reduce((s, v) => add(s, v) as NodeValue)
    case 'difference':
      return sub(values[0], values[1]) as NodeValue
    case 'product':
      return values.reduce((s, v) => mul(s, v) as NodeValue)
    case 'linear': {
      const dot = sum(mul(values[0], values[1])) as number
      return values.length > 2 ? dot + (values[2] as number) : dot
    }
    case 'probit':
      return normalCdf(values[0]) as NodeValue
    case 'logistic':
      return sigmoid(values[0]) as NodeValue
    case 'exp':
      return exp(values[0]) as NodeValue
    case 'index': {
      const table = values[0]
      if (typeof table === 'number') throw new Error('model: index needs a table')
      const flat = toFlat(table)
      const k = values.length - 1
      let offset = 0
      let stride = 1
      for (let i = table.shape.length - 1; i >= 0; i--) {
        if (i < k) offset += Math.round(values[i + 1] as number) * stride
        stride *= table.shape[i]
      }
      // Indexing fewer axes than the table has leaves a sub-table (e.g. a row of probabilities for a Categorical).
      const rest = table.shape.slice(k)
      const n = rest.reduce((a, b) => a * b, 1)
      const start = offset
      return rest.length === 0
        ? flat[start]
        : fromData(
            Float64Array.from({ length: n }, (_, j) => flat[start + j]),
            rest,
          )
    }
  }
}

/**
 * An environment over the expanded model: `values` for latent (and any overridden) instances, `fixed` for constants
 * and data, deterministic instances computed on demand.
 */
export function environment(
  em: ExpandedModel,
  values: ReadonlyMap<string, NodeValue> | Readonly<Record<string, NodeValue>>,
): Env {
  const get =
    values instanceof Map ? (k: string) => values.get(k) : (k: string) => (values as Record<string, NodeValue>)[k]
  const env: Env = (key) => {
    const v = get(key) ?? em.fixed.get(key)
    if (v !== undefined) return v
    const inst = em.byKey.get(key)
    if (!inst) throw new Error(`model: unknown instance ${key}`)
    if (inst.node.role === 'deterministic')
      return evaluateOp(
        inst.node.op!,
        inst.node.args!.map((a) => argValue(em, inst, a, env)),
      )
    throw new Error(`model: no value for ${key}`)
  }
  return env
}

/** The distribution of a stochastic instance given the values of everything else. */
export function conditionalOf(em: ExpandedModel, inst: Instance, env: Env): Distribution {
  return realise(
    inst.node.dist!,
    inst.node.dist!.args.map((a) => argValue(em, inst, a, env)),
  )
}

const asNumber = (v: unknown): number => (typeof v === 'number' ? v : (sum(v as Tensor) as number))

/** log p(value of `inst` | its parents) under `env`. */
export function instanceLogDensity(em: ExpandedModel, inst: Instance, env: Env): number {
  return asNumber(conditionalOf(em, inst, env).logProb(env(inst.key)))
}

/** log p(latent, data): the sum of every stochastic instance's log conditional density. */
export function logJoint(
  m: Model | ExpandedModel,
  values: ReadonlyMap<string, NodeValue> | Readonly<Record<string, NodeValue>>,
  bindings: Bindings = {},
): number {
  const em = 'instances' in m ? m : expandModel(m, bindings)
  const env = environment(em, values)
  let total = 0
  for (const inst of em.instances)
    if (inst.node.role === 'latent' || inst.node.role === 'observed') total += instanceLogDensity(em, inst, env)
  return total
}

// ── Structure ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** The stochastic instances an instance depends on directly, looking through deterministic nodes (all candidates of an `at`). */
export function stochasticParents(em: ExpandedModel, inst: Instance): string[] {
  const out = new Set<string>()
  const visit = (i: Instance) => {
    for (const a of i.node.dist?.args ?? i.node.args ?? []) {
      if (!isRef(a)) continue
      const r = resolveRef(em, i, a)
      for (const k of [...r.candidates, ...(r.selector ? [r.selector] : [])]) {
        const p = em.byKey.get(k)!
        if (p.node.role === 'deterministic') visit(p)
        else if (p.node.role !== 'constant') out.add(k)
      }
    }
  }
  visit(inst)
  return [...out]
}

/** Parents, children and co-parents of every stochastic instance. */
export function dependencyMaps(em: ExpandedModel): { parents: Map<string, string[]>; children: Map<string, string[]> } {
  const parents = new Map<string, string[]>()
  const children = new Map<string, string[]>()
  for (const inst of em.instances) {
    if (inst.node.role !== 'latent' && inst.node.role !== 'observed') continue
    const ps = stochasticParents(em, inst)
    parents.set(inst.key, ps)
    if (!children.has(inst.key)) children.set(inst.key, [])
    for (const p of ps) {
      if (!children.has(p)) children.set(p, [])
      children.get(p)!.push(inst.key)
    }
  }
  return { parents, children }
}

/** The Markov blanket of a stochastic instance: its parents, children and the children's other parents. */
export interface MarkovBlanket {
  parents: string[]
  children: string[]
  coParents: string[]
  /** All of the above, without repeats. */
  blanket: string[]
}

/**
 * The Markov blanket of `key` (an instance key such as `z[0,3]`, or a node name for an unplated node) in a model
 * expanded against `bindings`.
 */
export function modelMarkovBlanket(m: Model | ExpandedModel, key: string, bindings: Bindings = {}): MarkovBlanket {
  const em = 'instances' in m ? m : expandModel(m, bindings)
  const { parents, children } = dependencyMaps(em)
  if (!parents.has(key)) throw new Error(`markovBlanket: ${key} is not a stochastic instance`)
  const ch = children.get(key) ?? []
  const co = [...new Set(ch.flatMap((c) => parents.get(c)!).filter((p) => p !== key))]
  const ps = parents.get(key)!
  return { parents: ps, children: ch, coParents: co, blanket: [...new Set([...ps, ...ch, ...co])] }
}

/** The number of values of a discrete node (Bernoulli 2, Categorical K, Binomial n + 1), or null. */
export function cardinalityOf(em: ExpandedModel, inst: Instance): number | null {
  const d = inst.node.dist
  if (!d) return null
  if (d.family === 'Bernoulli') return 2
  if (d.family === 'Binomial') {
    const n = d.args[0]
    return typeof n === 'number' ? n + 1 : null
  }
  if (d.family !== 'Categorical') return null
  const probs = d.args[0]
  if (Array.isArray(probs)) return probs.length
  if (isTensor(probs)) return (probs as Tensor).shape[(probs as Tensor).shape.length - 1]
  if (!isRef(probs)) return null
  const target = em.byNode.get(probs.node)![0]
  if (target.node.role === 'constant') {
    const v = em.fixed.get(target.key)!
    return typeof v === 'number' ? null : v.shape[v.shape.length - 1]
  }
  if (target.node.op === 'index') {
    const t = target.node.args![0]
    const table = isRef(t)
      ? em.fixed.get(em.byNode.get(t.node)![0].key)
      : isTensor(t)
        ? (t as Tensor)
        : tensor(t as number[])
    return table === undefined || typeof table === 'number' ? null : table.shape[table.shape.length - 1]
  }
  const pd = target.node.dist
  if (pd?.family === 'Dirichlet') {
    const dim = pd.args[1]
    if (typeof dim === 'number') return dim
    if (dim !== undefined && isSize(dim)) return em.bindings.sizes?.[dim.name] as number
    const c = pd.args[0]
    if (Array.isArray(c)) return c.length
    if (isTensor(c)) return (c as Tensor).shape[0]
  }
  return null
}

/** Ancestral sampling: a value for every latent and unobserved instance, in declaration order. */
export function sampleModel(
  s: Stream,
  m: Model,
  bindings: Bindings & { given?: Readonly<Record<string, Nested>> } = {},
): Map<string, NodeValue> {
  const em = expandModel(m, bindings)
  const values = new Map<string, NodeValue>()
  const env = environment(em, values)
  for (const inst of em.instances) {
    if (inst.node.role !== 'latent' && inst.node.role !== 'observed') continue
    if (em.fixed.has(inst.key)) continue
    const given = pick(bindings.given?.[inst.node.name], inst.index, inst.node.name)
    values.set(inst.key, given ?? (conditionalOf(em, inst, env).sample(s.child(inst.key)) as NodeValue))
  }
  return values
}

/** Collect the values of one node's instances into nested arrays by plate index (e.g. sampled data). */
export function nestedValues(em: ExpandedModel, node: string, values: ReadonlyMap<string, NodeValue>): Nested {
  const out: Nested[] = []
  const insts = em.byNode.get(node)!
  if (insts.length === 1 && insts[0].index.length === 0) return values.get(insts[0].key) ?? em.fixed.get(insts[0].key)!
  for (const inst of insts) {
    let cur = out
    inst.index.slice(0, -1).forEach((i) => {
      cur[i] ??= []
      cur = cur[i] as Nested[]
    })
    const v = values.get(inst.key) ?? em.fixed.get(inst.key)!
    cur[inst.index[inst.index.length - 1]] = typeof v === 'number' ? v : (toArray(v) as Nested)
  }
  return out
}
