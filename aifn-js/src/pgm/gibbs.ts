/**
 * Gibbs sampling (Geman & Geman 1984; Gelfand & Smith 1990), built from the Markov blanket: each variable is redrawn
 * from its full conditional given the others.
 *
 * - On a discrete factor graph, the conditional of x_v is the normalised product of the factors that mention it.
 * - On a model description, a discrete latent variable's conditional is enumerated over its values (its own density
 *   times its children's), and a continuous one's comes from a conjugate update: Beta with Bernoulli or Binomial
 *   children, Dirichlet with Categorical children (through `at` selections too), Normal (unknown mean) with Normal
 *   children of known sd, and Gamma with Poisson children (Bishop 2006, "Pattern Recognition and Machine Learning",
 *   §2.1–2.3). A latent variable without such a conditional is an error at `init`.
 */

import { beta as betaDraw, categorical, dirichlet, gamma as gammaDraw, normal, type Stream } from 'aifn/random'
import { fromData, isTensor, toFlat, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { stridesOf, valuesOf, type DiscreteFactorGraph } from './factors'
import {
  argValue,
  cardinalityOf,
  dependencyMaps,
  environment,
  expandModel,
  instanceLogDensity,
  logJoint,
  resolveRef,
  sampleModel,
  type Bindings,
  type Env,
  type ExpandedModel,
  type Instance,
  type Model,
  type NodeRef,
  type NodeValue,
} from './model'

// ── Gibbs on a discrete factor graph ────────────────────────────────────────────────────────────────────────────────

/** Options of {@link factorGraphGibbsSteps}. */
export interface FactorGraphGibbsOptions {
  graph: DiscreteFactorGraph
  /** Starting values (default: drawn uniformly). */
  initial?: ArrayLike<number>
  /** Observed values, held fixed. */
  evidence?: Readonly<Record<number, number>>
  /** `variable`: one variable per step; `sweep`: every free variable once per step (default). */
  granularity?: 'variable' | 'sweep'
  /** The visiting order (default 0 … V − 1). */
  order?: readonly number[]
}

/** The state of Gibbs sampling on a discrete factor graph. */
export interface FactorGraphGibbsState {
  graph: DiscreteFactorGraph
  granularity: 'variable' | 'sweep'
  order: number[]
  stream: Stream
  /** The current values (int32). */
  assignment: Tensor
  /** Completed sweeps, and the position in the current one. */
  sweep: number
  position: number
  /** The variable updated last and the conditional it was drawn from (−1 and empty before the first draw). */
  variable: number
  conditional: Tensor
  /** Per variable, how often each value was held at the end of a sweep; divide by `sweep` for marginal estimates. */
  counts: Tensor[]
}

/** p(x_v = · | rest) from the factors mentioning v. */
function discreteConditional(g: DiscreteFactorGraph, a: Int32Array, v: number): Float64Array {
  const p = new Float64Array(g.cardinalities[v]).fill(1)
  for (const f of g.factors) {
    const i = f.scope.indexOf(v)
    if (i < 0) continue
    const strides = stridesOf(f.table.shape)
    const t = valuesOf(f.table)
    let base = 0
    f.scope.forEach((u, j) => j !== i && (base += a[u] * strides[j]))
    for (let x = 0; x < p.length; x++) p[x] *= t[base + x * strides[i]]
  }
  let z = 0
  for (const x of p) z += x
  return p.map((x) => x / z)
}

/**
 * Gibbs sampling on a discrete factor graph as a traceable algorithm. Each draw of variable v at sweep t uses the
 * substream `stream.child(t, v)`, so runs are reproducible and `seek` agrees with `run`.
 */
export const factorGraphGibbsSteps: Algorithm<FactorGraphGibbsOptions, FactorGraphGibbsState> = {
  name: 'pgm.gibbs.factor-graph',
  init: (o, s) => {
    const stream =
      s ??
      (() => {
        throw new Error('factorGraphGibbsSteps: needs a stream')
      })()
    const V = o.graph.cardinalities.length
    const a = new Int32Array(V)
    for (let v = 0; v < V; v++) a[v] = o.initial ? o.initial[v] : stream.child('init', v).int(o.graph.cardinalities[v])
    const evidence = o.evidence ?? {}
    for (const [k, x] of Object.entries(evidence)) a[Number(k)] = x
    const order = (o.order ?? a.map((_, v) => v)).filter((v) => evidence[v] === undefined)
    return {
      graph: o.graph,
      granularity: o.granularity ?? 'sweep',
      order: Array.from(order),
      stream,
      assignment: fromData(a, [V]),
      sweep: 0,
      position: 0,
      variable: -1,
      conditional: fromData(new Float64Array(0), [0]),
      counts: o.graph.cardinalities.map((k) => fromData(new Float64Array(k), [k])),
    }
  },
  step: (s) => {
    const a = Int32Array.from(s.assignment.data)
    const n = s.granularity === 'sweep' ? s.order.length - s.position : 1
    let position = s.position
    let variable = s.variable
    let conditional = s.conditional
    for (let i = 0; i < n; i++) {
      variable = s.order[position]
      const p = discreteConditional(s.graph, a, variable)
      a[variable] = categorical(s.stream.child(s.sweep, variable), p)
      conditional = fromData(p, [p.length])
      position++
    }
    let { sweep, counts } = s
    if (position >= s.order.length) {
      position = 0
      sweep += 1
      counts = counts.map((c, v) => {
        const d = Float64Array.from(c.data)
        d[a[v]] += 1
        return fromData(d, c.shape)
      })
    }
    return { ...s, assignment: fromData(a, [a.length]), sweep, position, variable, conditional, counts }
  },
}

/** Marginal estimates from a Gibbs state: the fraction of completed sweeps each variable spent at each value. */
export function gibbsMarginals(s: FactorGraphGibbsState): Tensor[] {
  return s.counts.map((c) =>
    fromData(
      c.data.map((x) => x / Math.max(s.sweep, 1)),
      c.shape,
    ),
  )
}

// ── Gibbs on a model description ────────────────────────────────────────────────────────────────────────────────────

/** Options of {@link gibbsSteps}. */
export interface GibbsOptions {
  model: Model
  bindings?: Bindings
  /** Starting values by instance key (default: an ancestral draw from the prior with the data held fixed). */
  initial?: Readonly<Record<string, NodeValue>>
  /** `variable`: one latent instance per step; `sweep`: all of them (default). */
  granularity?: 'variable' | 'sweep'
}

/** The state of Gibbs sampling on a model. */
export interface GibbsState {
  expanded: ExpandedModel
  granularity: 'variable' | 'sweep'
  stream: Stream
  /** Latent instance keys in visiting (declaration) order, and the kind of conditional of each. */
  order: string[]
  kinds: Record<string, ConditionalKind>
  /** Current values of the latent instances, by key. */
  values: Readonly<Record<string, NodeValue>>
  sweep: number
  position: number
  /** The instance updated last. */
  updated: string | null
  /** log p(latent, data) at the end of the last step. */
  logJoint: number
}

/** How a latent instance is redrawn. */
export type ConditionalKind = 'enumerate' | 'beta' | 'dirichlet' | 'normal' | 'gamma'

/** Children of `key` whose `argIndex`-th argument is a direct reference that can select `key`. */
function directChildren(em: ExpandedModel, key: string, children: readonly string[]) {
  return children.map((c) => {
    const inst = em.byKey.get(c)!
    const hits = inst.node.dist!.args.flatMap((a, i) => {
      if (typeof a !== 'object' || a === null || !('kind' in a) || a.kind !== 'ref') return []
      const r = resolveRef(em, inst, a as NodeRef)
      return r.candidates.includes(key) && !r.indexesValue ? [{ i, r }] : []
    })
    return { inst, hits }
  })
}

const CONJUGATE: Record<string, Partial<Record<string, number>>> = {
  // prior family → child family → which child argument must be the prior's variable
  Beta: { Bernoulli: 0, Binomial: 1 },
  Dirichlet: { Categorical: 0 },
  Normal: { Normal: 0 },
  Gamma: { Poisson: 0 },
}
const KIND: Record<string, ConditionalKind> = { Beta: 'beta', Dirichlet: 'dirichlet', Normal: 'normal', Gamma: 'gamma' }

function classify(em: ExpandedModel, inst: Instance, children: readonly string[]): ConditionalKind {
  if (cardinalityOf(em, inst) !== null) return 'enumerate'
  const family = inst.node.dist!.family
  const rule = CONJUGATE[family]
  if (!rule) throw new Error(`gibbs: no conjugate conditional for ${inst.key} (${family})`)
  for (const { inst: child, hits } of directChildren(em, inst.key, children)) {
    const want = rule[child.node.dist!.family]
    if (want === undefined || hits.length !== 1 || hits[0].i !== want)
      throw new Error(`gibbs: ${child.key} is not a conjugate child of ${inst.key} (${family})`)
  }
  return KIND[family]
}

const num = (v: NodeValue): number => (typeof v === 'number' ? v : toFlat(v)[0])

/** Draw a new value of `inst` from its full conditional. */
function redraw(
  em: ExpandedModel,
  inst: Instance,
  kind: ConditionalKind,
  children: readonly string[],
  values: Map<string, NodeValue>,
  s: Stream,
): NodeValue {
  const env: Env = environment(em, values)
  const args = inst.node.dist!.args.map((a) => argValue(em, inst, a, env))
  const active = directChildren(em, inst.key, children).filter(({ hits }) =>
    hits.every(({ r }) => r.choose(env) === inst.key),
  )
  const xs = active.map(({ inst: c }) => ({ c, x: env(c.key) }))
  switch (kind) {
    case 'enumerate': {
      const K = cardinalityOf(em, inst)!
      const logs = new Float64Array(K)
      const kids = children.map((c) => em.byKey.get(c)!)
      for (let k = 0; k < K; k++) {
        values.set(inst.key, k)
        logs[k] = instanceLogDensity(em, inst, env) + kids.reduce((t, c) => t + instanceLogDensity(em, c, env), 0)
      }
      const top = Math.max(...logs)
      return categorical(
        s,
        logs.map((l) => Math.exp(l - top)),
      )
    }
    case 'beta': {
      let [a, b] = [num(args[0]), num(args[1])]
      for (const { c, x } of xs) {
        const n = c.node.dist!.family === 'Binomial' ? num(argValue(em, c, c.node.dist!.args[0], env)) : 1
        a += num(x)
        b += n - num(x)
      }
      return betaDraw(s, a, b)
    }
    case 'dirichlet': {
      const conc = isTensor(args[0])
        ? Float64Array.from(toFlat(args[0] as Tensor))
        : new Float64Array(num(args[1])).fill(num(args[0]))
      for (const { x } of xs) conc[Math.round(num(x))] += 1
      return dirichlet(s, conc)
    }
    case 'normal': {
      let precision = 1 / num(args[1]) ** 2
      let shift = num(args[0]) * precision
      for (const { c, x } of xs) {
        const sd = num(argValue(em, c, c.node.dist!.args[1], env))
        precision += 1 / (sd * sd)
        shift += num(x) / (sd * sd)
      }
      return normal(s, shift / precision, Math.sqrt(1 / precision))
    }
    case 'gamma': {
      let [shape, rate] = [num(args[0]), num(args[1])]
      for (const { x } of xs) {
        shape += num(x)
        rate += 1
      }
      return gammaDraw(s, shape, 1 / rate)
    }
  }
}

/**
 * Gibbs sampling on a model description as a traceable algorithm. Latent instances are visited in declaration order;
 * the draw for instance `key` in sweep t uses `stream.child(t, key)`.
 */
export const gibbsSteps: Algorithm<GibbsOptions, GibbsState> = {
  name: 'pgm.gibbs',
  init: (o, s) => {
    if (!s) throw new Error('gibbsSteps: needs a stream')
    const bindings = o.bindings ?? {}
    const em = expandModel(o.model, bindings)
    const { children } = dependencyMaps(em)
    const order = em.instances
      .filter((i) => i.node.role === 'latent' || (i.node.role === 'observed' && !em.fixed.has(i.key)))
      .map((i) => i.key)
    const kinds: Record<string, ConditionalKind> = {}
    for (const key of order) kinds[key] = classify(em, em.byKey.get(key)!, children.get(key) ?? [])
    const drawn = sampleModel(s.child('init'), o.model, bindings)
    const values: Record<string, NodeValue> = {}
    for (const key of order) values[key] = o.initial?.[key] ?? drawn.get(key)!
    return {
      expanded: em,
      granularity: o.granularity ?? 'sweep',
      stream: s,
      order,
      kinds,
      values,
      sweep: 0,
      position: 0,
      updated: null,
      logJoint: logJoint(em, values),
    }
  },
  step: (s) => {
    const em = s.expanded
    const { children } = childrenCache(em)
    const values = new Map(Object.entries(s.values))
    const n = s.granularity === 'sweep' ? s.order.length - s.position : 1
    let position = s.position
    let updated = s.updated
    for (let i = 0; i < n; i++) {
      const key = s.order[position]
      const v = redraw(
        em,
        em.byKey.get(key)!,
        s.kinds[key],
        children.get(key) ?? [],
        values,
        s.stream.child(s.sweep, key),
      )
      values.set(key, v)
      updated = key
      position++
    }
    let sweep = s.sweep
    if (position >= s.order.length) {
      position = 0
      sweep++
    }
    const out = Object.fromEntries(values)
    return { ...s, values: out, sweep, position, updated, logJoint: logJoint(em, out) }
  },
}

const cache = new WeakMap<ExpandedModel, ReturnType<typeof dependencyMaps>>()
function childrenCache(em: ExpandedModel) {
  let c = cache.get(em)
  if (!c) cache.set(em, (c = dependencyMaps(em)))
  return c
}
