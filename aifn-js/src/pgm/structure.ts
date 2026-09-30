/**
 * From a model description to its structures: the factor graph (one factor per stochastic instance, over the
 * instance and its stochastic parents), the discrete factor graph the exact and message-passing engines run on
 * (observed values clamped), Markov blankets, and diagram specs for plate notation and factor graphs.
 *
 * Diagram specs are plain data in the shape of the lab's `DiagramSpec` (`aifn-lab/src/diagram/types.ts`): nodes
 * with grid positions or a layered layout, straight links, and plates as labelled groups. aifn emits data only; the
 * lab draws it.
 */

import { fromEdges, topologicalSort, type Graph } from 'aifn/graph'
import {
  discreteFactor,
  discreteFactorGraph,
  factorGraphNeighbours,
  type DiscreteFactor,
  type DiscreteFactorGraph,
} from './factors'
import {
  argRefs,
  cardinalityOf,
  dependencyMaps,
  environment,
  expandModel,
  instanceLogDensity,
  modelMarkovBlanket,
  plateChain,
  type Bindings,
  type ExpandedModel,
  type MarkovBlanket,
  type Model,
  type NodeValue,
} from './model'

/** A variable of a model's factor graph: a stochastic instance. */
export interface ModelVariable {
  key: string
  node: string
  observed: boolean
  /** Number of values for a discrete node, null otherwise. */
  cardinality: number | null
}

/** A factor of a model's factor graph: the conditional p(child | parents) of one stochastic instance. */
export interface ModelFactor {
  key: string
  node: string
  /** The variable whose conditional this is. */
  child: number
  /** Variable indices: the child first, then its stochastic parents. */
  scope: number[]
}

/** A model expanded into variables and factors, with the bipartite graph (variables first, then factors). */
export interface ModelFactorGraph {
  expanded: ExpandedModel
  variables: ModelVariable[]
  factors: ModelFactor[]
  graph: Graph
}

const expand = (m: Model | ExpandedModel, b: Bindings): ExpandedModel => ('instances' in m ? m : expandModel(m, b))

/**
 * Expand a model into its factor graph. Deterministic nodes are folded into the factors of their stochastic
 * children; an `at` reference makes the factor depend on every instance it could select, and on the selector.
 */
export function toFactorGraph(m: Model | ExpandedModel, bindings: Bindings = {}): ModelFactorGraph {
  const em = expand(m, bindings)
  const { parents } = dependencyMaps(em)
  const variables: ModelVariable[] = []
  const index = new Map<string, number>()
  for (const inst of em.instances) {
    if (inst.node.role !== 'latent' && inst.node.role !== 'observed') continue
    index.set(inst.key, variables.length)
    variables.push({
      key: inst.key,
      node: inst.node.name,
      observed: em.fixed.has(inst.key),
      cardinality: cardinalityOf(em, inst),
    })
  }
  const factors: ModelFactor[] = variables.map((v, child) => ({
    key: v.key,
    node: v.node,
    child,
    scope: [child, ...parents.get(v.key)!.map((p) => index.get(p)!)],
  }))
  const V = variables.length
  const graph = fromEdges(
    V + factors.length,
    factors.flatMap((f, k) => f.scope.map((v) => [v, V + k] as const)),
    { directed: false, labels: [...variables.map((v) => v.key), ...factors.map((f) => `p(${f.key})`)] },
  )
  return { expanded: em, variables, factors, graph }
}

/** A discrete factor graph over a model's latent variables, with the data clamped. */
export interface ModelDiscreteGraph {
  graph: DiscreteFactorGraph
  /** The instance key of each variable of `graph`. */
  keys: string[]
  /** Σ of the log-densities of factors with no latent variable (observed given constants): add to log Z. */
  logConstant: number
}

/**
 * Tabulate a model with discrete latent variables as a discrete factor graph: each factor p(child | parents) becomes
 * a table over its latent variables, with observed values fixed. log Z of the result plus `logConstant` is
 * log p(data). Throws when a latent variable is not discrete with a finite number of values.
 */
export function toDiscreteFactorGraph(m: Model | ExpandedModel, bindings: Bindings = {}): ModelDiscreteGraph {
  const fg = toFactorGraph(m, bindings)
  const em = fg.expanded
  const latent = fg.variables.map((v, i) => (v.observed ? -1 : i)).filter((i) => i >= 0)
  for (const i of latent)
    if (fg.variables[i].cardinality === null)
      throw new Error(`toDiscreteFactorGraph: ${fg.variables[i].key} is not discrete with finitely many values`)
  const position = new Map(latent.map((v, i) => [v, i]))
  const cards = latent.map((i) => fg.variables[i].cardinality!)
  const factors: DiscreteFactor[] = []
  let logConstant = 0
  for (const f of fg.factors) {
    const scope = f.scope.filter((v) => position.has(v))
    const child = em.byKey.get(f.key)!
    if (scope.length === 0) {
      logConstant += instanceLogDensity(em, child, environment(em, new Map()))
      continue
    }
    const values = new Map<string, NodeValue>()
    const env = environment(em, values)
    factors.push(
      discreteFactor(
        scope.map((v) => position.get(v)!),
        cards,
        (a) => {
          scope.forEach((v, i) => values.set(fg.variables[v].key, a[i]))
          return Math.exp(instanceLogDensity(em, child, env))
        },
        `p(${f.key})`,
      ),
    )
  }
  const keys = latent.map((i) => fg.variables[i].key)
  return { graph: discreteFactorGraph(cards, factors, keys), keys, logConstant }
}

/**
 * The Markov blanket. For a discrete factor graph and a variable index: the variables sharing a factor with it. For
 * a model (or expanded model) and an instance key: its parents, children and co-parents.
 */
export function markovBlanket(graph: DiscreteFactorGraph, variable: number): number[]
export function markovBlanket(m: Model | ExpandedModel, key: string, bindings?: Bindings): MarkovBlanket
export function markovBlanket(
  target: DiscreteFactorGraph | Model | ExpandedModel,
  which: number | string,
  bindings: Bindings = {},
): number[] | MarkovBlanket {
  if ('cardinalities' in target) return factorGraphNeighbours(target, which as number)
  return modelMarkovBlanket(target, which as string, bindings)
}

// ── Diagrams ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** A diagram node, in the shape of the lab's `DiagramNode`. */
export interface DiagramNodeData {
  id: string
  x?: number
  y?: number
  shape: 'circle' | 'factor' | 'text'
  label?: string
  tone?: number | 'neutral' | 'ink'
  filled?: boolean
  dashed?: boolean
  small?: boolean
  w?: number
  h?: number
  labelSide?: 'n' | 's' | 'e' | 'w'
  highlight?: boolean
  state?: 'idle' | 'active' | 'done'
}

/** A diagram edge, in the shape of the lab's `DiagramEdge`. */
export interface DiagramEdgeData {
  from: string
  to: string
  route: 'straight'
  arrow: 'end' | 'none' | 'mid'
  highlight?: boolean
  state?: 'idle' | 'active' | 'done'
}

/** A plate, in the shape of the lab's `DiagramGroup`. */
export interface DiagramGroupData {
  id: string
  label?: string
  tone?: 'ink'
  around: string[]
  pad: number
  labelAt: 'bottom-right'
}

/** A diagram spec (the lab's `DiagramSpec`). */
export interface DiagramData {
  nodes: DiagramNodeData[]
  edges: DiagramEdgeData[]
  groups: DiagramGroupData[]
  layout?: 'manual' | 'layered'
  layered?: { direction?: 'right' | 'down'; layerGap?: number; nodeGap?: number }
}

/** Options of {@link toPlateDiagram}. */
export interface PlateDiagramOptions {
  /** Fixed centres in grid units, by node name, overriding the automatic layout. */
  positions?: Readonly<Record<string, readonly [number, number]>>
  /** Horizontal distance between layers and vertical distance between nodes, in grid units (default 1.8, 1.6). */
  layerGap?: number
  nodeGap?: number
}

const tex = (s: string) => (s.includes('$') ? s : `$${s}$`)

/**
 * Plate notation for a model: latent variables as circles, observed ones shaded, deterministic ones dashed,
 * constants as small circles; directed links from each node's parents; each plate a group around its nodes, labelled
 * with its size at the bottom right. Nodes are placed in columns by depth (longest path from a source) and ordered in
 * each column by the mean height of their parents; `positions` overrides any of them.
 */
export function toPlateDiagram(m: Model, options: PlateDiagramOptions = {}): DiagramData {
  const { layerGap = 1.8, nodeGap = 1.6 } = options
  const idx = new Map(m.nodes.map((n, i) => [n.name, i]))
  const links: [number, number][] = []
  m.nodes.forEach((n, i) => {
    for (const r of new Set(argRefs(n.dist?.args ?? n.args))) links.push([idx.get(r)!, i])
  })
  const g = fromEdges(m.nodes.length, links)
  const order = Array.from(topologicalSort(g).order.data)
  const layer = new Array<number>(m.nodes.length).fill(0)
  for (const v of order) for (const [a, b] of links) if (a === v) layer[b] = Math.max(layer[b], layer[a] + 1)
  // Hyperparameters sit just before their first child rather than in column 0.
  m.nodes.forEach((n, i) => {
    if (n.role !== 'constant') return
    const kids = links.filter(([a]) => a === i).map(([, b]) => layer[b])
    if (kids.length) layer[i] = Math.min(...kids) - 1
  })
  const y = new Array<number>(m.nodes.length).fill(0)
  const layers = [...new Set(layer)].sort((a, b) => a - b)
  for (const L of layers) {
    const members = m.nodes.map((_, i) => i).filter((i) => layer[i] === L)
    const want = members.map((i) => {
      const ps = links
        .filter(([, b]) => b === i)
        .map(([a]) => a)
        .filter((a) => layer[a] < L)
      return ps.length ? ps.reduce((s, a) => s + y[a], 0) / ps.length : members.indexOf(i) * nodeGap
    })
    const sorted = members.map((i, k) => ({ i, w: want[k] })).sort((a, b) => a.w - b.w)
    let last = -Infinity
    for (const { i, w } of sorted) {
      y[i] = Math.max(w, last + nodeGap)
      last = y[i]
    }
  }
  const nodes: DiagramNodeData[] = m.nodes.map((n, i) => {
    const at = options.positions?.[n.name]
    const small = n.role === 'constant'
    return {
      id: n.name,
      x: at ? at[0] : layer[i] * layerGap,
      y: at ? at[1] : y[i],
      shape: 'circle',
      tone: 'ink',
      label: tex(n.label ?? n.name),
      ...(n.role === 'observed' ? { filled: true } : {}),
      ...(n.role === 'deterministic' ? { dashed: true } : {}),
      ...(small ? { small: true, w: 0.7, h: 0.7 } : {}),
    }
  })
  const edges: DiagramEdgeData[] = links.map(([a, b]) => ({
    from: m.nodes[a].name,
    to: m.nodes[b].name,
    route: 'straight',
    arrow: 'end',
  }))
  // Outer plates get more padding than the plates nested inside them, so the borders do not coincide.
  const depthBelow = (p: string): number => {
    const kids = m.plates.filter((q) => q.parent === p)
    return kids.length ? 1 + Math.max(...kids.map((q) => depthBelow(q.name))) : 0
  }
  const groups: DiagramGroupData[] = m.plates
    .map((p) => ({
      id: `plate:${p.name}`,
      label: tex(p.label ?? String(p.size)),
      tone: 'ink' as const,
      around: m.nodes.filter((n) => plateChain(m, n.plate).includes(p.name)).map((n) => n.name),
      pad: 0.3 + 0.4 * depthBelow(p.name),
      labelAt: 'bottom-right' as const,
    }))
    .filter((grp) => grp.around.length > 0)
  return { nodes, edges, groups, layout: 'manual' }
}

/** Options of {@link toFactorDiagram}. */
export interface FactorDiagramOptions {
  /** Bindings to expand a model against. */
  bindings?: Bindings
  /** Highlight this variable (an instance key for a model, an index for a discrete graph) and its Markov blanket. */
  highlight?: string | number
  /**
   * Fixed centres of the variables in grid units (by key for a model, by index for a discrete graph). Factors are
   * placed at the mean of their variables (offset above for unary factors). Without positions the layout is layered.
   */
  positions?: Readonly<Record<string | number, readonly [number, number]>>
  /** Labels for variables (TeX allowed), by key or index. */
  labels?: Readonly<Record<string | number, string>>
}

/**
 * A factor-graph diagram: variables as circles (observed ones shaded), factors as small squares, undirected links.
 * For a model, ids are `var <key>` and `factor <key>` and links run from parents through each factor to its child (the
 * layered layout reads them left to right); for a discrete factor graph, ids are `var <index>` and `factor <index>` (no colons: the lab reads `id:side` as a port). With
 * `highlight`, the variable is active, its blanket and the factors joining them done, and everything else idle.
 */
export function toFactorDiagram(
  target: Model | ExpandedModel | DiscreteFactorGraph,
  options: FactorDiagramOptions = {},
): DiagramData {
  type V = { id: string; name: string | number; label: string; observed: boolean }
  type F = { id: string; scope: number[]; child: number | null }
  let vars: V[]
  let facs: F[]
  let blanket = new Set<number>()
  let centre = -1
  if ('cardinalities' in target) {
    vars = target.cardinalities.map((_, i) => ({
      id: `var ${i}`,
      name: i,
      label: options.labels?.[i] ?? tex(target.names?.[i] ?? `x_{${i}}`),
      observed: false,
    }))
    facs = target.factors.map((f, k) => ({ id: `factor ${k}`, scope: [...f.scope], child: null }))
    if (options.highlight !== undefined) {
      centre = options.highlight as number
      blanket = new Set(factorGraphNeighbours(target, centre))
    }
  } else {
    const fg = toFactorGraph(target, options.bindings ?? {})
    vars = fg.variables.map((v) => ({
      id: `var ${v.key}`,
      name: v.key,
      label: options.labels?.[v.key] ?? tex(v.key.replace(/\[(.*)\]/, '_{$1}')),
      observed: v.observed,
    }))
    facs = fg.factors.map((f) => ({ id: `factor ${f.key}`, scope: f.scope, child: f.child }))
    if (options.highlight !== undefined) {
      centre = fg.variables.findIndex((v) => v.key === options.highlight)
      if (centre < 0) throw new Error(`toFactorDiagram: no variable ${options.highlight}`)
      const mb = modelMarkovBlanket(fg.expanded, options.highlight as string)
      const keyIndex = new Map(fg.variables.map((v, i) => [v.key, i]))
      blanket = new Set(mb.blanket.map((k) => keyIndex.get(k)!))
    }
  }
  const highlighting = centre >= 0
  const pos = options.positions
  const nodes: DiagramNodeData[] = vars.map((v, i) => {
    const at = pos?.[v.name]
    return {
      id: v.id,
      shape: 'circle',
      tone: 'ink',
      label: v.label,
      ...(at ? { x: at[0], y: at[1] } : {}),
      ...(v.observed ? { filled: true } : {}),
      ...(highlighting ? { state: i === centre ? 'active' : blanket.has(i) ? 'done' : 'idle' } : {}),
      ...(i === centre ? { highlight: true } : {}),
    }
  })
  const touches = (f: F) => f.scope.includes(centre)
  facs.forEach((f) => {
    let at: [number, number] | undefined
    if (pos) {
      const ps = f.scope.map((v) => pos[vars[v].name]).filter(Boolean) as (readonly [number, number])[]
      if (ps.length === f.scope.length) {
        at = [ps.reduce((s, p) => s + p[0], 0) / ps.length, ps.reduce((s, p) => s + p[1], 0) / ps.length]
        if (ps.length === 1) at = [at[0] - 0.7, at[1] - 0.7]
      }
    }
    nodes.push({
      id: f.id,
      shape: 'factor',
      labelSide: 'n',
      ...(at ? { x: at[0], y: at[1] } : {}),
      ...(highlighting ? { state: touches(f) ? 'done' : 'idle' } : {}),
    })
  })
  const edges: DiagramEdgeData[] = []
  facs.forEach((f) => {
    const on = (v: number) =>
      highlighting
        ? { state: (touches(f) && (v === centre || blanket.has(v)) ? 'done' : 'idle') as 'done' | 'idle' }
        : {}
    for (const v of f.scope) {
      if (v === f.child) continue
      edges.push({ from: vars[v].id, to: f.id, route: 'straight', arrow: 'none', ...on(v) })
    }
    if (f.child !== null)
      edges.push({ from: f.id, to: vars[f.child].id, route: 'straight', arrow: 'none', ...on(f.child) })
  })
  const layered = !pos || nodes.some((n) => n.x === undefined)
  return {
    nodes,
    edges,
    groups: [],
    ...(layered
      ? { layout: 'layered' as const, layered: { direction: 'right' as const, layerGap: 3, nodeGap: 1 } }
      : { layout: 'manual' as const }),
  }
}
