/**
 * Automatic layered layout for diagrams generated from data (Sugiyama, Tagawa and Toda, 1981): nodes go into layers
 * by topological depth (cycles broken at depth-first back edges and the order from Kahn's algorithm, both from
 * `aifn/graph`), long edges are split by virtual nodes, each layer is ordered by barycentre sweeps to reduce
 * crossings, and positions within a layer are pulled towards their neighbours while keeping the order and spacing.
 */
import { depthFirstSearch, topologicalSort } from 'aifn/graph/traversal'
import { fromEdges } from 'aifn/graph'
import type { DiagramEdge, DiagramSpec, PlacedNode } from './types'

const endId = (ref: string) => ref.split(':')[0]

/** Crossings between two adjacent layers, given each edge as (position above, position below). */
function crossings(pairs: [number, number][]): number {
  let count = 0
  for (let i = 0; i < pairs.length; i++)
    for (let j = i + 1; j < pairs.length; j++) {
      const [a1, b1] = pairs[i]
      const [a2, b2] = pairs[j]
      if ((a1 - a2) * (b1 - b2) < 0) count++
    }
  return count
}

/**
 * Least-squares positions for an ordered row: minimise Σ (yᵢ − dᵢ)² subject to yᵢ₊₁ − yᵢ ≥ gap. With zᵢ = dᵢ − i·gap
 * this is isotonic regression of z, solved by pooling adjacent violators.
 */
function spaced(desired: number[], gap: number): number[] {
  const blocks: { sum: number; n: number }[] = []
  desired.forEach((d, i) => {
    blocks.push({ sum: d - i * gap, n: 1 })
    while (blocks.length > 1) {
      const b = blocks[blocks.length - 1]
      const a = blocks[blocks.length - 2]
      if (a.sum / a.n <= b.sum / b.n) break
      blocks.splice(blocks.length - 2, 2, { sum: a.sum + b.sum, n: a.n + b.n })
    }
  })
  const out: number[] = []
  for (const b of blocks) for (let k = 0; k < b.n; k++) out.push(b.sum / b.n + out.length * gap)
  return out
}

/**
 * Place the nodes of a spec by the layered layout (see `LayeredOptions`), returning a spec with every node's `x` and
 * `y` set and waypoints on edges that span several layers. Edges without a direction still count from `from` to `to`;
 * cycles are broken by ignoring the edges that close them.
 */
export function layeredLayout(spec: DiagramSpec): Omit<DiagramSpec, 'nodes'> & { nodes: PlacedNode[] } {
  const { direction = 'right', layerGap = 2, nodeGap = 1.6, sweeps = 8, compactSources = true } = spec.layered ?? {}
  const ids = spec.nodes.map((n) => n.id)
  const index = new Map(ids.map((id, i) => [id, i]))
  const edges = (spec.edges ?? [])
    .map((e, k) => ({ k, a: index.get(endId(e.from)), b: index.get(endId(e.to)) }))
    .filter((e): e is { k: number; a: number; b: number } => {
      if (e.a === undefined || e.b === undefined) throw new Error(`diagram edge ${spec.edges![e.k].from}: unknown node`)
      return e.a !== e.b
    })

  // Break cycles: a depth-first back edge (to a node still on the stack) closes a cycle and is left out of the layering.
  const graph = fromEdges(
    ids.length,
    edges.map((e) => [e.a, e.b] as const),
  )
  const { edgeClass } = depthFirstSearch(graph)
  const dag = edges.filter((_, i) => edgeClass[i] !== 'back')
  const preds: number[][] = ids.map(() => [])
  const succs: number[][] = ids.map(() => [])
  for (const e of dag) {
    preds[e.b].push(e.a)
    succs[e.a].push(e.b)
  }

  // Layers: longest path from the sources, visiting nodes in Kahn's topological order, unless pinned.
  const layer = ids.map(() => 0)
  const { order: kahnOrder } = topologicalSort(
    fromEdges(
      ids.length,
      dag.map((e) => [e.a, e.b] as const),
    ),
  )
  const order = Array.from(kahnOrder.data)
  for (const v of order) {
    const pinned = spec.nodes[v].layer
    if (pinned !== undefined) layer[v] = pinned
    for (const w of succs[v]) layer[w] = Math.max(layer[w], layer[v] + 1)
  }
  if (compactSources)
    for (const v of order)
      if (preds[v].length === 0 && succs[v].length > 0 && spec.nodes[v].layer === undefined)
        layer[v] = Math.max(layer[v], Math.min(...succs[v].map((w) => layer[w])) - 1)

  // Virtual nodes split every edge that spans more than one layer, so it can be ordered and routed layer by layer.
  type Slot = { node?: number; chain?: number }
  const slots: Slot[] = ids.map((_, i) => ({ node: i }))
  const slotLayer = [...layer]
  const links: [number, number][] = []
  const chains: { edge: number; slots: number[]; flipped: boolean }[] = []
  for (const e of edges) {
    const flipped = layer[e.a] > layer[e.b]
    const [lo, hi] = flipped ? [e.b, e.a] : [e.a, e.b]
    const span = layer[hi] - layer[lo]
    if (span === 0) continue
    let prev = lo
    const chain: number[] = []
    for (let l = layer[lo] + 1; l < layer[hi]; l++) {
      const s = slots.length
      slots.push({ chain: chains.length })
      slotLayer.push(l)
      links.push([prev, s])
      chain.push(s)
      prev = s
    }
    links.push([prev, hi])
    chains.push({ edge: e.k, slots: chain, flipped })
  }
  const layers: number[][] = []
  slotLayer.forEach((l, s) => (layers[l] ??= []).push(s))
  for (let l = 0; l < layers.length; l++) layers[l] ??= []
  const up: number[][] = slots.map(() => [])
  const down: number[][] = slots.map(() => [])
  for (const [a, b] of links) {
    down[a].push(b)
    up[b].push(a)
  }

  // Order each layer by barycentre sweeps, keeping the order with the fewest crossings.
  const pos = new Map<number, number>()
  const setPositions = (rows: number[][]) => rows.forEach((row) => row.forEach((s, i) => pos.set(s, i)))
  const total = (rows: number[][]) => {
    setPositions(rows)
    let c = 0
    for (let l = 0; l + 1 < rows.length; l++)
      c += crossings(rows[l].flatMap((a) => down[a].map((b): [number, number] => [pos.get(a)!, pos.get(b)!])))
    return c
  }
  let rows = layers.map((r) => [...r])
  let best = rows.map((r) => [...r])
  let bestCount = total(rows)
  const reorder = (row: number[], neighbours: number[][]) => {
    const bary = new Map(
      row.map((s, i) => {
        const ns = neighbours[s]
        return [s, ns.length ? ns.reduce((acc, n) => acc + pos.get(n)!, 0) / ns.length : i] as const
      }),
    )
    return [...row].sort((a, b) => bary.get(a)! - bary.get(b)!)
  }
  for (let it = 0; it < sweeps && bestCount > 0; it++) {
    const downward = it % 2 === 0
    const ls = rows.map((_, l) => l)
    for (const l of downward ? ls.slice(1) : ls.slice(0, -1).reverse()) {
      setPositions(rows)
      rows[l] = reorder(rows[l], downward ? up : down)
    }
    const count = total(rows)
    if (count < bestCount) {
      bestCount = count
      best = rows.map((r) => [...r])
    }
  }
  rows = best

  // Positions across each layer: start centred, then pull each node towards its neighbours' mean, keeping order and gap.
  const across = new Map<number, number>()
  for (const row of rows) row.forEach((s, i) => across.set(s, (i - (row.length - 1) / 2) * nodeGap))
  for (let it = 0; it < 6; it++) {
    const ls = rows.map((_, l) => l)
    for (const l of it % 2 === 0 ? ls : ls.reverse()) {
      const row = rows[l]
      const desired = row.map((s) => {
        const ns = [...up[s], ...down[s]]
        return ns.length ? ns.reduce((acc, n) => acc + across.get(n)!, 0) / ns.length : across.get(s)!
      })
      spaced(desired, nodeGap).forEach((y, i) => across.set(row[i], y))
    }
  }
  // Centre the whole layout on zero across the layers.
  const values = [...across.values()]
  const mid = (Math.min(...values) + Math.max(...values)) / 2
  const place = (s: number): [number, number] => {
    const along = slotLayer[s] * layerGap
    const a = across.get(s)! - mid
    return direction === 'right' ? [along, a] : [a, along]
  }

  const via = new Map<number, [number, number][]>()
  for (const c of chains) {
    if (!c.slots.length) continue
    const pts = c.slots.map(place)
    via.set(c.edge, c.flipped ? pts.reverse() : pts)
  }
  return {
    ...spec,
    layout: 'manual',
    nodes: spec.nodes.map((n, i) => {
      const [x, y] = place(i)
      return { ...n, x, y }
    }),
    edges: spec.edges?.map((e, k): DiagramEdge => (e.via || !via.has(k) ? e : { ...e, via: via.get(k) })),
  }
}
