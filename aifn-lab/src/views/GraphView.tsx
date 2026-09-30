import type { Graph } from 'aifn/graph'
import { useMemo } from 'react'
import {
  circleLayout,
  Diagram,
  forceLayout,
  type DiagramEdge,
  type DiagramNode,
  type DiagramSpec,
  type ElementState,
  type LayeredOptions,
  type Point,
  type Side,
  type Tone,
} from '@lab/diagram'
import { formatValue } from './format'

/** A per-node or per-edge value: an array indexed like the nodes (edges), or a function of the index. */
export type PerItem<T> = readonly (T | undefined)[] | ((i: number) => T | undefined)

const at = <T,>(p: PerItem<T> | undefined, i: number): T | undefined =>
  p === undefined ? undefined : typeof p === 'function' ? p(i) : p[i]

export type GraphViewProps = {
  /** An `aifn/graph` graph. Node labels come from `graph.labels`, else the node index. */
  graph: Graph
  ariaLabel?: string
  /**
   * Where nodes go: `layered` (columns by topological depth, for directed graphs), `force` (a deterministic
   * force-directed layout), `circle`, or explicit positions in grid units (one per node). Default: layered when
   * directed, force when undirected.
   */
  layout?: 'layered' | 'force' | 'circle' | readonly Point[]
  layered?: LayeredOptions
  /** Multiplies positions to space the drawing out (see `DiagramSpec.spread`). */
  spread?: number | [number, number]
  /** Step state of each node and edge (`idle` dimmed, `active` emphasised, `done` plain). */
  nodeState?: PerItem<ElementState>
  edgeState?: PerItem<ElementState>
  /** Palette slot (or tone) of each node and edge. */
  nodeTone?: PerItem<Tone>
  edgeTone?: PerItem<Tone>
  /** Accent-coloured nodes and edges, e.g. the current node. */
  nodeHighlight?: PerItem<boolean>
  edgeHighlight?: PerItem<boolean>
  /** Node labels (TeX in `$…$` allowed), overriding the graph's. */
  nodeLabels?: PerItem<string>
  /** Annotations beside each node, per side, e.g. `{ n: '$d = 3$' }`. */
  nodeNotes?: PerItem<Partial<Record<Side, string>>>
  /** Edge labels; with `showWeights`, unlabelled edges show their weight. */
  edgeLabels?: PerItem<string>
  showWeights?: boolean
  /** A chip riding on each edge, e.g. a flow. */
  edgeNotes?: PerItem<string>
  edgeDashed?: PerItem<boolean>
  /** Further overrides of each node's or edge's diagram spec. */
  node?: (v: number) => Partial<DiagramNode>
  edge?: (k: number) => Partial<DiagramEdge>
  /** Height in pixels, or `fill` (the parent needs a definite height); by default the Figure frame's height. */
  height?: number | 'fill'
  onNodeClick?: (v: number) => void
}

const id = (v: number) => `n${v}`

/**
 * An `aifn/graph` graph drawn with the lab's diagram system: circles for nodes, straight edges (arrows when directed,
 * curved apart when two edges join the same pair), with optional per-node and per-edge state, tone, labels and notes
 * for stepping through an algorithm.
 */
export function GraphView({
  graph,
  ariaLabel = 'A graph',
  layout,
  layered,
  spread,
  nodeState,
  edgeState,
  nodeTone,
  edgeTone,
  nodeHighlight,
  edgeHighlight,
  nodeLabels,
  nodeNotes,
  edgeLabels,
  showWeights,
  edgeNotes,
  edgeDashed,
  node,
  edge,
  height,
  onNodeClick,
}: GraphViewProps) {
  const directed = graph.directed !== false
  const kind = layout ?? (directed ? 'layered' : 'force')
  const positions = useMemo((): readonly Point[] | null => {
    if (typeof kind !== 'string') return kind
    if (kind === 'force') return forceLayout(graph.nodes, graph.edges)
    if (kind === 'circle') return circleLayout(graph.nodes)
    return null
  }, [kind, graph])

  const spec = useMemo((): DiagramSpec => {
    // Edges joining the same pair of nodes (either way round) are bowed apart.
    const pairs = new Map<string, number[]>()
    graph.edges.forEach((e, k) => {
      const key = e.from < e.to ? `${e.from},${e.to}` : `${e.to},${e.from}`
      pairs.set(key, [...(pairs.get(key) ?? []), k])
    })
    const nodes = Array.from({ length: graph.nodes }, (_, v): DiagramNode => {
      const label = at(nodeLabels, v) ?? graph.labels?.[v] ?? String(v)
      return {
        id: id(v),
        shape: 'circle',
        label,
        ...(positions && { x: positions[v].x, y: positions[v].y }),
        tone: at(nodeTone, v) ?? 'ink',
        state: at(nodeState, v),
        highlight: at(nodeHighlight, v),
        notes: at(nodeNotes, v),
        ...node?.(v),
      }
    })
    const edges = graph.edges.map((e, k): DiagramEdge => {
      const key = e.from < e.to ? `${e.from},${e.to}` : `${e.to},${e.from}`
      const group = pairs.get(key)!
      const bend = group.length > 1 && e.from !== e.to ? (group.indexOf(k) - (group.length - 1) / 2) * 0.8 : 0
      // A bend is relative to the direction of travel: flip it for edges running the other way so they separate.
      const signed = e.from < e.to ? bend : -bend
      const label = at(edgeLabels, k) ?? (showWeights ? formatValue(e.weight ?? 1) : undefined)
      const note = at(edgeNotes, k)
      return {
        from: id(e.from),
        to: id(e.to),
        route: bend !== 0 ? 'curve' : 'straight',
        ...(bend !== 0 && { bend: signed }),
        arrow: directed ? 'end' : 'none',
        label,
        ...(note !== undefined && { note: { text: note } }),
        tone: at(edgeTone, k),
        state: at(edgeState, k),
        highlight: at(edgeHighlight, k),
        dashed: at(edgeDashed, k),
        ...edge?.(k),
      }
    })
    return {
      nodes,
      edges,
      ...(positions ? {} : { layout: 'layered' as const, layered: { direction: 'right' as const, ...layered } }),
      ...(spread !== undefined && { spread }),
    }
  }, [
    graph,
    positions,
    directed,
    layered,
    spread,
    nodeState,
    edgeState,
    nodeTone,
    edgeTone,
    nodeHighlight,
    edgeHighlight,
    nodeLabels,
    nodeNotes,
    edgeLabels,
    showWeights,
    edgeNotes,
    edgeDashed,
    node,
    edge,
  ])

  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel}
      height={height}
      onNodeClick={onNodeClick && ((s) => onNodeClick(Number(s.slice(1))))}
    />
  )
}
