/**
 * Layers that draw a small graph in a bare `Plot`: edges as segments, nodes as points grouped by class (a colour
 * class of colour refinement, an encoding class), so equal classes share a colour and a marker in every panel.
 */
import type { ReactElement } from 'react'
import { Points, Segments } from 'aifn-render'
import type { Drawn } from './graph-pairs'

export type GraphLayerOptions = {
  /** A class per node; class k takes palette slot k in every panel. Absent: all nodes muted. */
  classes?: ArrayLike<number>
  /** Legend names of the classes. */
  classNames?: readonly string[]
  /** Text beside each node. */
  labels?: readonly string[]
  /** Edges (as indices into `d.edges`) drawn dashed and muted, e.g. a deleted edge. */
  removedEdges?: ReadonlySet<number>
  /** Nodes ringed in ink, e.g. a deleted node or an ego root. */
  marked?: readonly number[]
  /** Series id prefix, unique per panel. */
  id: string
}

/** The axis range of a drawing, padded by `pad` on every side. */
export function drawingRange(ds: readonly Drawn[], pad = 0.45): { x: [number, number]; y: [number, number] } {
  const xs = ds.flatMap((d) => d.pos.map((p) => p[0]))
  const ys = ds.flatMap((d) => d.pos.map((p) => p[1]))
  return {
    x: [Math.min(...xs) - pad, Math.max(...xs) + pad],
    y: [Math.min(...ys) - pad, Math.max(...ys) + pad],
  }
}

/** The layers of one graph drawing, edges first. */
export function graphLayers(d: Drawn, options: GraphLayerOptions): ReactElement[] {
  const kept = d.edges.flatMap((e, k) => (options.removedEdges?.has(k) ? [] : [e]))
  const removed = d.edges.flatMap((e, k) => (options.removedEdges?.has(k) ? [e] : []))
  const seg = (es: readonly (readonly [number, number])[]) =>
    es.map(([a, b]) => ({ from: d.pos[a] as readonly [number, number], to: d.pos[b] as readonly [number, number] }))
  const x = d.pos.map((p) => p[0])
  const y = d.pos.map((p) => p[1])
  const layers: ReactElement[] = [
    <Segments key={`${options.id}-edges`} id={`${options.id}-edges`} name="edges" segments={seg(kept)} width={1.5} />,
  ]
  if (removed.length > 0) {
    layers.push(
      <Segments
        key={`${options.id}-removed`}
        id={`${options.id}-removed`}
        name="removed edge"
        segments={seg(removed)}
        dashed
        muted
      />,
    )
  }
  const marked = options.marked ?? []
  if (marked.length > 0) {
    layers.push(
      <Points
        key={`${options.id}-marked`}
        id={`${options.id}-marked`}
        name="marked node"
        x={marked.map((v) => x[v])}
        y={marked.map((v) => y[v])}
        emphasis
        size={22}
      />,
    )
  }
  layers.push(
    options.classes ? (
      <Points
        key={`${options.id}-nodes`}
        id={`${options.id}-nodes`}
        name="nodes"
        x={x}
        y={y}
        group={options.classes}
        groupNames={options.classNames}
        labels={options.labels}
        size={14}
      />
    ) : (
      <Points
        key={`${options.id}-nodes`}
        id={`${options.id}-nodes`}
        name="nodes"
        x={x}
        y={y}
        muted
        labels={options.labels}
        size={14}
      />
    ),
  )
  return layers
}
