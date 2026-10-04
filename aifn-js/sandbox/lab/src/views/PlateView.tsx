import { toDiagram, type DiagramOptions, type StructuredGraph } from 'aifn/graph/structured'
import { useMemo } from 'react'
import { Diagram, type DiagramSpec } from '@lab/diagram'

export type PlateViewProps = {
  /** A structured graph, e.g. an `aifn/inference/model` model description (a model is one). */
  graph: StructuredGraph
  /** Fixed centres (grid units) by node name, overriding the automatic layout. */
  positions?: DiagramOptions['positions']
  /** Node names drawn in the accent colour, e.g. the node a figure is about. */
  highlight?: readonly string[]
  /** Called with a node's name when it is clicked. */
  onNodeClick?: (name: string) => void
  /** Height in pixels, or `fill`; by default the Figure frame's height. */
  height?: number | 'fill'
  ariaLabel?: string
}

/**
 * A structured graph in plate notation, drawn from `toDiagram` (`aifn/graph/structured`): latent variables as circles,
 * observed ones shaded, deterministic ones dashed, parameters small; plates and templates as groups labelled with their
 * size, lagged edges labelled with their lag.
 */
export function PlateView({ graph, positions, highlight, onNodeClick, height, ariaLabel }: PlateViewProps) {
  const spec = useMemo((): DiagramSpec => {
    const d: DiagramSpec = toDiagram(graph, { positions })
    const on = new Set(highlight ?? [])
    return { ...d, nodes: d.nodes.map((n) => (on.has(n.id) ? { ...n, highlight: true } : n)) }
  }, [graph, positions, highlight])
  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel ?? `${graph.name ?? 'A graph'} in plate notation`}
      height={height}
      onNodeClick={onNodeClick}
    />
  )
}
