import { toPlateDiagram, type Model, type PlateDiagramOptions } from 'aifn/pgm'
import { useMemo } from 'react'
import { Diagram, type DiagramSpec } from '@lab/diagram'

export type PlateViewProps = {
  /** An `aifn/pgm` model description. */
  model: Model
  /** Fixed centres (grid units) by node name, overriding the automatic layout. */
  positions?: PlateDiagramOptions['positions']
  /** Node names drawn in the accent colour, e.g. the node a figure is about. */
  highlight?: readonly string[]
  /** Called with a node's name when it is clicked. */
  onNodeClick?: (name: string) => void
  /** Height in pixels, or `fill`; by default the Figure frame's height. */
  height?: number | 'fill'
  ariaLabel?: string
}

/**
 * A model description in plate notation, from `toPlateDiagram`: latent variables as circles, observed ones shaded,
 * deterministic ones dashed, constants small; plates as groups labelled with their size.
 */
export function PlateView({ model, positions, highlight, onNodeClick, height, ariaLabel }: PlateViewProps) {
  const spec = useMemo((): DiagramSpec => {
    const d = toPlateDiagram(model, { positions })
    const on = new Set(highlight ?? [])
    return { ...d, nodes: d.nodes.map((n) => (on.has(n.id) ? { ...n, highlight: true } : n)) }
  }, [model, positions, highlight])
  return (
    <Diagram
      spec={spec}
      ariaLabel={ariaLabel ?? `${model.name} in plate notation`}
      height={height}
      onNodeClick={onNodeClick}
    />
  )
}
