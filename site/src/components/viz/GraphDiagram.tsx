import { useMemo } from 'react'
import { useTheme } from '@/components/theme-provider'
import { EChart } from './EChart'
import { chrome, seriesColor } from './palette'

/**
 * A node of a small hand-placed graph. `x` and `y` are layout units (y grows downwards); the diagram scales them to
 * fit. Random variables are circles, filled when `observed`; factors are small squares.
 */
export type GraphNode = {
  id: string
  label?: string
  x: number
  y: number
  kind?: 'variable' | 'observed' | 'factor'
}

export type GraphEdge = { source: string; target: string; directed?: boolean }

/**
 * A fixed-layout diagram of a graphical model, factor graph or computation graph. Nodes in `highlight` take the
 * first palette colour, which is how a figure marks the variables a sentence is about (a Markov blanket, a query).
 */
export function GraphDiagram({
  nodes,
  edges,
  highlight = [],
  height = 220,
  ariaLabel,
}: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  highlight?: string[]
  height?: number
  ariaLabel?: string
}) {
  const { resolved: mode } = useTheme()
  // Value dependencies: callers usually pass inline arrays, so key the memo on their contents.
  const key = JSON.stringify([nodes, edges, highlight])
  const option = useMemo(() => {
    const [n, e, h] = JSON.parse(key) as [GraphNode[], GraphEdge[], string[]]
    const c = chrome(mode)
    const accent = seriesColor(mode, 0)
    const marked = new Set(h)
    return {
      series: [
        {
          type: 'graph',
          layout: 'none',
          silent: true,
          animation: false,
          label: { show: true, color: c.ink, fontSize: 13 },
          lineStyle: { color: c.inkSecondary, width: 1.5, opacity: 1 },
          edgeSymbolSize: 8,
          data: n.map((node) => {
            const factor = node.kind === 'factor'
            const filled = node.kind === 'observed'
            const stroke = marked.has(node.id) ? accent : c.ink
            return {
              id: node.id,
              name: node.label ?? node.id,
              x: node.x,
              y: node.y,
              symbol: factor ? 'rect' : 'circle',
              symbolSize: factor ? 12 : 34,
              itemStyle: {
                color: factor ? stroke : filled ? c.grid : c.surface,
                borderColor: stroke,
                borderWidth: marked.has(node.id) ? 2.5 : 1.5,
              },
              label: { show: !factor, color: marked.has(node.id) ? accent : c.ink },
            }
          }),
          links: e.map((edge) => ({
            source: edge.source,
            target: edge.target,
            symbol: edge.directed === false ? ['none', 'none'] : ['none', 'arrow'],
          })),
        },
      ],
    }
  }, [key, mode])
  return <EChart option={option} height={height} cartesian={false} ariaLabel={ariaLabel} />
}
