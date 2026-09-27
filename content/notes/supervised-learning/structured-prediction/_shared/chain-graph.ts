/**
 * Node and edge lists for drawing a stretch of a linear chain, positions n - 1, n and n + 1, in the notation of
 * Twomey, Diethe & Flach (2016): labels y_n, observations x_n, node potentials ψ_n and edge potentials Ψ_n.
 */
import type { GraphEdge, GraphNode } from '@/components/viz'

export const POSITIONS = ['n-1', 'n', 'n+1'] as const
const STEP = 1.6

export const at = (i: number) => i * STEP

/** The label nodes y_{n-1}, y_n, y_{n+1} on row `row`. */
export const labelNodes = (row = 0, kind: GraphNode['kind'] = 'variable'): GraphNode[] =>
  POSITIONS.map((p, i) => ({ id: `y${i}`, label: `y_{${p}}`, x: at(i), y: row, kind }))

/** "⋯" markers just outside the chain, and edges into and out of it, so the stretch reads as part of a longer chain. */
export function chainEnds(row: number, first: string, last: string, directed: boolean) {
  const nodes: GraphNode[] = [
    { id: 'more-left', label: '⋯', x: -0.9, y: row, kind: 'text', labelPosition: 'left' },
    { id: 'more-right', label: '⋯', x: at(2) + 0.9, y: row, kind: 'text', labelPosition: 'right' },
  ]
  const edges: GraphEdge[] = [
    { source: 'more-left', target: first, directed },
    { source: last, target: 'more-right', directed },
  ]
  return { nodes, edges }
}

/** The chain in potential form: node factors ψ_n below each label and edge factors Ψ_n between neighbours. */
export function potentialChain(): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const ys = labelNodes(0)
  const psi: GraphNode[] = POSITIONS.map((p, i) => ({
    id: `psi${i}`,
    label: `\\mathbf{ψ}_{${p}}`,
    x: at(i),
    y: 1,
    kind: 'factor',
    labelPosition: 'bottom',
  }))
  const Psi: GraphNode[] = [0, 1].map((i) => ({
    id: `Psi${i}`,
    label: `\\mathbf{Ψ}_{${POSITIONS[i]}}`,
    x: at(i) + STEP / 2,
    y: 0,
    kind: 'factor',
  }))
  const undirected = (source: string, target: string): GraphEdge => ({ source, target, directed: false })
  const ends = chainEnds(0, 'y0', 'y2', false)
  return {
    nodes: [...ys, ...psi, ...Psi, ...ends.nodes],
    edges: [
      ...[0, 1, 2].map((i) => undirected(`y${i}`, `psi${i}`)),
      ...[0, 1].flatMap((i) => [undirected(`y${i}`, `Psi${i}`), undirected(`Psi${i}`, `y${i + 1}`)]),
      ...ends.edges,
    ],
  }
}
