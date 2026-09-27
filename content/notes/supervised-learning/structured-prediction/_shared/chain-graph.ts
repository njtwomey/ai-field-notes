/**
 * Diagram parts for a stretch of a linear chain, positions n - 1, n and n + 1, in the notation of
 * Twomey, Diethe & Flach (2016): labels y_n, observations x_n, node potentials ψ_n and edge potentials Ψ_n.
 */
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramEdge, DiagramNode } from '@/components/diagram/types'

export const POSITIONS = ['n-1', 'n', 'n+1'] as const
const STEP = 2

export const at = (i: number) => i * STEP

type Part = { nodes: DiagramNode[]; edges: DiagramEdge[] }

/** The label nodes y_{n-1}, y_n, y_{n+1} on row `row`. */
export const labelNodes = (row = 0): DiagramNode[] =>
  POSITIONS.map((p, i) => variable(`y${i}`, at(i), row, `$y_{${p}}$`, { w: 1.05, h: 1.05 }))

/** "⋯" markers just outside the chain, and edges into and out of it, so the stretch reads as part of a longer chain. */
export function chainEnds(row: number, first: string, last: string, directed: boolean): Part {
  const dots = (id: string, x: number): DiagramNode => ({ id, x, y: row, shape: 'text', w: 0.5, label: '$\\cdots$' })
  return {
    nodes: [dots('more-left', -1.3), dots('more-right', at(2) + 1.3)],
    edges: [link('more-left', first, directed), link(last, 'more-right', directed)],
  }
}

/** The chain in potential form: node factors ψ_n below each label and edge factors Ψ_n between neighbours. */
export function potentialChain(): Part {
  const psi = POSITIONS.map((p, i) => factor(`psi${i}`, at(i), 1.4, `$\\psivec_{${p}}$`, 's'))
  const Psi = [0, 1].map((i) => factor(`Psi${i}`, at(i) + STEP / 2, 0, `$\\Psimat_{${POSITIONS[i]}}$`, 'n'))
  const ends = chainEnds(0, 'y0', 'y2', false)
  return {
    nodes: [...labelNodes(0), ...psi, ...Psi, ...ends.nodes],
    edges: [
      ...[0, 1, 2].map((i) => link(`y${i}`, `psi${i}`, false)),
      ...[0, 1].flatMap((i) => [link(`y${i}`, `Psi${i}`, false), link(`Psi${i}`, `y${i + 1}`, false)]),
      ...ends.edges,
    ],
  }
}
