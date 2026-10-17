// The note's hypergraphs: six authors with features 1 to 6. "papers" has three papers, {1, 2, 3}, {3, 4} and
// {4, 5, 6}; "split" replaces the first paper by three two-author papers {1, 2}, {2, 3} and {1, 3}. Both have the same
// clique expansion.
import type { Hyperedges } from 'aifn-compute/graph'

export const AUTHORS = 6
export const FEATURES = [1, 2, 3, 4, 5, 6]

export type Structure = 'papers' | 'split'

/** Hyperedges with 0-based author indices. */
export const HYPEREDGES: Record<Structure, Hyperedges> = {
  papers: [
    [0, 1, 2],
    [2, 3],
    [3, 4, 5],
  ],
  split: [
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 3],
    [3, 4, 5],
  ],
}

export const EDGE_NAMES: Record<Structure, string[]> = {
  papers: ['e1', 'e2', 'e3'],
  split: ['e1a', 'e1b', 'e1c', 'e2', 'e3'],
}

export const POSITIONS: [number, number][] = [
  [0, 1],
  [0, -1],
  [1.6, 0],
  [3.2, 0],
  [4.8, 1],
  [4.8, -1],
]

/** A hyperedge is drawn at the centroid of its members, nudged off the line when it has two members. */
export function edgePosition(e: readonly number[]): [number, number] {
  const x = e.reduce((s, v) => s + POSITIONS[v][0], 0) / e.length
  const y = e.reduce((s, v) => s + POSITIONS[v][1], 0) / e.length
  return e.length === 2 && Math.abs(POSITIONS[e[0]][1] - POSITIONS[e[1]][1]) < 1e-9 ? [x, y + 0.45] : [x, y]
}
