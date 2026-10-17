// The heterogeneous graph of the note's figure: 8 authors, 8 papers and 4 venues with two features each, and three
// relations into papers. Paper P1 and its neighbours are the worked example.
import type { Graph } from 'aifn-compute/graph'

export const AUTHORS = 8
export const PAPERS = 8
export const VENUES = 4
export const N = AUTHORS + PAPERS + VENUES

/** Node index of author a, paper p and venue v (1-based names, 0-based indices). */
export const author = (a: number) => a - 1
export const paper = (p: number) => AUTHORS + p - 1
export const venue = (v: number) => AUTHORS + PAPERS + v - 1

export const NAMES = [
  ...Array.from({ length: AUTHORS }, (_, i) => `A${i + 1}`),
  ...Array.from({ length: PAPERS }, (_, i) => `P${i + 1}`),
  ...Array.from({ length: VENUES }, (_, i) => `V${i + 1}`),
]

export const FEATURES: [number, number][] = [
  // authors A1..A8
  [0, 1],
  [1, 1],
  [2, 1],
  [0, 2],
  [1, 0],
  [2, 2],
  [0.5, 1.5],
  [1.5, 0.5],
  // papers P1..P8
  [1, 0],
  [2, 0],
  [0, 2],
  [1, 1],
  [2, 1],
  [0, 1],
  [1, 2],
  [2, 2],
  // venues V1..V4
  [1, -1],
  [-1, 1],
  [2, 0],
  [0, -2],
]

/** The authors of each paper (written-by: author to paper). */
const WRITTEN_BY: Record<number, number[]> = {
  1: [1, 2],
  2: [2, 3],
  3: [4],
  4: [1, 5, 6],
  5: [3, 7],
  6: [8],
  7: [6, 7, 8],
  8: [4, 5],
}
/** The papers each paper cites (cites: cited paper to citing paper). */
const CITES: Record<number, number[]> = {
  1: [2, 3],
  2: [],
  3: [],
  4: [1, 2],
  5: [1, 3, 4],
  6: [5],
  7: [4, 6],
  8: [1, 5, 6, 7],
}
/** The venue of each paper (published-in: venue to paper). */
const PUBLISHED_IN: Record<number, number> = { 1: 1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 4, 7: 2, 8: 3 }

const directed = (edges: [number, number][]): Graph => ({
  kind: 'graph',
  nodes: N,
  edges: edges.map(([from, to]) => ({ from, to })),
  directed: true,
})

export const RELATION_NAMES = ['written by', 'cites', 'published in'] as const

export const RELATION_GRAPHS: Graph[] = [
  directed(Object.entries(WRITTEN_BY).flatMap(([p, as]) => as.map((a): [number, number] => [author(a), paper(+p)]))),
  directed(Object.entries(CITES).flatMap(([p, cs]) => cs.map((c): [number, number] => [paper(c), paper(+p)]))),
  directed(Object.entries(PUBLISHED_IN).map(([p, v]): [number, number] => [venue(v), paper(+p)])),
]

/**
 * The relation weights of the worked example, acting on column vectors as in the note (all four are symmetric, so the
 * row-convention weights are the same matrices): W0 = I, written by swaps the features, cites halves them, published in
 * keeps the first.
 */
export const SELF_WEIGHT = [
  [1, 0],
  [0, 1],
]
export const RELATION_WEIGHTS = [
  [
    [0, 1],
    [1, 0],
  ],
  [
    [0.5, 0],
    [0, 0.5],
  ],
  [
    [1, 0],
    [0, 0],
  ],
]

/** Layout: authors left, papers in a zigzag in the middle (so citation edges are visible), venues right. */
export const POSITIONS: [number, number][] = [
  ...Array.from({ length: AUTHORS }, (_, i): [number, number] => [0, 7 - i]),
  ...Array.from({ length: PAPERS }, (_, i): [number, number] => [i % 2 === 0 ? 2.1 : 2.9, 7 - i]),
  ...Array.from({ length: VENUES }, (_, i): [number, number] => [5, 6.2 - 1.8 * i]),
]
