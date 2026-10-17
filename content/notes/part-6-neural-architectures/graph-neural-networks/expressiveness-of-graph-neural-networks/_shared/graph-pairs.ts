/**
 * Small graphs used across the expressiveness notes, with drawing positions: pairs that the 1-WL test separates or
 * not, built as engine `Graph`s (undirected).
 */
import { fromEdges, type Graph } from 'aifn-compute/graph'

export type Drawn = {
  /** A short name for legends and readouts. */
  name: string
  graph: Graph
  /** Drawing position of each node. */
  pos: readonly (readonly [number, number])[]
  /** Edge list as pairs (the same edges as `graph`). */
  edges: readonly (readonly [number, number])[]
}

export type Pair = {
  id: string
  label: string
  left: Drawn
  right: Drawn
  /** One sentence on what the pair shows. */
  note: string
}

function drawn(name: string, pos: [number, number][], edges: [number, number][]): Drawn {
  return { name, graph: fromEdges(pos.length, edges, { directed: false }), pos, edges }
}

const ring = (n: number, cx = 0, cy = 0, r = 1, phase = Math.PI / 2): [number, number][] =>
  Array.from({ length: n }, (_, i) => [
    cx + r * Math.cos(phase + (2 * Math.PI * i) / n),
    cy + r * Math.sin(phase + (2 * Math.PI * i) / n),
  ])

const cycleEdges = (n: number, offset = 0): [number, number][] =>
  Array.from({ length: n }, (_, i) => [offset + i, offset + ((i + 1) % n)])

/** The 6-cycle. */
export const hexagon = drawn('6-cycle', ring(6), cycleEdges(6))

/** Two disjoint triangles. */
export const twoTriangles = drawn(
  'two triangles',
  [...ring(3, -1.1, 0, 0.8), ...ring(3, 1.1, 0, 0.8)],
  [...cycleEdges(3), ...cycleEdges(3, 3)],
)

/** The path on four nodes. */
export const path4 = drawn(
  'path',
  [
    [-1.5, 0],
    [-0.5, 0],
    [0.5, 0],
    [1.5, 0],
  ],
  [
    [0, 1],
    [1, 2],
    [2, 3],
  ],
)

/** The star with three leaves. */
export const star3 = drawn(
  'star',
  [[0, 0], ...ring(3, 0, 0, 1.2)],
  [
    [0, 1],
    [0, 2],
    [0, 3],
  ],
)

/**
 * Decalin's carbon skeleton: two hexagons sharing the edge 0-1 (nodes 0 and 1 have degree 3).
 */
export const decalin = (() => {
  const s = Math.sqrt(3) / 2
  const pos: [number, number][] = [
    [0, 0.5],
    [0, -0.5],
    [-s, -1],
    [-2 * s, -0.5],
    [-2 * s, 0.5],
    [-s, 1],
    [s, -1],
    [2 * s, -0.5],
    [2 * s, 0.5],
    [s, 1],
  ]
  const edges: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 0],
    [1, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 0],
  ]
  return drawn('decalin', pos, edges)
})()

/** Bicyclopentyl's carbon skeleton: two pentagons joined by the edge 0-5 (nodes 0 and 5 have degree 3). */
export const bicyclopentyl = (() => {
  const left = ring(5, -1.35, 0, 0.85, 0)
  const right = ring(5, 1.35, 0, 0.85, Math.PI)
  const edges: [number, number][] = [...cycleEdges(5), ...cycleEdges(5, 5), [0, 5]]
  return drawn('bicyclopentyl', [...left, ...right], edges)
})()

/** Node (x, y) of a 4 x 4 grid is node 4y + x. */
const grid4 = (x: number, y: number) => 4 * (((y % 4) + 4) % 4) + (((x % 4) + 4) % 4)

/** The 16-node graphs are drawn on a circle in grid order. */
const circle16 = ring(16, 0, 0, 1.4)

/** The 4 x 4 rook's graph: (x, y) is adjacent to every node in its row and its column. Strongly regular (16, 6, 2, 2). */
export const rook = (() => {
  const edges: [number, number][] = []
  for (let a = 0; a < 16; a++)
    for (let b = a + 1; b < 16; b++) {
      const [xa, ya, xb, yb] = [a % 4, Math.floor(a / 4), b % 4, Math.floor(b / 4)]
      if (xa === xb || ya === yb) edges.push([a, b])
    }
  return drawn("4 × 4 rook's graph", circle16, edges)
})()

/** The Shrikhande graph: (x, y) is adjacent to (x ± 1, y), (x, y ± 1) and (x ± 1, y ± 1), mod 4. Also (16, 6, 2, 2). */
export const shrikhande = (() => {
  const set = new Set<string>()
  const edges: [number, number][] = []
  for (let y = 0; y < 4; y++)
    for (let x = 0; x < 4; x++)
      for (const [dx, dy] of [
        [1, 0],
        [0, 1],
        [1, 1],
      ]) {
        const a = grid4(x, y)
        const b = grid4(x + dx, y + dy)
        const k = `${Math.min(a, b)}-${Math.max(a, b)}`
        if (!set.has(k)) {
          set.add(k)
          edges.push([Math.min(a, b), Math.max(a, b)])
        }
      }
  return drawn('Shrikhande graph', circle16, edges)
})()

export const PAIRS: readonly Pair[] = [
  {
    id: 'path-star',
    label: 'path and star',
    left: path4,
    right: star3,
    note: 'Four nodes and three edges each; the degrees differ, so one round separates them.',
  },
  {
    id: 'hexagon-triangles',
    label: '6-cycle and two triangles',
    left: hexagon,
    right: twoTriangles,
    note: 'Both graphs are 2-regular on six nodes, so every node keeps one colour.',
  },
  {
    id: 'decalin-bicyclopentyl',
    label: 'decalin and bicyclopentyl',
    left: decalin,
    right: bicyclopentyl,
    note: 'Ten nodes, eleven edges and two adjacent degree-3 nodes each; rings of six against rings of five.',
  },
  {
    id: 'rook-shrikhande',
    label: "rook's and Shrikhande graphs",
    left: rook,
    right: shrikhande,
    note: 'Two strongly regular graphs with parameters (16, 6, 2, 2); they defeat 1-WL and 2-FWL.',
  },
]

export const pairById = (id: string): Pair => PAIRS.find((p) => p.id === id) ?? PAIRS[0]
