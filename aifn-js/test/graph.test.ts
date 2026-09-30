import { describe, expect, it } from 'vitest'
import {
  aStar,
  bellmanFord,
  bipartite,
  breadthFirstSearch,
  breadthFirstSteps,
  condensation,
  connectedComponents,
  createHeap,
  depthFirstSearch,
  depthFirstSteps,
  dijkstra,
  findCycle,
  findRoot,
  floydWarshall,
  fromAdjacency,
  fromEdges,
  fromMatrix,
  heapPop,
  heapPush,
  inDegree,
  isDag,
  iterativeDeepening,
  maxFlow,
  minimumSpanningTree,
  neighbours,
  outDegree,
  path,
  reverse,
  stronglyConnectedComponents,
  subgraph,
  toMatrix,
  topologicalSort,
  unionFind,
  unite,
  unweightedShortestPaths,
  type Graph,
  type TraversalOptions,
} from 'aifn/graph'
import { stream } from 'aifn/random'
import { toFlat, toRows } from 'aifn/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/trace'

const names = (s: string) => s.split('')
const byName = (labels: string[]) => (x: string) => labels.indexOf(x)

/** A graph from edges written as two-letter strings, 'uv' for u → v, over the given node letters. */
function lettered(nodes: string, edges: string[], directed = true): Graph {
  const at = byName(names(nodes))
  return fromEdges(
    nodes.length,
    edges.map((e) => [at(e[0]), at(e[1])] as const),
    { directed, labels: names(nodes) },
  )
}

/** A random graph on n nodes with each ordered pair present with probability p and weights in [0, 10). */
function randomGraph(seed: number, n: number, p: number, directed = true): Graph {
  const s = stream(seed)
  const edges: [number, number, number][] = []
  for (let i = 0; i < n; i++)
    for (let j = directed ? 0 : i + 1; j < n; j++)
      if (i !== j && s.uniform() < p) edges.push([i, j, Math.floor(10 * s.uniform())])
  return fromEdges(n, edges, { directed })
}

describe('representation', () => {
  it('builds the same graph from edges, adjacency lists and a matrix', () => {
    const a = fromEdges(3, [
      [0, 1, 2],
      [0, 2, 5],
      [1, 2, 1],
    ])
    const b = fromAdjacency([[1, 2], [2], []], { weights: [[2, 5], [1], []] })
    const c = fromMatrix([
      [0, 2, 5],
      [0, 0, 1],
      [0, 0, 0],
    ])
    expect(toRows(toMatrix(a))).toEqual(toRows(toMatrix(b)))
    expect(toRows(toMatrix(a))).toEqual(toRows(toMatrix(c)))
    expect(toFlat(neighbours(a, 0))).toEqual([1, 2])
    expect(toFlat(outDegree(a))).toEqual([2, 1, 0])
    expect(toFlat(inDegree(a))).toEqual([0, 1, 2])
    expect(toFlat(inDegree(reverse(a)))).toEqual([2, 1, 0])
    const sub = subgraph(a, [2, 1])
    expect(sub.edges).toEqual([{ from: 1, to: 0, weight: 1 }])
    expect(sub.labels).toEqual(['2', '1'])
    // Undirected adjacency lists may list a pair from both ends; the matrix is symmetric.
    const u = fromAdjacency([[1], [0, 2], [1]], { directed: false })
    expect(u.edges.length).toBe(2)
    expect(toRows(toMatrix(u))).toEqual([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ])
    expect(toFlat(path([-1, 0, 1, 1], 3))).toEqual([0, 1, 3])
    expect(() => fromEdges(2, [[0, 2]])).toThrow()
  })

  it('heap pops in priority order (ties first-in) and union–find merges sets', () => {
    const h = createHeap<string>()
    ;[5, 1, 4, 1, 3].forEach((p, k) => heapPush(h, `${p}${'abcde'[k]}`, p))
    const out: string[] = []
    for (let e = heapPop(h); e; e = heapPop(h)) out.push(e.value)
    expect(out).toEqual(['1b', '1d', '3e', '4c', '5a'])
    const uf = unionFind(5)
    expect(unite(uf, 0, 1)).toBe(true)
    expect(unite(uf, 3, 4)).toBe(true)
    expect(unite(uf, 1, 0)).toBe(false)
    expect(uf.count).toBe(3)
    expect(findRoot(uf, 0)).toBe(findRoot(uf, 1))
    expect(findRoot(uf, 2)).not.toBe(findRoot(uf, 3))
  })
})

describe('traversal', () => {
  // CLRS Figure 22.3: breadth-first search from s.
  const clrs223 = lettered('rstuvwxy', ['rs', 'rv', 'sw', 'wt', 'wx', 'tx', 'tu', 'xu', 'xy', 'uy'], false)
  // CLRS Figure 22.4: depth-first search, neighbours in alphabetical order.
  const clrs224 = lettered('uvwxyz', ['uv', 'ux', 'vy', 'wy', 'wz', 'xv', 'yx', 'zz'])

  it('breadth-first search gives CLRS 22.3 layers', () => {
    const at = byName(names('rstuvwxy'))
    const r = breadthFirstSearch(clrs223, at('s'))
    const letters = (t: ArrayLike<number>) => Array.from(t, (v) => 'rstuvwxy'[v]).join('')
    expect(r.layers.map((l) => letters(l.data))).toEqual(['s', 'rw', 'vtx', 'uy'])
    expect(toFlat(r.depth)).toEqual([1, 0, 2, 3, 2, 1, 2, 3])
    expect(letters(path(r.parent, at('y')).data)).toEqual('swxy')
    expect(toFlat(unweightedShortestPaths(clrs223, at('s')).distance)).toEqual([1, 0, 2, 3, 2, 1, 2, 3])
  })

  it('depth-first search gives CLRS 22.4 times and edge classes', () => {
    const r = depthFirstSearch(clrs224)
    expect(toFlat(r.discovery)).toEqual([1, 2, 9, 4, 3, 10])
    expect(toFlat(r.finish)).toEqual([8, 7, 12, 5, 6, 11])
    // Edges: uv ux vy wy wz xv yx zz.
    expect(r.edgeClass).toEqual(['tree', 'forward', 'tree', 'cross', 'tree', 'back', 'tree', 'back'])
    expect(toFlat(r.preorder)).toEqual([0, 1, 4, 3, 2, 5])
    expect(toFlat(r.postorder)).toEqual([3, 4, 1, 0, 5, 2])
    // An undirected graph has only tree and back edges.
    const u = depthFirstSearch(clrs223)
    expect(new Set(u.edgeClass)).toEqual(new Set(['tree', 'back']))
    expect(u.edgeClass.filter((c) => c === 'tree').length).toBe(7)
  })

  it('iterative deepening finds a shortest path and the breadth-first depths', () => {
    const at = byName(names('rstuvwxy'))
    const r = iterativeDeepening(clrs223, at('s'), at('u'))
    expect(r.found).toBe(true)
    expect(r.path.shape[0]).toBe(4)
    expect(r.limit).toBe(3)
    const all = iterativeDeepening(clrs223, at('s'))
    expect(toFlat(all.depth)).toEqual(toFlat(breadthFirstSearch(clrs223, at('s')).depth))
  })

  function protocol<S extends { current: number }>(alg: Algorithm<TraversalOptions, S>) {
    const opts = { graph: clrs224 }
    const full = trace(alg, opts, 100)
    expect(full.meta.stopped).toBe('done')
    expect(trace(alg, opts, 100).steps.at(-1)).toEqual(full.steps.at(-1))
    expect(seek(alg, opts, 7, { checkpoints: trace(alg, opts, 12, { checkpointEvery: 5 }) })).toEqual(run(alg, opts, 7))
    const short = trace(alg, opts, 6, { record: { current: (s) => s.current } })
    const long = trace(alg, opts, 15, { record: { current: (s) => s.current } })
    expect(toFlat(extend(short, alg, opts, 9).series.current)).toEqual(toFlat(long.series.current))
  }

  it('follows the trace protocol', () => {
    protocol(breadthFirstSteps)
    protocol(depthFirstSteps)
  })
})

describe('order and cycles', () => {
  it('topological orders are valid, and cycles are found', () => {
    for (let seed = 0; seed < 10; seed++) {
      // A random DAG: edges only from lower to higher index, then relabelled by a permutation.
      const g0 = randomGraph(seed, 9, 0.3)
      const perm = [4, 7, 0, 2, 8, 1, 6, 3, 5]
      const dag = fromEdges(
        9,
        g0.edges.filter((e) => e.from < e.to).map((e) => [perm[e.from], perm[e.to]] as const),
      )
      expect(isDag(dag)).toBe(true)
      for (const method of ['kahn', 'depth-first'] as const) {
        const { order, cycle } = topologicalSort(dag, { method })
        expect(cycle).toBeNull()
        const rank = new Map(Array.from(order.data, (v, i) => [v, i]))
        expect(rank.size).toBe(9)
        for (const e of dag.edges) expect(rank.get(e.from)!).toBeLessThan(rank.get(e.to)!)
      }
    }
    const cyclic = fromEdges(5, [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 1],
      [3, 4],
    ])
    expect(isDag(cyclic)).toBe(false)
    expect(toFlat(findCycle(cyclic)!)).toEqual([1, 2, 3])
    const sorted = topologicalSort(cyclic)
    expect(toFlat(sorted.order)).toEqual([0])
    expect(sorted.cycle).not.toBeNull()
    expect(toFlat(findCycle(fromEdges(2, [[1, 1]]))!)).toEqual([1])
  })
})

describe('components', () => {
  // CLRS Figure 22.9.
  const clrs229 = lettered('abcdefgh', [
    'ab',
    'bc',
    'be',
    'bf',
    'cd',
    'cg',
    'dc',
    'dh',
    'ea',
    'ef',
    'fg',
    'gf',
    'gh',
    'hh',
  ])
  const partitionOf = (members: { data: ArrayLike<number> }[]) =>
    members
      .map((m) =>
        Array.from(m.data, (v) => 'abcdefgh'[v])
          .sort()
          .join(''),
      )
      .sort()

  it('Tarjan and Kosaraju find the CLRS 22.9 components', () => {
    const tarjan = stronglyConnectedComponents(clrs229)
    const kosaraju = stronglyConnectedComponents(clrs229, { method: 'kosaraju' })
    expect(partitionOf(tarjan.members)).toEqual(['abe', 'cd', 'fg', 'h'])
    expect(partitionOf(kosaraju.members)).toEqual(['abe', 'cd', 'fg', 'h'])
    // Tarjan numbers components in reverse topological order, Kosaraju in topological order.
    expect(partitionOf([tarjan.members[0]])).toEqual(['h'])
    expect(partitionOf([kosaraju.members[0]])).toEqual(['abe'])
    const c = condensation(clrs229)
    expect(isDag(c.graph)).toBe(true)
    expect(c.graph.edges.length).toBe(5)
    expect(c.graph.labels).toContain('a,b,e')
  })

  it('connected components ignore direction', () => {
    const g = fromEdges(6, [
      [0, 1],
      [2, 1],
      [3, 4],
    ])
    const r = connectedComponents(g)
    expect(r.count).toBe(3)
    expect(toFlat(r.labels)).toEqual([0, 0, 0, 1, 1, 2])
  })

  it('bipartite graphs get a two-colouring, others an odd cycle', () => {
    const square = fromEdges(
      4,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
      ],
      { directed: false },
    )
    const r = bipartite(square)
    expect(r.bipartite).toBe(true)
    if (r.bipartite) for (const e of square.edges) expect(r.colour.data[e.from]).not.toBe(r.colour.data[e.to])
    const pentagon = fromEdges(
      6,
      [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0],
        [4, 5],
      ],
      { directed: false },
    )
    const p = bipartite(pentagon)
    expect(p.bipartite).toBe(false)
    if (!p.bipartite) {
      const cycle = toFlat(p.oddCycle)
      expect(cycle.length % 2).toBe(1)
      const adjacent = (a: number, b: number) =>
        pentagon.edges.some((e) => (e.from === a && e.to === b) || (e.from === b && e.to === a))
      cycle.forEach((v, i) => expect(adjacent(v, cycle[(i + 1) % cycle.length])).toBe(true))
    }
  })
})

describe('shortest paths', () => {
  it('Dijkstra agrees with Bellman–Ford and Floyd–Warshall on random graphs', () => {
    for (let seed = 0; seed < 20; seed++) {
      const g = randomGraph(100 + seed, 10, 0.25, seed % 2 === 0)
      const d = toFlat(dijkstra(g, 0).distance)
      expect(toFlat(bellmanFord(g, 0).distance)).toEqual(d)
      expect(toRows(floydWarshall(g).distance)[0]).toEqual(d)
      // A* with a zero heuristic is Dijkstra.
      for (let t = 1; t < 10; t++) expect(aStar(g, 0, t, () => 0).distance).toBe(d[t])
    }
  })

  it('A* with a Manhattan heuristic finds the same length on a grid and expands fewer nodes', () => {
    const W = 8
    const edges: [number, number][] = []
    for (let r = 0; r < W; r++)
      for (let c = 0; c < W; c++) {
        if (c + 1 < W && !(c === 3 && r < W - 1)) edges.push([r * W + c, r * W + c + 1])
        if (r + 1 < W) edges.push([r * W + c, (r + 1) * W + c])
      }
    const g = fromEdges(W * W, edges, { directed: false })
    const target = W * W - 1
    const manhattan = (v: number) => Math.abs(Math.floor(v / W) - (W - 1)) + Math.abs((v % W) - (W - 1))
    const a = aStar(g, 0, target, manhattan)
    const z = aStar(g, 0, target, () => 0)
    expect(a.distance).toBe(z.distance)
    expect(a.path.shape[0]).toBe(a.distance + 1)
    expect(a.expanded).toBeLessThan(z.expanded)
  })

  it('Bellman–Ford returns a negative cycle as a witness', () => {
    const g = fromEdges(5, [
      [0, 1, 1],
      [1, 2, 2],
      [2, 3, -4],
      [3, 1, 1],
      [3, 4, 1],
    ])
    const r = bellmanFord(g, 0)
    const cycle = toFlat(r.negativeCycle!)
    expect(new Set(cycle)).toEqual(new Set([1, 2, 3]))
    let total = 0
    cycle.forEach((v, i) => {
      const e = g.edges.find((e) => e.from === v && e.to === cycle[(i + 1) % cycle.length])
      expect(e).toBeDefined()
      total += e!.weight!
    })
    expect(total).toBeLessThan(0)
  })
})

describe('trees and flows', () => {
  it('Kruskal and Prim give minimum spanning trees of equal weight', () => {
    for (let seed = 0; seed < 10; seed++) {
      const g = randomGraph(200 + seed, 12, 0.4, false)
      const k = minimumSpanningTree(g)
      const p = minimumSpanningTree(g, { method: 'prim' })
      expect(p.weight).toBe(k.weight)
      expect(p.trees).toBe(k.trees)
      expect(k.trees).toBe(connectedComponents(g).count)
    }
  })

  it('maximum flow equals the minimum cut capacity (CLRS Figure 26.1)', () => {
    // s v1 v2 v3 v4 t
    const g = fromEdges(6, [
      [0, 1, 16],
      [0, 2, 13],
      [2, 1, 4],
      [1, 3, 12],
      [3, 2, 9],
      [2, 4, 14],
      [4, 3, 7],
      [3, 5, 20],
      [4, 5, 4],
    ])
    const r = maxFlow(g, 0, 5)
    expect(r.value).toBe(23)
    expect(r.cutCapacity).toBe(23)
    expect(r.sourceSide.data[0]).toBe(1)
    expect(r.sourceSide.data[5]).toBe(0)
    for (let seed = 0; seed < 10; seed++) {
      const h = randomGraph(300 + seed, 8, 0.4, seed % 2 === 0)
      const f = maxFlow(h, 0, 7)
      expect(f.value).toBeCloseTo(f.cutCapacity, 12)
    }
  })
})
