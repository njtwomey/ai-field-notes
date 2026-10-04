/** A small HNSW (Malkov and Yashunin, Algorithms 1 to 5) in 2-D, instrumented to record a query's search path. */
import { dist2, type P } from '../_shared/geometry'
import { stream, uniform } from 'aifn/foundation/random'

export type Hnsw = {
  points: P[]
  /** Top layer of each point. */
  level: number[]
  /** links[layer][i]: neighbours of point i on that layer (empty if the point is not on it). */
  links: number[][][]
  entry: number
  top: number
}

/** One expansion during search: the node whose neighbours were examined, on which layer, and the work so far. */
export type Step = { layer: number; node: number; evaluated: number[] }

type Scored = [number, number] // [squared distance, index]

/**
 * Beam search on one layer (Algorithm 2). Keeps the ef closest points found; stops when the closest unexpanded
 * candidate is farther than the farthest kept point. `onExpand` reports each expanded node.
 */
function searchLayer(
  g: Hnsw,
  q: P,
  entries: number[],
  ef: number,
  layer: number,
  onExpand?: (node: number, evaluated: Set<number>) => void,
): Scored[] {
  const visited = new Set(entries)
  const candidates: Scored[] = entries.map((e) => [dist2(g.points[e], q), e])
  let found: Scored[] = [...candidates].sort((a, b) => a[0] - b[0]).slice(0, ef)
  while (candidates.length) {
    candidates.sort((a, b) => a[0] - b[0])
    const [dc, c] = candidates.shift()!
    if (dc > found[found.length - 1][0]) break
    for (const e of g.links[layer][c]) {
      if (visited.has(e)) continue
      visited.add(e)
      const de = dist2(g.points[e], q)
      if (found.length < ef || de < found[found.length - 1][0]) {
        candidates.push([de, e])
        found.push([de, e])
        found.sort((a, b) => a[0] - b[0])
        if (found.length > ef) found = found.slice(0, ef)
      }
    }
    onExpand?.(c, visited)
  }
  return found
}

/**
 * Neighbour selection heuristic (Algorithm 4). Candidates carry their distance to the base point. Take them closest first, and keep one only if it is closer to
 * the base point than to every neighbour already kept. This keeps edges pointing in different directions.
 */
function selectNeighbours(g: Hnsw, cands: Scored[], m: number): number[] {
  const kept: number[] = []
  for (const [d, c] of [...cands].sort((a, b) => a[0] - b[0])) {
    if (kept.length >= m) break
    if (kept.every((r) => d < dist2(g.points[c], g.points[r]))) kept.push(c)
  }
  return kept
}

export function build(points: P[], m: number, efConstruction: number, seed: number): Hnsw {
  const r = stream(seed)
  const mL = 1 / Math.log(m)
  const mMax0 = 2 * m
  const level = points.map(() => Math.floor(-Math.log(Math.max(uniform(r), 1e-12)) * mL))
  const maxLevel = Math.max(...level)
  const g: Hnsw = {
    points,
    level,
    links: Array.from({ length: maxLevel + 1 }, () => points.map(() => [] as number[])),
    entry: 0,
    top: level[0],
  }
  for (let i = 1; i < points.length; i++) {
    const q = points[i]
    let ep = [g.entry]
    for (let lc = g.top; lc > level[i]; lc--) ep = [searchLayer(g, q, ep, 1, lc)[0][1]]
    for (let lc = Math.min(g.top, level[i]); lc >= 0; lc--) {
      const found = searchLayer(g, q, ep, efConstruction, lc)
      const neighbours = selectNeighbours(g, found, m)
      g.links[lc][i] = neighbours
      const cap = lc === 0 ? mMax0 : m
      for (const e of neighbours) {
        const list = [...g.links[lc][e], i]
        g.links[lc][e] =
          list.length > cap
            ? selectNeighbours(
                g,
                list.map((x): Scored => [dist2(points[x], points[e]), x]),
                cap,
              )
            : list
      }
      ep = found.map(([, x]) => x)
    }
    if (level[i] > g.top) {
      g.top = level[i]
      g.entry = i
    }
  }
  return g
}

/** Search (Algorithm 5): greedy with ef = 1 on the upper layers, beam of width ef on layer 0. Records every step. */
export function search(g: Hnsw, q: P, ef: number): { steps: Step[]; result: number[] } {
  const steps: Step[] = []
  let ep = [g.entry]
  const evaluated = new Set<number>([g.entry])
  const record = (layer: number) => (node: number, seen: Set<number>) => {
    seen.forEach((s) => evaluated.add(s))
    steps.push({ layer, node, evaluated: [...evaluated] })
  }
  for (let lc = g.top; lc > 0; lc--) ep = [searchLayer(g, q, ep, 1, lc, record(lc))[0][1]]
  const found = searchLayer(g, q, ep, ef, 0, record(0))
  return { steps, result: found.map(([, x]) => x) }
}
