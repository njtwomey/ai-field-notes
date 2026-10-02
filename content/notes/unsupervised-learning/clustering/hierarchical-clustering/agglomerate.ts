import { linkage as aifnLinkage, cutTree } from 'aifn-applied/unsupervised/clustering'
import { fromData, toFlat } from 'aifn/foundation/tensor'

export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One merge: clusters `a` and `b` (ids below n are points, id n + m is the cluster made by merge m). */
export type Merge = { a: number; b: number; height: number; size: number }

/**
 * Agglomerative clustering of points in any dimension, backed by aifn-applied.
 */
export function agglomerate(points: readonly (readonly number[])[], linkage: Linkage): Merge[] {
  const n = points.length
  if (n < 2) return []
  const dim = points[0].length
  const flat = new Float64Array(n * dim)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < dim; c++) flat[i * dim + c] = points[i][c]
  }
  const x = fromData(flat, [n, dim])
  const Z = aifnLinkage(x, linkage)
  const zData = toFlat(Z)
  const merges: Merge[] = []
  for (let k = 0; k < n - 1; k++) {
    merges.push({
      a: zData[k * 4],
      b: zData[k * 4 + 1],
      height: zData[k * 4 + 2],
      size: zData[k * 4 + 3],
    })
  }
  return merges
}

function mergesToTensor(merges: Merge[]): { tensor: ReturnType<typeof fromData>; n: number } {
  const r = merges.length
  const n = r + 1
  const flat = new Float64Array(r * 4)
  for (let i = 0; i < r; i++) {
    flat[i * 4] = merges[i].a
    flat[i * 4 + 1] = merges[i].b
    flat[i * 4 + 2] = merges[i].height
    flat[i * 4 + 3] = merges[i].size
  }
  return { tensor: fromData(flat, [r, 4]), n }
}

/** Cluster labels after applying every merge at or below `height`, backed by aifn-applied. */
export function cut(n: number, merges: Merge[], height: number): number[] {
  if (merges.length === 0) return Array.from({ length: n }, (_, i) => i)
  const { tensor } = mergesToTensor(merges)
  return Array.from(toFlat(cutTree(tensor, { height })))
}

/** Cluster labels for exactly `k` clusters, backed by aifn-applied. */
export function cutK(n: number, merges: Merge[], k: number): number[] {
  if (merges.length === 0) return Array.from({ length: n }, (_, i) => i)
  const { tensor } = mergesToTensor(merges)
  return Array.from(toFlat(cutTree(tensor, { clusters: k })))
}

/** Cophenetic distances as a flat n × n matrix: the height of the merge that first joins each pair. */
export function cophenetic(n: number, merges: Merge[]): Float64Array {
  const C = new Float64Array(n * n)
  const members: number[][] = Array.from({ length: n }, (_, i) => [i])
  merges.forEach((m) => {
    const A = members[m.a]
    const B = members[m.b]
    for (const i of A)
      for (const j of B) {
        C[i * n + j] = m.height
        C[j * n + i] = m.height
      }
    members.push([...A, ...B])
  })
  return C
}

/**
 * Dendrogram as one polyline: a depth-first tour that retraces each vertical on the way back, so the retraced parts
 * are invisible and no breaks are needed. Leaves sit at x = 0, 1, … in tree order and at height 0.
 */
export function dendrogram(n: number, merges: Merge[]): { x: number[]; y: number[] } {
  const pos = new Array<number>(2 * n - 1).fill(0)
  const height = (id: number) => (id < n ? 0 : merges[id - n].height)
  let next = 0
  const place = (id: number) => {
    if (id < n) {
      pos[id] = next++
      return
    }
    const m = merges[id - n]
    place(m.a)
    place(m.b)
    pos[id] = (pos[m.a] + pos[m.b]) / 2
  }
  const root = 2 * n - 2
  place(root)
  const x: number[] = []
  const y: number[] = []
  const emit = (px: number, py: number) => {
    x.push(px)
    y.push(py)
  }
  const tour = (id: number) => {
    const h = height(id)
    emit(pos[id], h)
    if (id < n) return
    const { a, b } = merges[id - n]
    emit(pos[a], h)
    tour(a)
    emit(pos[a], h)
    emit(pos[b], h)
    tour(b)
    emit(pos[b], h)
    emit(pos[id], h)
  }
  tour(root)
  return { x, y }
}
