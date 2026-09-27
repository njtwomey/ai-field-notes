import type { Point } from '../../_shared/datasets'

export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One merge: clusters `a` and `b` (ids below n are points, id n + m is the cluster made by merge m). */
export type Merge = { a: number; b: number; height: number; size: number }

/**
 * Agglomerative clustering with the Lance–Williams update. O(n³), fine for the few dozen points in the figure. Ward
 * heights follow the usual convention: the updated Euclidean distance, so a merge of two points sits at their distance.
 */
export function agglomerate(points: Point[], linkage: Linkage): Merge[] {
  const n = points.length
  const d = new Map<number, Map<number, number>>()
  const size = new Map<number, number>()
  const active = new Set<number>()
  const put = (i: number, j: number, v: number) => {
    d.get(i)!.set(j, v)
    d.get(j)!.set(i, v)
  }
  for (let i = 0; i < n; i++) {
    d.set(i, new Map())
    size.set(i, 1)
    active.add(i)
  }
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) put(i, j, Math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]))
  const merges: Merge[] = []
  for (let m = 0; m < n - 1; m++) {
    let best = [-1, -1, Infinity]
    for (const i of active) for (const [j, v] of d.get(i)!) if (j > i && active.has(j) && v < best[2]) best = [i, j, v]
    const [i, j, h] = best
    const ni = size.get(i)!
    const nj = size.get(j)!
    const id = n + m
    d.set(id, new Map())
    active.delete(i)
    active.delete(j)
    for (const k of active) {
      const dki = d.get(k)!.get(i)!
      const dkj = d.get(k)!.get(j)!
      const nk = size.get(k)!
      let v: number
      if (linkage === 'single') v = Math.min(dki, dkj)
      else if (linkage === 'complete') v = Math.max(dki, dkj)
      else if (linkage === 'average') v = (ni * dki + nj * dkj) / (ni + nj)
      else v = Math.sqrt(((ni + nk) * dki ** 2 + (nj + nk) * dkj ** 2 - nk * h ** 2) / (ni + nj + nk))
      put(k, id, v)
    }
    active.add(id)
    size.set(id, ni + nj)
    merges.push({ a: i, b: j, height: h, size: ni + nj })
  }
  return merges
}

/** Cluster labels after applying every merge at or below `height`: labels are the smallest point index in each. */
export function cut(n: number, merges: Merge[], height: number): number[] {
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i)
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  merges.forEach((m, k) => {
    if (m.height <= height) {
      parent[find(m.a)] = n + k
      parent[find(m.b)] = n + k
    }
  })
  const label = new Map<number, number>()
  return Array.from({ length: n }, (_, i) => {
    const root = find(i)
    if (!label.has(root)) label.set(root, label.size)
    return label.get(root)!
  })
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
