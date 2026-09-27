export type Linkage = 'single' | 'complete' | 'average' | 'ward'

/** One merge: clusters `a` and `b` (ids below n are points, id n + m is the cluster made by merge m). */
export type Merge = { a: number; b: number; height: number; size: number }

/**
 * Agglomerative clustering of points in any dimension with the Lance–Williams update, Euclidean distance. Each row of
 * the distance matrix caches its nearest active neighbour, so a merge rescans only the rows whose neighbour changed:
 * close to O(n²) in practice, fast enough to rebuild dozens of trees of 150 points while a slider moves. Ward heights
 * follow the usual convention: the updated Euclidean distance, so a merge of two points sits at their distance.
 */
export function agglomerate(points: readonly (readonly number[])[], linkage: Linkage): Merge[] {
  const n = points.length
  if (n < 2) return []
  const dim = points[0].length
  const D = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      let s = 0
      for (let c = 0; c < dim; c++) s += (points[i][c] - points[j][c]) ** 2
      D[i * n + j] = D[j * n + i] = Math.sqrt(s)
    }
  // Slot i holds cluster id[i]; a merge keeps the cluster in slot i and retires slot j.
  const id = Int32Array.from({ length: n }, (_, i) => i)
  const size = new Float64Array(n).fill(1)
  const active = new Uint8Array(n).fill(1)
  const nn = new Int32Array(n)
  const nnd = new Float64Array(n)
  const refresh = (i: number) => {
    let best = -1
    let bestD = Infinity
    for (let k = 0; k < n; k++)
      if (k !== i && active[k] && D[i * n + k] < bestD) {
        best = k
        bestD = D[i * n + k]
      }
    nn[i] = best
    nnd[i] = bestD
  }
  for (let i = 0; i < n; i++) refresh(i)
  const merges: Merge[] = []
  for (let m = 0; m < n - 1; m++) {
    let i = -1
    for (let r = 0; r < n; r++) if (active[r] && (i < 0 || nnd[r] < nnd[i])) i = r
    const j = nn[i]
    const h = nnd[i]
    const ni = size[i]
    const nj = size[j]
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === i || k === j) continue
      const dki = D[k * n + i]
      const dkj = D[k * n + j]
      const nk = size[k]
      let v: number
      if (linkage === 'single') v = Math.min(dki, dkj)
      else if (linkage === 'complete') v = Math.max(dki, dkj)
      else if (linkage === 'average') v = (ni * dki + nj * dkj) / (ni + nj)
      else v = Math.sqrt(Math.max(0, ((ni + nk) * dki ** 2 + (nj + nk) * dkj ** 2 - nk * h ** 2) / (ni + nj + nk)))
      D[k * n + i] = D[i * n + k] = v
    }
    active[j] = 0
    merges.push({ a: Math.min(id[i], id[j]), b: Math.max(id[i], id[j]), height: h, size: ni + nj })
    id[i] = n + m
    size[i] = ni + nj
    if (m === n - 2) break
    refresh(i)
    for (let k = 0; k < n; k++) {
      if (!active[k] || k === i) continue
      if (nn[k] === i || nn[k] === j) refresh(k)
      else if (D[k * n + i] < nnd[k]) {
        nn[k] = i
        nnd[k] = D[k * n + i]
      }
    }
  }
  return merges
}

/** Cluster labels after applying the first `count` merges: labels number the clusters in order of their first point. */
function labelsAfter(n: number, merges: Merge[], count: number): number[] {
  const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i)
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])))
  for (let k = 0; k < count; k++) {
    parent[find(merges[k].a)] = n + k
    parent[find(merges[k].b)] = n + k
  }
  const label = new Map<number, number>()
  return Array.from({ length: n }, (_, i) => {
    const root = find(i)
    if (!label.has(root)) label.set(root, label.size)
    return label.get(root)!
  })
}

/** Cluster labels after applying every merge at or below `height` (merge heights never decrease). */
export function cut(n: number, merges: Merge[], height: number): number[] {
  return labelsAfter(n, merges, merges.filter((m) => m.height <= height).length)
}

/** Cluster labels for exactly `k` clusters: the first n − k merges applied. */
export function cutK(n: number, merges: Merge[], k: number): number[] {
  return labelsAfter(n, merges, Math.max(0, Math.min(n - 1, n - k)))
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
