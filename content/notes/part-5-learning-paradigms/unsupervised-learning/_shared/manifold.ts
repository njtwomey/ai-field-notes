/**
 * Toy manifolds in three dimensions and light embedding methods for the dimensionality-reduction widgets. Every method
 * takes rows of data and returns an n × 2 embedding. All are exact and dense, so they suit a few hundred points.
 */
import { symmetricEigen } from './eigen'
import { affinities, squaredDistances, tsne } from './tsne'
import { normal, stream, uniform } from 'aifn/foundation/random'

export type Rows = number[][]

export type Manifold = {
  x: Rows
  /** Intrinsic coordinate used for colouring: position along the manifold, or a cluster label. */
  t: number[]
  /** True when `t` is a cluster label rather than a continuous coordinate. */
  labelled: boolean
  /** The two data axes that show the shape best. */
  view: [number, number]
}

export type DatasetId = 'swiss-roll' | 's-curve' | 'moons' | 'blobs' | 'circle'

/** Swiss roll: a 2D sheet (angle t in [1.5π, 4.5π], height h in [0, 10]) rolled up in 3D. */
function swissRoll(n: number, seed: number): Manifold {
  const r = stream(seed)
  const x: Rows = []
  const t: number[] = []
  for (let i = 0; i < n; i++) {
    const s = 1.5 * Math.PI * (1 + 2 * uniform(r))
    const h = 10 * uniform(r)
    x.push([s * Math.cos(s), h, s * Math.sin(s)])
    t.push(s)
  }
  return { x, t, labelled: false, view: [0, 2] }
}

/** S-curve: a sheet bent into an S, t in [-1.5π, 1.5π] along the S and height in [0, 2]. */
function sCurve(n: number, seed: number): Manifold {
  const r = stream(seed)
  const x: Rows = []
  const t: number[] = []
  for (let i = 0; i < n; i++) {
    const s = 3 * Math.PI * (uniform(r) - 0.5)
    x.push([Math.sin(s), 2 * uniform(r), Math.sign(s) * (Math.cos(s) - 1)])
    t.push(s)
  }
  return { x, t, labelled: false, view: [0, 2] }
}

/** Two interleaved half circles, tilted out of the plane and with noise in all three coordinates. */
function moons3d(n: number, seed: number): Manifold {
  const r = stream(seed)
  const x: Rows = []
  const t: number[] = []
  const c = Math.cos(0.6)
  const s = Math.sin(0.6)
  for (let i = 0; i < n; i++) {
    const upper = i % 2 === 0
    const a = Math.PI * uniform(r)
    const [u, v] = upper ? [Math.cos(a), Math.sin(a)] : [1 - Math.cos(a), 0.5 - Math.sin(a)]
    x.push([u + 0.08 * normal(r), c * v + 0.08 * normal(r), s * v + 0.08 * normal(r)])
    t.push(upper ? 0 : 1)
  }
  return { x, t, labelled: true, view: [0, 1] }
}

/** Three Gaussian blobs of different spreads. */
function blobs3d(n: number, seed: number): Manifold {
  const r = stream(seed)
  const centres = [
    [0, 0, 0],
    [4, 0, 0],
    [0, 8, 2],
  ]
  const sd = [0.6, 1.2, 0.6]
  const x: Rows = []
  const t: number[] = []
  for (let i = 0; i < n; i++) {
    const j = i % 3
    x.push(centres[j].map((m) => m + sd[j] * normal(r)))
    t.push(j)
  }
  return { x, t, labelled: true, view: [0, 1] }
}

/** A unit circle in a tilted plane with noise of standard deviation 0.1 in every coordinate. */
function noisyCircle(n: number, seed: number): Manifold {
  const r = stream(seed)
  const x: Rows = []
  const t: number[] = []
  for (let i = 0; i < n; i++) {
    const a = 2 * Math.PI * uniform(r)
    const [u, v] = [Math.cos(a), Math.sin(a)]
    x.push([u + 0.1 * normal(r), 0.5 * v + 0.1 * normal(r), 0.866 * v + 0.1 * normal(r)])
    t.push(a)
  }
  return { x, t, labelled: false, view: [0, 2] }
}

export function manifold(id: DatasetId, n: number, seed = 3): Manifold {
  switch (id) {
    case 'swiss-roll':
      return swissRoll(n, seed)
    case 's-curve':
      return sCurve(n, seed)
    case 'moons':
      return moons3d(n, seed)
    case 'blobs':
      return blobs3d(n, seed)
    case 'circle':
      return noisyCircle(n, seed)
  }
}

/** Euclidean distances between rows. */
export function distances(x: Rows): number[][] {
  return squaredDistances(x).map((row) => row.map(Math.sqrt))
}

/** Indices of the k nearest neighbours of each point, nearest first, from a distance matrix. */
export function knn(d: number[][], k: number): number[][] {
  return d.map((row, i) =>
    row
      .map((v, j) => [v, j] as const)
      .filter(([, j]) => j !== i)
      .sort((a, b) => a[0] - b[0])
      .slice(0, k)
      .map(([, j]) => j),
  )
}

/** Double-centre a symmetric matrix: J A J with J = I − 11ᵀ/n. */
export function doubleCentre(a: number[][]): number[][] {
  const n = a.length
  const row = a.map((r) => r.reduce((s, v) => s + v, 0) / n)
  const all = row.reduce((s, v) => s + v, 0) / n
  return a.map((r, i) => r.map((v, j) => v - row[i] - row[j] + all))
}

/** Top `dims` eigenvectors of a symmetric matrix, scaled by √λ: the classical-MDS and kernel-PCA coordinates. */
export function topScaled(b: number[][], dims = 2): { y: Rows; values: number[] } {
  const { values, vectors } = symmetricEigen(b)
  const y = b.map((_, i) => Array.from({ length: dims }, (_, k) => vectors[k][i] * Math.sqrt(Math.max(values[k], 0))))
  return { y, values }
}

export function pca(x: Rows): Rows {
  const n = x.length
  const dim = x[0].length
  const mean = Array.from({ length: dim }, (_, m) => x.reduce((s, r) => s + r[m], 0) / n)
  const c = x.map((r) => r.map((v, m) => v - mean[m]))
  const cov = Array.from({ length: dim }, (_, a) =>
    Array.from({ length: dim }, (_, b) => c.reduce((s, r) => s + r[a] * r[b], 0) / n),
  )
  const { vectors } = symmetricEigen(cov)
  return c.map((r) => [0, 1].map((k) => vectors[k].reduce((s, v, m) => s + v * r[m], 0)))
}

/** Classical MDS of a distance matrix: B = −½ J D⁽²⁾ J, top eigenvectors scaled by √λ. */
export function classicalMds(d: number[][], dims = 2): Rows {
  return topScaled(doubleCentre(d.map((r) => r.map((v) => -0.5 * v * v))), dims).y
}

/** Symmetrised k-NN graph as an adjacency list of neighbour sets. */
function knnGraph(d: number[][], k: number): Set<number>[] {
  const nb = knn(d, k)
  const adj = d.map(() => new Set<number>())
  nb.forEach((js, i) =>
    js.forEach((j) => {
      adj[i].add(j)
      adj[j].add(i)
    }),
  )
  return adj
}

/** Isomap: shortest paths in the k-NN graph (Floyd–Warshall), then classical MDS. Returns whether it is connected. */
export function isomap(d: number[][], k: number): { y: Rows; connected: boolean } {
  const n = d.length
  const adj = knnGraph(d, k)
  const g = Array.from({ length: n }, (_, i) => {
    const row = new Float64Array(n).fill(Infinity)
    row[i] = 0
    adj[i].forEach((j) => (row[j] = d[i][j]))
    return row
  })
  for (let m = 0; m < n; m++) {
    const gm = g[m]
    for (let i = 0; i < n; i++) {
      const gim = g[i][m]
      if (gim === Infinity) continue
      const gi = g[i]
      for (let j = 0; j < n; j++) if (gim + gm[j] < gi[j]) gi[j] = gim + gm[j]
    }
  }
  let max = 0
  let connected = true
  for (const row of g)
    for (const v of row) {
      if (v === Infinity) connected = false
      else if (v > max) max = v
    }
  // A disconnected graph has infinite distances; cap them so that MDS still returns a picture.
  const capped = g.map((row) => Array.from(row, (v) => (v === Infinity ? 1.5 * max : v)))
  return { y: classicalMds(capped), connected }
}

/** Mean distance from each point to its k-th nearest neighbour: a local length scale for kernels. */
export function kthDistance(d: number[][], k: number): number {
  const nb = knn(d, k)
  return nb.reduce((s, js, i) => s + d[i][js[js.length - 1]], 0) / d.length
}

/**
 * Eigenvectors 2 and 3 of the generalised problem L y = λ D y for a symmetric weight matrix W. Solved through
 * S = D^{-1/2} W D^{-1/2}: if S v = μ v then y = D^{-1/2} v satisfies L y = (1 − μ) D y.
 */
function generalisedBottom(w: number[][], power = 1): Rows {
  const deg = w.map((r) => r.reduce((s, v) => s + v, 0))
  const s = w.map((r, i) => r.map((v, j) => v / Math.sqrt(deg[i] * deg[j])))
  const { values, vectors } = symmetricEigen(s)
  return w.map((_, i) => [1, 2].map((k) => (vectors[k][i] / Math.sqrt(deg[i])) * Math.max(values[k], 0) ** power))
}

/** Laplacian eigenmaps on the symmetrised k-NN graph with heat-kernel weights. */
export function laplacianEigenmaps(d: number[][], k: number): Rows {
  const adj = knnGraph(d, k)
  const sigma = kthDistance(d, k)
  const w = d.map((row, i) => row.map((v, j) => (adj[i].has(j) ? Math.exp(-(v * v) / (sigma * sigma)) : 0)))
  const y = generalisedBottom(w, 0)
  return y.map((p) => p.map((v) => v * Math.sqrt(d.length)))
}

/**
 * Diffusion map with α = 1 normalisation at diffusion time t. The Gaussian kernel of width ε = σ² is truncated to the
 * symmetrised k-NN graph (plus self-loops): with a few hundred points, a dense kernel wide enough to connect the sheet
 * also leaks between the layers of a roll.
 */
export function diffusionMap(d: number[][], k: number, t = 1): Rows {
  const sigma = kthDistance(d, k)
  const adj = knnGraph(d, k)
  const kern = d.map((row, i) =>
    row.map((v, j) => (i === j || adj[i].has(j) ? Math.exp(-(v * v) / (sigma * sigma)) : 0)),
  )
  const q = kern.map((r) => r.reduce((s, v) => s + v, 0))
  // α = 1: divide out the sampling density so that the geometry, not the density, drives the diffusion.
  const k1 = kern.map((r, i) => r.map((v, j) => v / (q[i] * q[j])))
  const deg = k1.map((r) => r.reduce((s, v) => s + v, 0))
  const vol = deg.reduce((s, v) => s + v, 0)
  const y = generalisedBottom(k1, t)
  return y.map((p) => p.map((v) => v * Math.sqrt(vol)))
}

/** Kernel PCA with a Gaussian kernel of width σ: the double-centred kernel matrix, top eigenvectors scaled by √λ. */
export function kernelPca(d: number[][], sigma: number): Rows {
  const kern = d.map((row) => row.map((v) => Math.exp(-(v * v) / (2 * sigma * sigma))))
  return topScaled(doubleCentre(kern)).y
}

export function tsneEmbedding(x: Rows, perplexity: number, seed = 1): Rows {
  const run = tsne(affinities(squaredDistances(x), perplexity), seed, 400, 400)
  return run.snapshots[run.snapshots.length - 1]
}

/**
 * A compact UMAP: fuzzy k-NN graph with per-point ρ and σ, fuzzy union, spectral initialisation, and stochastic
 * gradient descent over edges with negative sampling, following umap-learn's defaults (min_dist 0.1: a = 1.577,
 * b = 0.895; 5 negative samples; 200 epochs).
 */
export function umap(d: number[][], k: number, seed = 1): Rows {
  const n = d.length
  const nb = knn(d, k)
  const directed = new Map<number, number>()
  nb.forEach((js, i) => {
    const rho = d[i][js[0]]
    const target = Math.log2(k)
    let lo = 0
    let hi = Infinity
    let sigma = 1
    for (let it = 0; it < 64; it++) {
      const s = js.reduce((acc, j) => acc + Math.exp(-Math.max(d[i][j] - rho, 0) / sigma), 0)
      if (Math.abs(s - target) < 1e-5) break
      if (s > target) {
        hi = sigma
        sigma = (lo + hi) / 2
      } else {
        lo = sigma
        sigma = hi === Infinity ? sigma * 2 : (lo + hi) / 2
      }
    }
    js.forEach((j) => directed.set(i * n + j, Math.exp(-Math.max(d[i][j] - rho, 0) / sigma)))
  })
  const edges: [number, number, number][] = []
  directed.forEach((a, key) => {
    const i = Math.floor(key / n)
    const j = key % n
    const b = directed.get(j * n + i) ?? 0
    if (b > 0 && j < i) return // the pair is added once, from its lower index
    edges.push([i, j, a + b - a * b])
  })
  const a = 1.577
  const b = 0.895
  // Spectral initialisation from the fuzzy graph, scaled to a box of side 20 as in umap-learn.
  const w = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  edges.forEach(([i, j, v]) => (w[i][j] = w[j][i] = v))
  const init = generalisedBottom(w, 0)
  const span = Math.max(...init.flat().map(Math.abs)) || 1
  const y = init.map((p) => p.map((v) => (10 * v) / span))
  const r = stream(seed)
  const maxW = Math.max(...edges.map((e) => e[2]))
  const epochs = 200
  const clip = (v: number) => Math.max(-4, Math.min(4, v))
  for (let epoch = 0; epoch < epochs; epoch++) {
    const lr = 1 - epoch / epochs
    for (const [i, j, v] of edges) {
      if (uniform(r) > v / maxW) continue
      for (const [p, q] of [
        [i, j],
        [j, i],
      ]) {
        const dx = y[p][0] - y[q][0]
        const dy = y[p][1] - y[q][1]
        const d2 = dx * dx + dy * dy
        if (d2 > 0) {
          const c = (-2 * a * b * d2 ** (b - 1)) / (1 + a * d2 ** b)
          y[p][0] += lr * clip(c * dx)
          y[p][1] += lr * clip(c * dy)
          y[q][0] -= lr * clip(c * dx)
          y[q][1] -= lr * clip(c * dy)
        }
        for (let s = 0; s < 5; s++) {
          const o = Math.floor(uniform(r) * n)
          if (o === p) continue
          const ex = y[p][0] - y[o][0]
          const ey = y[p][1] - y[o][1]
          const e2 = ex * ex + ey * ey
          const c = (2 * b) / ((0.001 + e2) * (1 + a * e2 ** b))
          y[p][0] += lr * clip(c * ex)
          y[p][1] += lr * clip(c * ey)
        }
      }
    }
  }
  return y
}

/** Rank of every point in every row: rank[i][j] = 1 for i's nearest neighbour, n − 1 for its farthest, 0 for itself. */
export function ranks(d: number[][]): number[][] {
  return d.map((row, i) => {
    const order = row
      .map((v, j) => [v, j] as const)
      .filter(([, j]) => j !== i)
      .sort((p, q) => p[0] - q[0])
    const out = new Array<number>(row.length).fill(0)
    order.forEach(([, j], r) => (out[j] = r + 1))
    return out
  })
}

/**
 * Trustworthiness and continuity of an embedding at neighbourhood size k (Venna and Kaski). Trustworthiness penalises
 * embedding neighbours that are far in the data (intruders) by how far down the data ranking they are; continuity
 * penalises data neighbours pushed out of the embedding neighbourhood. Both are 1 for a perfect embedding.
 */
export function trustworthiness(rData: number[][], rEmb: number[][], k: number): { trust: number; cont: number } {
  const n = rData.length
  let t = 0
  let c = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      if (rEmb[i][j] <= k && rData[i][j] > k) t += rData[i][j] - k
      if (rData[i][j] <= k && rEmb[i][j] > k) c += rEmb[i][j] - k
    }
  const z = 2 / (n * k * (2 * n - 3 * k - 1))
  return { trust: 1 - z * t, cont: 1 - z * c }
}

/**
 * Accuracy of the best linear split of two labelled groups in the plane: project onto Fisher's direction
 * S_W⁻¹(m₁ − m₀) and try every threshold.
 */
export function linearSplitAccuracy(z: Rows, labels: number[]): number {
  const mean = (g: number) => {
    const pts = z.filter((_, i) => labels[i] === g)
    return [0, 1].map((k) => pts.reduce((s, p) => s + p[k], 0) / pts.length)
  }
  const m = [mean(0), mean(1)]
  const s = [
    [1e-12, 0],
    [0, 1e-12],
  ]
  z.forEach((p, i) => {
    const c = m[labels[i]]
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) s[a][b] += (p[a] - c[a]) * (p[b] - c[b])
  })
  const det = s[0][0] * s[1][1] - s[0][1] * s[1][0]
  const dm = [m[1][0] - m[0][0], m[1][1] - m[0][1]]
  const w = [(s[1][1] * dm[0] - s[0][1] * dm[1]) / det, (-s[1][0] * dm[0] + s[0][0] * dm[1]) / det]
  const proj = z.map((p, i) => [w[0] * p[0] + w[1] * p[1], labels[i]]).sort((a, b) => a[0] - b[0])
  // Sweep the threshold: everything at or below position i is called group 0.
  let best = 0
  let zerosBelow = 0
  const ones = labels.filter((l) => l === 1).length
  let onesBelow = 0
  for (let i = -1; i < proj.length; i++) {
    if (i >= 0 && proj[i][1] === 0) zerosBelow++
    else if (i >= 0) onesBelow++
    const correct = zerosBelow + (ones - onesBelow)
    best = Math.max(best, correct, labels.length - correct)
  }
  return best / labels.length
}

/** Co-ranking matrix: q[a][b] counts pairs (i, j) with data rank a + 1 and embedding rank b + 1 (Lee and Verleysen). */
export function coranking(rData: number[][], rEmb: number[][]): number[][] {
  const n = rData.length
  const q = Array.from({ length: n - 1 }, () => new Array<number>(n - 1).fill(0))
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) q[rData[i][j] - 1][rEmb[i][j] - 1]++
  return q
}

/**
 * Trustworthiness, continuity and the neighbourhood agreement Q_NX at size K, all read off the co-ranking matrix.
 * Intruders sit in the block with embedding rank ≤ K and data rank > K; extrusions in the transposed block.
 */
export function rankQuality(q: number[][], k: number): { trust: number; cont: number; qnx: number } {
  const n = q.length + 1
  let t = 0
  let c = 0
  let kept = 0
  for (let a = 0; a < n - 1; a++)
    for (let b = 0; b < n - 1; b++) {
      const v = q[a][b]
      if (!v) continue
      if (b < k && a >= k) t += (a + 1 - k) * v
      if (a < k && b >= k) c += (b + 1 - k) * v
      if (a < k && b < k) kept += v
    }
  const z = 2 / (n * k * (2 * n - 3 * k - 1))
  return { trust: 1 - z * t, cont: 1 - z * c, qnx: kept / (k * n) }
}
