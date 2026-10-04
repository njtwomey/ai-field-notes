import { normal, stream } from 'aifn/foundation/random'

export type Embedding = [number, number][]

/** Squared Euclidean distances between the rows of x. */
export function squaredDistances(x: number[][]): number[][] {
  return x.map((a) => x.map((b) => a.reduce((s, v, m) => s + (v - b[m]) ** 2, 0)))
}

/**
 * Symmetric input affinities p_ij. For each point, a binary search on the precision β = 1/(2σ²) makes the entropy of
 * p_{·|i} equal log(perplexity); then p_ij = (p_{j|i} + p_{i|j}) / 2n.
 */
export function affinities(d2: number[][], perplexity: number): number[][] {
  const n = d2.length
  const target = Math.log(perplexity)
  const cond = d2.map((row, i) => {
    let lo = 0
    let hi = Infinity
    let beta = 1
    let p: number[] = []
    for (let it = 0; it < 60; it++) {
      p = row.map((d, j) => (j === i ? 0 : Math.exp(-beta * d)))
      const z = p.reduce((a, b) => a + b, 0) || 1e-300
      // Entropy of the normalised row: H = log z + β Σ p_j d_j / z.
      const h = Math.log(z) + (beta * p.reduce((s, pj, j) => s + pj * row[j], 0)) / z
      p = p.map((v) => v / z)
      if (Math.abs(h - target) < 1e-5) break
      if (h > target) {
        lo = beta
        beta = hi === Infinity ? beta * 2 : (beta + hi) / 2
      } else {
        hi = beta
        beta = (beta + lo) / 2
      }
    }
    return p
  })
  return cond.map((row, i) => row.map((v, j) => (v + cond[j][i]) / (2 * n)))
}

export type Run = { snapshots: Embedding[]; kl: number[] }

/**
 * Exact t-SNE by gradient descent with momentum, per-parameter gains and early exaggeration, as in van der Maaten and
 * Hinton (2008). Returns the embedding every `every` iterations and the KL divergence at each snapshot.
 */
export function tsne(p: number[][], seed: number, iterations = 500, every = 10): Run {
  const n = p.length
  const r = stream(seed)
  const y: Embedding = Array.from({ length: n }, () => [1e-2 * normal(r), 1e-2 * normal(r)])
  const velocity = y.map(() => [0, 0])
  const gains = y.map(() => [1, 1])
  const eta = Math.max(n / 12, 50)
  const snapshots: Embedding[] = [y.map((v) => [v[0], v[1]])]
  const kl: number[] = [klDivergence(p, y)]
  for (let it = 1; it <= iterations; it++) {
    const exaggeration = it <= 100 ? 12 : 1
    const momentum = it <= 100 ? 0.5 : 0.8
    const w = y.map((a, i) => y.map((b, j) => (i === j ? 0 : 1 / (1 + (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))))
    const z = w.reduce((s, row) => s + row.reduce((a, b) => a + b, 0), 0)
    for (let i = 0; i < n; i++) {
      // ∂C/∂y_i = 4 Σ_j (p_ij − q_ij) w_ij (y_i − y_j), with q_ij = w_ij / Z.
      let gx = 0
      let gy = 0
      for (let j = 0; j < n; j++) {
        if (i === j) continue
        const f = (exaggeration * p[i][j] - w[i][j] / z) * w[i][j]
        gx += 4 * f * (y[i][0] - y[j][0])
        gy += 4 * f * (y[i][1] - y[j][1])
      }
      const g = [gx, gy]
      for (let k = 0; k < 2; k++) {
        gains[i][k] = Math.sign(g[k]) === Math.sign(velocity[i][k]) ? gains[i][k] * 0.8 : gains[i][k] + 0.2
        gains[i][k] = Math.max(gains[i][k], 0.01)
        velocity[i][k] = momentum * velocity[i][k] - eta * gains[i][k] * g[k]
      }
    }
    for (let i = 0; i < n; i++) {
      y[i][0] += velocity[i][0]
      y[i][1] += velocity[i][1]
    }
    // Keep the embedding centred; the cost is translation invariant.
    const mx = y.reduce((s, v) => s + v[0], 0) / n
    const my = y.reduce((s, v) => s + v[1], 0) / n
    for (const v of y) {
      v[0] -= mx
      v[1] -= my
    }
    if (it % every === 0) {
      snapshots.push(y.map((v) => [v[0], v[1]]))
      kl.push(klDivergence(p, y))
    }
  }
  return { snapshots, kl }
}

export function klDivergence(p: number[][], y: Embedding): number {
  const n = p.length
  let z = 0
  const w = y.map((a, i) =>
    y.map((b, j) => {
      if (i === j) return 0
      const v = 1 / (1 + (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
      z += v
      return v
    }),
  )
  let c = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (p[i][j] > 1e-12) c += p[i][j] * Math.log(p[i][j] / (w[i][j] / z))
  return c
}
