/**
 * Deep SVDD (Ruff et al. 2018) and Deep SAD (Ruff et al. 2020) on 2-D data, small enough to train in the browser.
 *
 * The network is a ReLU multilayer perceptron 2 → H → H → 2 trained full-batch with Adam. Every quantity is plain
 * arrays, so the same code runs under Node for checking against a NumPy reimplementation.
 *
 * Objectives, for embeddings oᵢ = φ(xᵢ; W) and a centre c:
 *   one-class      (1/n) Σ ‖oᵢ − c‖²                                   + (λ/2) Σ ‖Wˡ‖²
 *   soft-boundary  R² + 1/(νn) Σ max(0, ‖oᵢ − c‖² − R²)                  + (λ/2) Σ ‖Wˡ‖²
 *   Deep SAD       1/(n+m) [Σ unlabelled ‖oᵢ − c‖² + η Σ labelled (‖oⱼ − c‖² + ε)⁻¹] + (λ/2) Σ ‖Wˡ‖²
 * Weight decay applies to weight matrices only, never to biases or a learnable centre.
 */

export type Point = [number, number]
export type Rand = { uniform: () => number; normal: () => number }

export type Net = {
  sizes: number[]
  /** W[l] is row-major, sizes[l+1] × sizes[l]. */
  W: Float64Array[]
  /** Biases, or null for a bias-free network. */
  b: Float64Array[] | null
}

export function initNet(sizes: number[], bias: boolean, r: Rand): Net {
  const W: Float64Array[] = []
  const b: Float64Array[] = []
  for (let l = 0; l + 1 < sizes.length; l++) {
    const fanIn = sizes[l]
    const w = new Float64Array(sizes[l + 1] * fanIn)
    // He initialisation, suited to ReLU layers.
    for (let k = 0; k < w.length; k++) w[k] = r.normal() * Math.sqrt(2 / fanIn)
    W.push(w)
    b.push(new Float64Array(sizes[l + 1]))
  }
  return { sizes, W, b: bias ? b : null }
}

export const cloneNet = (net: Net): Net => ({
  sizes: net.sizes,
  W: net.W.map((w) => w.slice()),
  b: net.b ? net.b.map((v) => v.slice()) : null,
})

/** Forward pass keeping every layer's activations (post-ReLU; the last layer is linear). */
function forwardAll(net: Net, x: Point): Float64Array[] {
  const acts: Float64Array[] = [Float64Array.from(x)]
  const L = net.W.length
  for (let l = 0; l < L; l++) {
    const nIn = net.sizes[l]
    const nOut = net.sizes[l + 1]
    const a = acts[l]
    const w = net.W[l]
    const z = new Float64Array(nOut)
    for (let o = 0; o < nOut; o++) {
      let s = net.b ? net.b[l][o] : 0
      for (let i = 0; i < nIn; i++) s += w[o * nIn + i] * a[i]
      z[o] = l < L - 1 ? Math.max(0, s) : s
    }
    acts.push(z)
  }
  return acts
}

export function embed(net: Net, x: Point): Point {
  const out = forwardAll(net, x)[net.W.length]
  return [out[0], out[1]]
}

export const sqDist = (a: Point, b: Point) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

export type Objective =
  | { kind: 'one-class' }
  | { kind: 'soft-boundary'; nu: number }
  | { kind: 'semi-supervised'; eta: number; labelled: Point[] }

export type TrainOptions = {
  objective: Objective
  /** Learn c jointly with the weights: a known cause of hypersphere collapse. */
  learnCentre: boolean
  epochs: number
  lr: number
  weightDecay: number
  /** Keep a snapshot every this many epochs (epoch 0 included). */
  every: number
  /** Epochs before the soft-boundary radius is first updated. */
  warmUp?: number
}

export type Snapshot = { epoch: number; net: Net; c: Point; R: number; loss: number }

/**
 * The centre used by Ruff et al.: the mean of the initial embeddings of the training data, with coordinates near zero
 * pushed out to ±eps, since a zero coordinate can be matched by a unit whose weights are all zero.
 */
export function initialCentre(net: Net, x: Point[], eps = 0.1): Point {
  const c: Point = [0, 0]
  for (const p of x) {
    const o = embed(net, p)
    c[0] += o[0] / x.length
    c[1] += o[1] / x.length
  }
  return c.map((v) => (Math.abs(v) < eps ? (v < 0 ? -eps : eps) : v)) as Point
}

/** The (1 − ν) quantile of the distances ‖oᵢ − c‖, the radius that minimises the soft-boundary objective. */
export function radiusQuantile(d2: number[], nu: number): number {
  const s = d2.map(Math.sqrt).sort((a, b) => a - b)
  const k = Math.min(s.length - 1, Math.max(0, Math.ceil((1 - nu) * s.length) - 1))
  return s[k]
}

export const EPS_SAD = 1e-6

/** Full-batch Adam on the chosen objective. Returns snapshots every `every` epochs. */
export function train(net0: Net, x: Point[], c0: Point, opt: TrainOptions): Snapshot[] {
  const net = cloneNet(net0)
  const L = net.W.length
  const c: Point = [c0[0], c0[1]]
  const { objective } = opt
  const labelled = objective.kind === 'semi-supervised' ? objective.labelled : []
  const all = [...x, ...labelled]
  const n = x.length
  const denom = n + labelled.length
  let R = 0
  const warmUp = opt.warmUp ?? 10
  // Adam state: one moment pair per weight matrix, per bias vector and for c.
  const params: Float64Array[] = [...net.W, ...(net.b ?? [])]
  const cArr = Float64Array.from(c)
  if (opt.learnCentre) params.push(cArr)
  const m1 = params.map((p) => new Float64Array(p.length))
  const m2 = params.map((p) => new Float64Array(p.length))
  const b1 = 0.9
  const b2 = 0.999
  const snaps: Snapshot[] = []

  const lossAndGrad = (withGrad: boolean) => {
    const gW = net.W.map((w) => new Float64Array(w.length))
    const gb = net.b ? net.b.map((v) => new Float64Array(v.length)) : null
    const gc = new Float64Array(2)
    let loss = 0
    const d2: number[] = []
    for (let i = 0; i < all.length; i++) {
      const acts = forwardAll(net, all[i])
      const o = acts[L]
      const dx = o[0] - cArr[0]
      const dy = o[1] - cArr[1]
      const d = dx * dx + dy * dy
      if (i < n) d2.push(d)
      // dℓ/dd for this point's term.
      let dl = 0
      if (i >= n) {
        const eta = objective.kind === 'semi-supervised' ? objective.eta : 0
        loss += eta / (d + EPS_SAD) / denom
        dl = -eta / (d + EPS_SAD) ** 2 / denom
      } else if (objective.kind === 'soft-boundary') {
        const excess = d - R * R
        if (excess > 0) {
          loss += excess / (objective.nu * n)
          dl = 1 / (objective.nu * n)
        }
      } else {
        loss += d / denom
        dl = 1 / denom
      }
      if (!withGrad || dl === 0) continue
      gc[0] -= 2 * dx * dl
      gc[1] -= 2 * dy * dl
      let delta = new Float64Array([2 * dx * dl, 2 * dy * dl])
      for (let l = L - 1; l >= 0; l--) {
        const nIn = net.sizes[l]
        const nOut = net.sizes[l + 1]
        const a = acts[l]
        const w = net.W[l]
        const g = gW[l]
        for (let o2 = 0; o2 < nOut; o2++) {
          const dv = delta[o2]
          if (dv === 0) continue
          if (gb) gb[l][o2] += dv
          for (let k = 0; k < nIn; k++) g[o2 * nIn + k] += dv * a[k]
        }
        if (l === 0) break
        const prev = new Float64Array(nIn)
        for (let k = 0; k < nIn; k++) {
          if (a[k] <= 0) continue // ReLU derivative
          let s = 0
          for (let o2 = 0; o2 < nOut; o2++) s += w[o2 * nIn + k] * delta[o2]
          prev[k] = s
        }
        delta = prev
      }
    }
    if (objective.kind === 'soft-boundary') loss += R * R
    for (let l = 0; l < L; l++) {
      const w = net.W[l]
      let s = 0
      for (let k = 0; k < w.length; k++) {
        s += w[k] * w[k]
        gW[l][k] += opt.weightDecay * w[k]
      }
      loss += (opt.weightDecay / 2) * s
    }
    return { loss, gW, gb, gc, d2 }
  }

  const snap = (epoch: number, loss: number) =>
    snaps.push({ epoch, net: cloneNet(net), c: [cArr[0], cArr[1]], R, loss })

  for (let t = 0; t <= opt.epochs; t++) {
    const { loss, gW, gb, gc, d2 } = lossAndGrad(t < opt.epochs)
    if (t % opt.every === 0) snap(t, loss)
    if (t === opt.epochs) break
    const grads: Float64Array[] = [...gW, ...(gb ?? [])]
    if (opt.learnCentre) grads.push(gc)
    const bc1 = 1 - b1 ** (t + 1)
    const bc2 = 1 - b2 ** (t + 1)
    params.forEach((p, j) => {
      const g = grads[j]
      for (let k = 0; k < p.length; k++) {
        m1[j][k] = b1 * m1[j][k] + (1 - b1) * g[k]
        m2[j][k] = b2 * m2[j][k] + (1 - b2) * g[k] * g[k]
        p[k] -= (opt.lr * (m1[j][k] / bc1)) / (Math.sqrt(m2[j][k] / bc2) + 1e-8)
      }
    })
    // The radius is not a gradient parameter: after the warm-up it is set to its optimum given the weights.
    if (objective.kind === 'soft-boundary' && t + 1 >= warmUp) R = radiusQuantile(d2, objective.nu)
  }
  return snaps
}

/** Area under the ROC curve, counting ties as one half. */
export function auc(normal: number[], anomalous: number[]): number {
  let s = 0
  for (const a of anomalous) for (const b of normal) s += a > b ? 1 : a === b ? 0.5 : 0
  return s / (normal.length * anomalous.length)
}

export type Shape = 'ring' | 'blobs'

/** Normal training data: a noisy ring of radius 2, or two Gaussian blobs, all away from the origin. */
export function normalData(shape: Shape, n: number, r: Rand): Point[] {
  return Array.from({ length: n }, (_, i) => {
    if (shape === 'ring') {
      const a = 2 * Math.PI * r.uniform()
      const rad = 2 + 0.15 * r.normal()
      return [rad * Math.cos(a), rad * Math.sin(a)] as Point
    }
    const [mx, my] = i % 2 === 0 ? [-1.6, 1.2] : [1.6, 1.4]
    return [mx + 0.4 * r.normal(), my + 0.4 * r.normal()] as Point
  })
}

/** Test anomalies: uniform on the square [−lim, lim]², rejecting points within `gap` of any normal point. */
export function uniformAnomalies(n: number, lim: number, normal: Point[], gap: number, r: Rand): Point[] {
  const out: Point[] = []
  while (out.length < n) {
    const p: Point = [lim * (2 * r.uniform() - 1), lim * (2 * r.uniform() - 1)]
    if (normal.every((q) => sqDist(p, q) > gap * gap)) out.push(p)
  }
  return out
}

/** Scores ‖φ(x) − c‖² on a grid, row-major: z[i][j] is the score at (axis[j], axis[i]). */
export function scoreGrid(net: Net, c: Point, axis: number[]): number[][] {
  return axis.map((v) => axis.map((u) => sqDist(embed(net, [u, v]), c)))
}
