import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { normalCdf, normalPdf, sigmoid } from 'aifn-compute/numerics/special'
/**
 * Shared data, models and metrics for the ordinal regression notes. Every figure in the category draws from the same
 * seeded datasets, so a model can be compared with another on identical points.
 *
 * Classes are 0-based in code (0, …, K − 1) and 1-based in the notes. Thresholds θ_0 < … < θ_{K−2} split a real score
 * into K intervals; θ_{−1} = −∞ and θ_{K−1} = +∞ are implicit.
 */

export type Point = [number, number]
export type Sample = { x: Point[]; y: number[] }

// ---------------------------------------------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------------------------------------------

/**
 * The synthetic manifolds of Twomey et al. (2019): a latent ordinal coordinate t ∈ [0, 1] is mapped onto a curve in the
 * plane, isotropic Gaussian noise is added, and the class is the index of the equal-width interval of t that the point
 * came from. The paper names the four curves Linear, Sine, Circle and Spiral.
 */
export type Shape = 'linear' | 'sine' | 'arc' | 'spiral'

/** Everything that defines a dataset. Every ordinal widget draws its 2-D data from one of these. */
export type DataSpec = {
  shape: Shape
  k: number
  noise: number
  seed: number
  /** Training points per class; the training set has m·K points. */
  m: number
}

/** 20 points per class: 100 training points at K = 5, the paper's training size. */
export const DEFAULT_DATA: DataSpec = { shape: 'arc', k: 5, noise: 0.2, seed: 1, m: 20 }

export const SHAPE_OPTIONS: { value: Shape; label: string }[] = [
  { value: 'linear', label: 'linear' },
  { value: 'sine', label: 'sine' },
  { value: 'arc', label: 'arc' },
  { value: 'spiral', label: 'spiral' },
]

/** Held-out points per class: 1000 in all at K = 5, the paper's test size. Fixed, whatever the training size. */
export const TEST_PER_CLASS = 200

export type Dataset = DataSpec & {
  train: Sample
  test: Sample
  /** Both axes span this symmetric range, which covers the curve and its noise. */
  range: [number, number]
}

const clip = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** The class of a latent value: the number of thresholds below it. */
export const intervalOf = (value: number, thresholds: number[]) => thresholds.filter((t) => t < value).length

const LINE = [Math.cos(Math.PI / 6), Math.sin(Math.PI / 6)]

/** Archimedean spiral r = a + bθ: a quarter turn per class, so K classes wind K/4 turns outwards. */
const SPIRAL = { a: 0.4, b: 0.3, perClass: Math.PI / 2 }
const spiralAngle = (t: number, k: number) => t * k * SPIRAL.perClass
const spiralRadius = (t: number, k: number) => SPIRAL.a + SPIRAL.b * spiralAngle(t, k)

/**
 * The curve γ(t) for t ∈ [0, 1] with K classes. The linear, sine and arc curves stay inside [−3, 3]² whatever K; the
 * spiral gives each class a quarter turn, so it grows outwards as K grows.
 */
export function curve(shape: Shape, t: number, k: number): Point {
  switch (shape) {
    case 'linear': {
      // A straight segment of length 5 at 30° to the x-axis.
      const u = -2.5 + 5 * t
      return [u * LINE[0], u * LINE[1]]
    }
    case 'sine':
      // Left to right while going up and down twice.
      return [-2.5 + 5 * t, 1.5 * Math.sin(4 * Math.PI * t)]
    case 'arc': {
      // Three quarters of a circle of radius 2.2, anticlockwise from −45°. A per-class angle would pass a full turn and
      // bring the last class back next to the first, so the arc keeps a fixed total angle.
      const a = -Math.PI / 4 + 1.5 * Math.PI * t
      return [2.2 * Math.cos(a), 2.2 * Math.sin(a) - 0.3]
    }
    case 'spiral': {
      const a = spiralAngle(t, k)
      const r = spiralRadius(t, k)
      return [r * Math.cos(a), r * Math.sin(a)]
    }
  }
}

/**
 * How the noise scales along the curve. On the spiral it is proportional to the radius, divided by the radius at the
 * middle of the curve so that σ keeps the same average size as on the other curves: the inner turns stay tight and
 * the outer turns spread, as in the paper's spiral. The other curves are isotropic with constant noise.
 */
export const noiseScale = (shape: Shape, t: number, k: number) =>
  shape === 'spiral' ? spiralRadius(t, k) / spiralRadius(0.5, k) : 1

/** The class of latent coordinate t: its interval among K equal intervals of [0, 1]. */
export const classOf = (t: number, k: number) => Math.min(Math.floor(t * k), k - 1)

/** `perClass` points in each class, with t uniform within the class's interval, so every class has the same count. */
function draw(spec: DataSpec, perClass: number, seed: number): Sample {
  const r = stream(seed)
  const x: Point[] = []
  const y: number[] = []
  for (let c = 0; c < spec.k; c++)
    for (let i = 0; i < perClass; i++) {
      const t = (c + uniform(r)) / spec.k
      const [cx, cy] = curve(spec.shape, t, spec.k)
      const sd = spec.noise * noiseScale(spec.shape, t, spec.k)
      x.push([cx + sd * normal(r), cy + sd * normal(r)])
      y.push(c)
    }
  return { x, y }
}

export const specKey = (s: DataSpec) => `${s.shape}/${s.k}/${s.noise}/${s.seed}/${s.m}`

const datasetCache = new Map<string, Dataset>()

/** Bounded caches: a slider drag visits many specs, and each only needs to be kept while it is on screen. */
function remember<T>(cache: Map<string, T>, key: string, value: T): T {
  if (cache.size > 64) cache.delete(cache.keys().next().value as string)
  cache.set(key, value)
  return value
}

/**
 * m training points and 200 test points per class for a spec; the same on every device. The test set does not depend
 * on m, so metrics stay comparable across training sizes.
 */
export function dataset(spec: DataSpec): Dataset {
  const key = specKey(spec)
  const hit = datasetCache.get(key)
  if (hit) return hit
  const train = draw(spec, spec.m, 1000 * spec.seed + 1)
  const test = draw(spec, TEST_PER_CLASS, 1000 * spec.seed + 2)
  // The plot is square and fits the data with a small margin, rounded up to half a unit.
  const reach = Math.max(...[...train.x, ...test.x].map(([u, v]) => Math.max(Math.abs(u), Math.abs(v))))
  const R = Math.ceil((reach + 0.25) * 2) / 2
  return remember(datasetCache, key, { ...spec, train, test, range: [-R, R] })
}

// ---------------------------------------------------------------------------------------------------------------
// Optimisation
// ---------------------------------------------------------------------------------------------------------------

/** A smooth objective: fills `grad` at `p` and returns the loss. */
export type LossGrad = (p: number[], grad: number[]) => number

/**
 * Adam on a full-batch loss (the datasets are small), run in slices so a long fit can yield to the browser. It stops
 * when the loss has fallen by less than `tol` (relative) over the last 50 steps, or at `maxSteps`, so models with more
 * parameters get the steps they need rather than a fixed budget.
 */
export class Optimiser {
  static readonly WINDOW = 50
  readonly p: number[]
  steps = 0
  loss = NaN
  done = false
  private readonly m: number[]
  private readonly v: number[]
  private readonly g: number[]
  private readonly history: number[] = []

  private readonly lossGrad: LossGrad
  private readonly rate: number
  private readonly maxSteps: number
  private readonly tol: number

  constructor(init: number[], lossGrad: LossGrad, rate = 0.05, maxSteps = 4000, tol = 1e-4) {
    this.p = init.slice()
    this.m = init.map(() => 0)
    this.v = init.map(() => 0)
    this.g = init.map(() => 0)
    this.lossGrad = lossGrad
    this.rate = rate
    this.maxSteps = maxSteps
    this.tol = tol
  }

  /** Take steps until converged or until `ms` milliseconds have passed; returns whether it has converged. */
  run(ms = Infinity): boolean {
    const until = performance.now() + ms
    const { p, m, v, g, history } = this
    while (!this.done && performance.now() < until) {
      g.fill(0)
      const loss = this.lossGrad(p, g)
      history.push(loss)
      this.loss = loss
      const t = ++this.steps
      const W = Optimiser.WINDOW
      if ((t > W && history[t - 1 - W] - loss < this.tol * (1 + Math.abs(loss))) || t >= this.maxSteps) {
        this.done = true
        break
      }
      const c1 = 1 - 0.9 ** t
      const c2 = 1 - 0.999 ** t
      for (let i = 0; i < p.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i]
        v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
        p[i] -= (this.rate * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8)
      }
    }
    return this.done
  }
}

/** Run an optimiser to convergence. */
export function minimise(init: number[], lossGrad: LossGrad, rate = 0.05): number[] {
  const o = new Optimiser(init, lossGrad, rate)
  o.run()
  return o.p
}

export const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)))
const logisticPdf = (z: number) => {
  const s = sigmoid(z)
  return s * (1 - s)
}

// ---------------------------------------------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------------------------------------------

export type ModelId =
  | 'cumulative-logit'
  | 'cumulative-probit'
  | 'continuation-ratio'
  | 'adjacent-category'
  | 'immediate-threshold'
  | 'all-threshold'
  | 'binary-decomposition'
  | 'storm'
  | 'multiclass'
  | 'regression'
  | 'svor'
  | 'li-lin'
  | 'orcnn'
  | 'coral'
  | 'corn'
  | 'sord'
  | 'storm-poly2'
  | 'storm-poly3'
  | 'storm-nystrom'

export const MODEL_LABELS: Record<ModelId, string> = {
  'cumulative-logit': 'cumulative logit',
  'cumulative-probit': 'cumulative probit',
  'continuation-ratio': 'continuation ratio',
  'adjacent-category': 'adjacent category',
  'immediate-threshold': 'immediate threshold',
  'all-threshold': 'all threshold (logistic)',
  'binary-decomposition': 'Frank & Hall',
  storm: 'StORM, linear',
  multiclass: 'multiclass logistic',
  regression: 'regress then round',
  svor: 'SVOR (all-threshold hinge)',
  'li-lin': 'Li & Lin (network on extended examples)',
  orcnn: 'OR-CNN heads (network)',
  coral: 'CORAL (network)',
  corn: 'CORN (network)',
  sord: 'SORD (network)',
  'storm-poly2': 'StORM, polynomial degree 2',
  'storm-poly3': 'StORM, polynomial degree 3',
  'storm-nystrom': 'StORM, Nyström',
}

export const ALL_MODELS = Object.keys(MODEL_LABELS) as ModelId[]

export type Fitted = {
  /** Class probabilities at a point. Frank & Hall can return negative entries; StORM leaves mass on invalid codes. */
  probs: (x: Point) => number[]
  predict: (x: Point) => number
  /** A linear score and ordered thresholds, for the models that have them. */
  score?: (x: Point) => number
  thresholds?: number[]
  /** StORM only: the probability of bit patterns that encode no class. */
  invalidMass?: (x: Point) => number
  /** StORM only: the chain, for bit marginals and interval queries. */
  storm?: StormModel
}

/** E[y | x] on the 1-based class scale, from class probabilities normalised to sum to one. */
export function expectedClass(probs: number[]): number {
  const total = probs.reduce((a, b) => a + b, 0)
  return probs.reduce((s, p, k) => s + (k + 1) * p, 0) / total
}

const argmax = (v: number[]) => v.reduce((best, x, i) => (x > v[best] ? i : best), 0)

/** The index of the largest entry. */
export const argmaxIndex = argmax
const L2 = 1e-3

/**
 * Per-example loss of a score s = wᵀx against thresholds θ for class y. Returns the loss and dL/ds, and adds dL/dθ into
 * `dTheta`.
 */
type ThresholdLoss = (s: number, theta: number[], y: number, dTheta: number[]) => [number, number]

/** Cumulative link: P(y = k) = F(θ_k − s) − F(θ_{k−1} − s). */
function cumulativeLoss(F: (z: number) => number, f: (z: number) => number): ThresholdLoss {
  return (s, th, y, dTheta) => {
    const upper = y < th.length
    const lower = y > 0
    const Fu = upper ? F(th[y] - s) : 1
    const fu = upper ? f(th[y] - s) : 0
    const Fl = lower ? F(th[y - 1] - s) : 0
    const fl = lower ? f(th[y - 1] - s) : 0
    const p = Math.max(Fu - Fl, 1e-12)
    if (upper) dTheta[y] -= fu / p
    if (lower) dTheta[y - 1] += fl / p
    return [-Math.log(p), (fu - fl) / p]
  }
}

export function cumulativeProbs(F: (z: number) => number, s: number, th: number[]): number[] {
  const cdf = [...th.map((t) => F(t - s)), 1]
  return cdf.map((c, k) => c - (k > 0 ? cdf[k - 1] : 0))
}

/** Continuation ratio: P(y = k | y ≥ k) = σ(θ_k − s). */
const continuationLoss: ThresholdLoss = (s, th, y, dTheta) => {
  let loss = 0
  let ds = 0
  if (y < th.length) {
    const z = th[y] - s
    loss += softplus(-z)
    const g = sigmoid(z) - 1
    dTheta[y] += g
    ds -= g
  }
  for (let j = 0; j < y; j++) {
    const z = s - th[j]
    loss += softplus(-z)
    const g = sigmoid(z) - 1
    ds += g
    dTheta[j] -= g
  }
  return [loss, ds]
}

export function continuationProbs(s: number, th: number[]): number[] {
  let remaining = 1
  const out: number[] = []
  for (const t of th) {
    const h = sigmoid(t - s)
    out.push(remaining * h)
    remaining *= 1 - h
  }
  out.push(remaining)
  return out
}

/** Adjacent category: log P(y = k + 1)/P(y = k) = s − θ_k, so class k has logit k·s − Σ_{j<k} θ_j. */
function adjacentLogits(s: number, th: number[]): number[] {
  const a = [0]
  let cumulative = 0
  for (let k = 1; k <= th.length; k++) {
    cumulative += th[k - 1]
    a.push(k * s - cumulative)
  }
  return a
}

function softmax(a: number[]): number[] {
  const top = Math.max(...a)
  const e = a.map((v) => Math.exp(v - top))
  const z = e.reduce((x, y) => x + y, 0)
  return e.map((v) => v / z)
}

const adjacentLoss: ThresholdLoss = (s, th, y, dTheta) => {
  const p = softmax(adjacentLogits(s, th))
  let ds = 0
  for (let k = 0; k < p.length; k++) {
    const g = p[k] - (k === y ? 1 : 0)
    ds += g * k
    for (let j = 0; j < k; j++) dTheta[j] -= g
  }
  return [-Math.log(Math.max(p[y], 1e-12)), ds]
}

export const adjacentProbs = (s: number, th: number[]) => softmax(adjacentLogits(s, th))

/**
 * Threshold losses of Rennie & Srebro with the logistic surrogate. All-threshold penalises every threshold on the wrong
 * side of s; immediate-threshold only the two that bound the true class.
 */
function thresholdLoss(all: boolean): ThresholdLoss {
  return (s, th, y, dTheta) => {
    let loss = 0
    let ds = 0
    for (let j = 0; j < th.length; j++) {
      if (!all && j !== y - 1 && j !== y) continue
      if (j < y) {
        // s should lie above θ_j.
        const z = th[j] - s
        loss += softplus(z)
        ds -= sigmoid(z)
        dTheta[j] += sigmoid(z)
      } else {
        const z = s - th[j]
        loss += softplus(z)
        ds += sigmoid(z)
        dTheta[j] -= sigmoid(z)
      }
    }
    return [loss, ds]
  }
}

/** P(y > k) = σ(s − θ_k), differenced into classes. Valid whenever the thresholds are ordered. */
export function exceedanceProbs(exceed: number[]): number[] {
  const q = [1, ...exceed, 0]
  return q.slice(0, -1).map((v, k) => v - q[k + 1])
}

/**
 * Fit w (no intercept) and K − 1 thresholds. `ordered` parametrises θ_0 and log-gaps, so θ stays increasing; the
 * continuation-ratio and adjacent-category models do not need the order.
 */
/** A model to fit: a starting point, its loss and how to turn optimal parameters into a fitted model. */
export type Problem = { init: number[]; lossGrad: LossGrad; rate?: number; build: (p: number[]) => Fitted }

/**
 * w (no intercept) and K − 1 thresholds. `ordered` parametrises θ_0 and log-gaps, so θ stays increasing; the
 * continuation-ratio and adjacent-category models do not need the order.
 */
function scoreModel(data: Sample, k: number, loss: ThresholdLoss, ordered: boolean) {
  const nT = k - 1
  const toTheta = (raw: number[]) => {
    if (!ordered) return raw.slice()
    const th = [raw[0]]
    for (let j = 1; j < nT; j++) th.push(th[j - 1] + Math.exp(raw[j]))
    return th
  }
  const init = [0, 0, ...(ordered ? [-1, ...Array(nT - 1).fill(Math.log(2 / (nT - 1)))] : linspaceArr(-1, 1, nT))]
  const n = data.y.length
  const lossGrad: LossGrad = (params, grad) => {
    const raw = params.slice(2)
    const th = toTheta(raw)
    const dTheta = Array(nT).fill(0)
    let total = 0
    for (let i = 0; i < n; i++) {
      const [x0, x1] = data.x[i]
      const [l, ds] = loss(params[0] * x0 + params[1] * x1, th, data.y[i], dTheta)
      total += l
      grad[0] += ds * x0
      grad[1] += ds * x1
    }
    grad[0] = grad[0] / n + L2 * params[0]
    grad[1] = grad[1] / n + L2 * params[1]
    if (ordered) {
      // θ_j = raw_0 + Σ_{i≤j, i≥1} exp(raw_i), so dL/draw_i = exp(raw_i) Σ_{j≥i} dL/dθ_j.
      let tail = 0
      for (let j = nT - 1; j >= 0; j--) {
        tail += dTheta[j] / n
        grad[2 + j] = j === 0 ? tail : Math.exp(raw[j]) * tail
      }
    } else {
      for (let j = 0; j < nT; j++) grad[2 + j] = dTheta[j] / n
    }
    return total / n
  }
  return { init, lossGrad, decode: (p: number[]) => ({ w: [p[0], p[1]] as Point, theta: toTheta(p.slice(2)) }) }
}

function linspaceArr(a: number, b: number, n: number) {
  return n === 1 ? [0] : Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
}

const withBias = (x: Point) => [x[0], x[1], 1]
const dot3 = (w: number[], offset: number, f: number[]) =>
  w[offset] * f[0] + w[offset + 1] * f[1] + w[offset + 2] * f[2]

/** Multinomial logistic regression: one weight vector per class, the order ignored. */
function multiclassProblem(data: Sample, k: number): Problem {
  const n = data.y.length
  return {
    init: Array(3 * k).fill(0),
    lossGrad: (params, grad) => {
      let total = 0
      for (let i = 0; i < n; i++) {
        const f = withBias(data.x[i])
        const pr = softmax(Array.from({ length: k }, (_, c) => dot3(params, 3 * c, f)))
        total -= Math.log(Math.max(pr[data.y[i]], 1e-12))
        for (let c = 0; c < k; c++) {
          const g = pr[c] - (c === data.y[i] ? 1 : 0)
          for (let d = 0; d < 3; d++) grad[3 * c + d] += (g * f[d]) / n
        }
      }
      for (let j = 0; j < params.length; j++) if (j % 3 !== 2) grad[j] += L2 * params[j]
      return total / n
    },
    build: (p) => {
      const probs = (x: Point) => softmax(Array.from({ length: k }, (_, c) => dot3(p, 3 * c, withBias(x))))
      return { probs, predict: (x) => argmax(probs(x)) }
    },
  }
}

/** Frank & Hall: K − 1 independent logistic regressions for y > k, differenced into classes. */
function binaryDecompositionProblem(data: Sample, k: number): Problem {
  const n = data.y.length
  return {
    init: Array(3 * (k - 1)).fill(0),
    lossGrad: (params, grad) => {
      let total = 0
      for (let i = 0; i < n; i++) {
        const f = withBias(data.x[i])
        for (let j = 0; j < k - 1; j++) {
          const z = dot3(params, 3 * j, f)
          const t = data.y[i] > j ? 1 : 0
          total += softplus(t ? -z : z)
          const g = sigmoid(z) - t
          for (let d = 0; d < 3; d++) grad[3 * j + d] += (g * f[d]) / n
        }
      }
      for (let j = 0; j < params.length; j++) if (j % 3 !== 2) grad[j] += L2 * params[j]
      return total / n
    },
    build: (p) => {
      const probs = (x: Point) =>
        exceedanceProbs(Array.from({ length: k - 1 }, (_, j) => sigmoid(dot3(p, 3 * j, withBias(x)))))
      return { probs, predict: (x) => argmax(probs(x)) }
    },
  }
}

/**
 * Least squares on the class index, rounded. Its probabilities treat the fit as a cumulative probit with thresholds
 * fixed half-way between integers and the residual standard deviation as the noise scale.
 */
function fitRegression(data: Sample, k: number): Fitted {
  const A = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]
  const b = [0, 0, 0]
  data.x.forEach((x, i) => {
    const f = withBias(x)
    for (let r = 0; r < 3; r++) {
      b[r] += f[r] * data.y[i]
      for (let c = 0; c < 3; c++) A[r][c] += f[r] * f[c]
    }
  })
  const beta = solve3(A, b)
  const fit = (x: Point) => dot3(beta, 0, withBias(x))
  const rss = data.x.reduce((s, x, i) => s + (data.y[i] - fit(x)) ** 2, 0)
  const sigma = Math.sqrt(rss / (data.y.length - 3))
  const th = Array.from({ length: k - 1 }, (_, j) => j + 0.5)
  return {
    probs: (x) =>
      cumulativeProbs(
        (v: number) => normalCdf(v),
        fit(x) / sigma,
        th.map((t) => t / sigma),
      ),
    predict: (x) => clip(Math.round(fit(x)), 0, k - 1),
    score: fit,
    thresholds: th,
  }
}

function solve3(A: number[][], b: number[]): number[] {
  const m = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < 3; c++) {
    const pivot = m.slice(c).reduce((best, row, i) => (Math.abs(row[c]) > Math.abs(m[best][c]) ? c + i : best), c)
    ;[m[c], m[pivot]] = [m[pivot], m[c]]
    for (let r = 0; r < 3; r++) {
      if (r === c) continue
      const factor = m[r][c] / m[c][c]
      for (let j = c; j < 4; j++) m[r][j] -= factor * m[c][j]
    }
  }
  return m.map((row, i) => row[3] / row[i])
}

// ---------------------------------------------------------------------------------------------------------------
// StORM: a chain CRF over the up-to-k encoding
// ---------------------------------------------------------------------------------------------------------------

/** The up-to-k code of class y: K − 1 bits, bit n on when n < y. Class 3 of 5 (0-based 2) is 1100. */
export const upToK = (y: number, k: number) => Array.from({ length: k - 1 }, (_, n) => (n < y ? 1 : 0))

const logSumExp = (a: number, b: number) => {
  if (a === -Infinity) return b
  if (b === -Infinity) return a
  const m = Math.max(a, b)
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m))
}

/** Log node potentials ln[n][b] and log edge potentials le[n][u][v] of one input. */
type Potentials = { ln: number[][]; le: number[][][] }

/** Forward–backward in log space: log α, log β and log Z, with α_1 = β_{K−1} = 1. */
function forwardBackward({ ln, le }: Potentials) {
  const bits = ln.length
  const la: number[][] = [[0, 0]]
  for (let n = 0; n < bits - 1; n++)
    la.push([0, 1].map((v) => logSumExp(la[n][0] + ln[n][0] + le[n][0][v], la[n][1] + ln[n][1] + le[n][1][v])))
  const lb: number[][] = Array(bits)
  lb[bits - 1] = [0, 0]
  for (let n = bits - 2; n >= 0; n--)
    lb[n] = [0, 1].map((u) =>
      logSumExp(le[n][u][0] + ln[n + 1][0] + lb[n + 1][0], le[n][u][1] + ln[n + 1][1] + lb[n + 1][1]),
    )
  const logZ = logSumExp(la[0][0] + ln[0][0] + lb[0][0], la[0][1] + ln[0][1] + lb[0][1])
  return { la, lb, logZ }
}

/** log Z with some bit values forbidden: `allowed[n][b]` false removes value b at bit n. */
function clampedLogZ(p: Potentials, allowed: boolean[][]): number {
  const ln = p.ln.map((row, n) => row.map((v, b) => (allowed[n][b] ? v : -Infinity)))
  return forwardBackward({ ln, le: p.le }).logZ
}

export type StormModel = {
  k: number
  /** P(y = k | x) for each class: the probability of its valid code. */
  classProbs: (x: Point) => number[]
  /** P(bit n = 1 | x) for each bit. */
  bitMarginals: (x: Point) => number[]
  /**
   * P(a ≤ y ≤ b | x, the code is valid), 0-based classes: the class probabilities from a to b over their total. Far
   * from the data the chain puts most of its mass on invalid codes, and only this conditional stays meaningful there.
   */
  interval: (x: Point, a: number, b: number) => number
  /**
   * The raw chain marginal P(bit a−1 on, bit b off | x), which equals P(a ≤ y ≤ b | x) on valid codes but also counts
   * every invalid code with those two bits.
   */
  bitInterval: (x: Point, a: number, b: number) => number
  /** The probability of codes that encode no class. */
  invalidMass: (x: Point) => number
}

// ---------------------------------------------------------------------------------------------------------------
// Feature maps for StORM's potentials
// ---------------------------------------------------------------------------------------------------------------

/** A map from the plane to the features that the potentials are linear in; the last feature is the constant 1. */
export type FeatureMap = (x: Point) => number[]

export type Features = 'linear' | 'poly2' | 'poly3' | 'nystrom'

/** Landmarks of the Nyström map, sampled from the training points with a fixed seed. */
export const NYSTROM_LANDMARKS = 64

/** Standardise raw features to zero mean and unit variance on the training set, then append the constant. */
function standardised(raw: (x: Point) => number[], train: Point[]): FeatureMap {
  const rows = train.map(raw)
  const D = rows[0].length
  const mean = Array.from({ length: D }, (_, d) => rows.reduce((s, r) => s + r[d], 0) / rows.length)
  const sd = Array.from({ length: D }, (_, d) =>
    Math.max(Math.sqrt(rows.reduce((s, r) => s + (r[d] - mean[d]) ** 2, 0) / rows.length), 1e-9),
  )
  return (x) => [...raw(x).map((v, d) => (v - mean[d]) / sd[d]), 1]
}

/** Every monomial x₁^a x₂^b with 1 ≤ a + b ≤ degree: 5 features at degree 2, 9 at degree 3. */
function polynomial(degree: number, train: Point[]): FeatureMap {
  const powers: [number, number][] = []
  for (let total = 1; total <= degree; total++) for (let a = total; a >= 0; a--) powers.push([a, total - a])
  return standardised(([u, v]) => powers.map(([a, b]) => u ** a * v ** b), train)
}

/**
 * The Nyström approximation of an RBF kernel: m landmarks z_j drawn from the training points, the lengthscale set to
 * the median distance between landmarks, and features L⁻¹k(x) with K_mm = LLᵀ, so that the features' inner products
 * equal k(x)ᵀK_mm⁻¹k(x′). This is K_nm K_mm^{−1/2} up to a rotation, which a linear model does not notice.
 */
function nystrom(train: Point[], m = NYSTROM_LANDMARKS, seed = 3): FeatureMap {
  const r = stream(seed)
  const order = train.map((_, i) => i).sort(() => uniform(r) - 0.5)
  const z = order.slice(0, Math.min(m, train.length)).map((i) => train[i])
  const dists = z
    .flatMap((a, i) => z.slice(i + 1).map((b) => Math.hypot(a[0] - b[0], a[1] - b[1])))
    .sort((a, b) => a - b)
  const ell = dists[Math.floor(dists.length / 2)] || 1
  const kern = (a: Point, b: Point) => Math.exp(-((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / (2 * ell * ell))
  const M = z.length
  // Cholesky of K_mm with a little jitter for numerical safety.
  const L: number[][] = Array.from({ length: M }, () => Array(M).fill(0) as number[])
  for (let i = 0; i < M; i++)
    for (let j = 0; j <= i; j++) {
      let s = kern(z[i], z[j]) + (i === j ? 1e-6 : 0)
      for (let q = 0; q < j; q++) s -= L[i][q] * L[j][q]
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j]
    }
  const raw = (x: Point) => {
    const kx = z.map((zj) => kern(x, zj))
    const out: number[] = []
    for (let i = 0; i < M; i++) {
      let s = kx[i]
      for (let q = 0; q < i; q++) s -= L[i][q] * out[q]
      out.push(s / L[i][i])
    }
    return out
  }
  return standardised(raw, train)
}

export function featureMap(features: Features, train: Point[]): FeatureMap {
  switch (features) {
    case 'linear':
      return withBias
    case 'poly2':
      return polynomial(2, train)
    case 'poly3':
      return polynomial(3, train)
    case 'nystrom':
      return nystrom(train)
  }
}

/** The default ℓ2 strength of StORM's weights. */
export const STORM = { l2: 1e-3 }

/**
 * The ℓ2 strength per feature map. The 65 Nyström features overfit 100 training points at 1e-3 (held-out MAE 0.11 on
 * the default spiral against 0.08 at 1e-2); the polynomial maps are small enough for the default.
 */
export const FEATURE_L2: Record<Features, number> = { linear: 1e-3, poly2: 1e-3, poly3: 1e-3, nystrom: 1e-2 }

/**
 * Twomey et al.'s StORM: node potentials ψ_n = exp{W_n x} and edge potentials Ψ_n = exp{U_n x}, with no weight sharing
 * along the chain, fitted by gradient descent on the negative log-likelihood. Forward–backward gives the node and edge
 * marginals that the gradient needs in O(K) per example. The potentials are linear in the features φ(x): the
 * inputs and a constant by default, or a polynomial or Nyström expansion of them.
 */
function stormProblem(data: Sample, k: number, phi: FeatureMap = withBias, l2 = STORM.l2): Problem {
  const feats = data.x.map(phi)
  const D = feats[0].length
  const dotD = (w: number[], offset: number, f: number[]) => {
    let s = 0
    for (let d = 0; d < D; d++) s += w[offset + d] * f[d]
    return s
  }
  const bits = k - 1
  const nodeAt = (n: number, b: number) => D * (2 * n + b)
  const edgeAt = (e: number, u: number, v: number) => D * (2 * bits) + D * (4 * e + 2 * u + v)
  const dim = D * (2 * bits + 4 * (bits - 1))
  const potentials = (params: number[], f: number[]): Potentials => ({
    ln: Array.from({ length: bits }, (_, n) => [dotD(params, nodeAt(n, 0), f), dotD(params, nodeAt(n, 1), f)]),
    le: Array.from({ length: bits - 1 }, (_, e) =>
      [0, 1].map((u) => [dotD(params, edgeAt(e, u, 0), f), dotD(params, edgeAt(e, u, 1), f)]),
    ),
  })
  const codes = Array.from({ length: k }, (_, y) => upToK(y, k))
  const n = data.y.length
  const lossGrad: LossGrad = (params, grad) => {
    let total = 0
    for (let i = 0; i < n; i++) {
      const f = feats[i]
      const pot = potentials(params, f)
      const { la, lb, logZ } = forwardBackward(pot)
      const code = codes[data.y[i]]
      let score = 0
      for (let m = 0; m < bits; m++) {
        score += pot.ln[m][code[m]]
        for (let b = 0; b < 2; b++) {
          const g = (Math.exp(la[m][b] + pot.ln[m][b] + lb[m][b] - logZ) - (code[m] === b ? 1 : 0)) / n
          const o = nodeAt(m, b)
          for (let d = 0; d < D; d++) grad[o + d] += g * f[d]
        }
      }
      for (let e = 0; e < bits - 1; e++) {
        score += pot.le[e][code[e]][code[e + 1]]
        for (let u = 0; u < 2; u++)
          for (let v = 0; v < 2; v++) {
            const marginal = Math.exp(
              la[e][u] + pot.ln[e][u] + pot.le[e][u][v] + pot.ln[e + 1][v] + lb[e + 1][v] - logZ,
            )
            const g = (marginal - (code[e] === u && code[e + 1] === v ? 1 : 0)) / n
            const o = edgeAt(e, u, v)
            for (let d = 0; d < D; d++) grad[o + d] += g * f[d]
          }
      }
      total += logZ - score
    }
    let penalty = 0
    for (let j = 0; j < dim; j++) {
      grad[j] += l2 * params[j]
      penalty += params[j] * params[j]
    }
    return total / n + (l2 / 2) * penalty
  }
  const model = (p: number[]): StormModel => {
    const at = (x: Point) => {
      const pot = potentials(p, phi(x))
      return { pot, ...forwardBackward(pot) }
    }
    const classProbs = (x: Point) => {
      const { pot, logZ } = at(x)
      return codes.map((code) => {
        let score = 0
        code.forEach((b, m) => (score += pot.ln[m][b]))
        for (let e = 0; e < bits - 1; e++) score += pot.le[e][code[e]][code[e + 1]]
        return Math.exp(score - logZ)
      })
    }
    return {
      k,
      classProbs,
      bitMarginals: (x) => {
        const { pot, la, lb, logZ } = at(x)
        return pot.ln.map((row, m) => Math.exp(la[m][1] + row[1] + lb[m][1] - logZ))
      },
      interval: (x, a, b) => {
        const probs = classProbs(x)
        const total = probs.reduce((sum, q) => sum + q, 0)
        return probs.slice(a, b + 1).reduce((sum, q) => sum + q, 0) / total
      },
      invalidMass: (x) => 1 - classProbs(x).reduce((sum, q) => sum + q, 0),
      bitInterval: (x, a, b) => {
        const { pot, logZ } = at(x)
        const allowed = pot.ln.map((_, m) => [!(a > 0 && m === a - 1), !(b < k - 1 && m === b)])
        return Math.exp(clampedLogZ(pot, allowed) - logZ)
      },
    }
  }
  return {
    init: Array(dim).fill(0),
    lossGrad,
    // More parameters than the other models and no weight sharing: a larger step reaches convergence sooner.
    rate: 0.1,
    build: (p) => {
      const storm = model(p)
      return {
        probs: storm.classProbs,
        predict: (x) => argmax(storm.classProbs(x)),
        invalidMass: storm.invalidMass,
        storm,
      }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Hinge threshold loss (support vector ordinal regression) and small neural networks
// ---------------------------------------------------------------------------------------------------------------

/**
 * The all-threshold hinge loss of SVOR with implicit constraints: max(0, 1 − margin) at every threshold, where the
 * margin is s − θ_j for thresholds below the true class and θ_j − s above it. Its subgradient drives Adam.
 */
const hingeAllThreshold: ThresholdLoss = (s, th, y, dTheta) => {
  let loss = 0
  let ds = 0
  for (let j = 0; j < th.length; j++) {
    const below = j < y
    const margin = below ? s - th[j] : th[j] - s
    if (margin >= 1) continue
    loss += 1 - margin
    ds += below ? -1 : 1
    dTheta[j] += below ? 1 : -1
  }
  return [loss, ds]
}

/** Units in the hidden layer of the neural models: enough to follow the spiral, small enough to fit in the browser. */
export const HIDDEN = 24

/**
 * An ordinal output head on the hidden layer h. `free` heads have a weight vector per output; the `shared` head (CORAL)
 * has one weight vector and a bias per output. `loss` returns the loss and dL/dz for the outputs z; `probs` and
 * `predict` read the outputs.
 */
type Head = {
  outputs: number
  shared: boolean
  loss: (z: number[], y: number) => [number, number[]]
  probs: (z: number[]) => number[]
  predict?: (z: number[]) => number
}

const bce = (z: number, t: number): [number, number] => [softplus(t ? -z : z), sigmoid(z) - t]

/** Count of binary outputs above ½: the rank rule of OR-CNN, CORAL and CORN. */
const countRule = (q: number[]) => q.filter((v) => v > 0.5).length

function heads(k: number): Record<'orcnn' | 'coral' | 'corn' | 'sord', Head> {
  const binary = (z: number[], y: number): [number, number[]] => {
    let loss = 0
    const dz = z.map((zj, j) => {
      const [l, g] = bce(zj, y > j ? 1 : 0)
      loss += l
      return g
    })
    return [loss, dz]
  }
  const exceed = (z: number[]) => exceedanceProbs(z.map((v: number) => sigmoid(v)))
  // CORN: output j estimates P(y > j | y > j − 1) and is trained only on the examples that reached level j.
  const cornExceed = (z: number[]) => {
    let run = 1
    return z.map((zj) => (run *= sigmoid(zj)))
  }
  return {
    orcnn: {
      outputs: k - 1,
      shared: false,
      loss: binary,
      probs: exceed,
      predict: (z) => countRule(z.map((v: number) => sigmoid(v))),
    },
    coral: {
      outputs: k - 1,
      shared: true,
      loss: binary,
      probs: exceed,
      predict: (z) => countRule(z.map((v: number) => sigmoid(v))),
    },
    corn: {
      outputs: k - 1,
      shared: false,
      loss: (z, y) => {
        let loss = 0
        const dz = z.map((zj, j) => {
          if (y < j) return 0
          const [l, g] = bce(zj, y > j ? 1 : 0)
          loss += l
          return g
        })
        return [loss, dz]
      },
      probs: (z) => exceedanceProbs(cornExceed(z)),
      predict: (z) => countRule(cornExceed(z)),
    },
    sord: {
      outputs: k,
      shared: false,
      loss: (z, y) => {
        // Soft target t_c ∝ exp(−|c − y|): cross-entropy −Σ t_c log p_c, with gradient p − t in the logits.
        const t = softmax(z.map((_, c) => -Math.abs(c - y)))
        const p = softmax(z)
        return [-t.reduce((s, tc, c) => s + tc * Math.log(Math.max(p[c], 1e-12)), 0), p.map((pc, c) => pc - t[c])]
      },
      probs: softmax,
    },
  }
}

/**
 * A one-hidden-layer network, h = tanh(W₁x + b₁), with an ordinal head, fitted by backpropagation. Weights start small
 * and seeded, so every fit of a spec is the same.
 */
function mlpProblem(data: Sample, head: Head, seed = 7): Problem {
  const H = HIDDEN
  const out = head.outputs
  const inDim = 3 * H // W₁ (H × 2) and b₁
  const headDim = head.shared ? H + out : out * (H + 1)
  const r = stream(seed)
  const init = [...Array.from({ length: 2 * H }, () => 0.8 * normal(r)), ...Array(H).fill(0), ...Array(headDim).fill(0)]
  const n = data.y.length
  const hidden = (p: number[], x: Point) =>
    Array.from({ length: H }, (_, u) => Math.tanh(p[2 * u] * x[0] + p[2 * u + 1] * x[1] + p[2 * H + u]))
  const outputs = (p: number[], h: number[]) =>
    Array.from({ length: out }, (_, j) => {
      if (head.shared) return h.reduce((s, hu, u) => s + p[inDim + u] * hu, 0) + p[inDim + H + j]
      const o = inDim + j * (H + 1)
      return h.reduce((s, hu, u) => s + p[o + u] * hu, 0) + p[o + H]
    })
  const lossGrad: LossGrad = (p, grad) => {
    let total = 0
    for (let i = 0; i < n; i++) {
      const x = data.x[i]
      const h = hidden(p, x)
      const z = outputs(p, h)
      const [l, dz] = head.loss(z, data.y[i])
      total += l
      const dh = Array(H).fill(0) as number[]
      for (let j = 0; j < out; j++) {
        const g = dz[j] / n
        if (g === 0) continue
        if (head.shared) {
          for (let u = 0; u < H; u++) {
            grad[inDim + u] += g * h[u]
            dh[u] += g * p[inDim + u]
          }
          grad[inDim + H + j] += g
        } else {
          const o = inDim + j * (H + 1)
          for (let u = 0; u < H; u++) {
            grad[o + u] += g * h[u]
            dh[u] += g * p[o + u]
          }
          grad[o + H] += g
        }
      }
      for (let u = 0; u < H; u++) {
        const g = dh[u] * (1 - h[u] * h[u])
        grad[2 * u] += g * x[0]
        grad[2 * u + 1] += g * x[1]
        grad[2 * H + u] += g
      }
    }
    let penalty = 0
    for (let j = 0; j < p.length; j++) {
      grad[j] += L2 * p[j]
      penalty += p[j] * p[j]
    }
    return total / n + (L2 / 2) * penalty
  }
  return {
    init,
    lossGrad,
    build: (p) => {
      const z = (x: Point) => outputs(p, hidden(p, x))
      const probs = (x: Point) => head.probs(z(x))
      return { probs, predict: (x) => (head.predict ? head.predict(z(x)) : argmax(probs(x))) }
    },
  }
}

/**
 * Li and Lin's reduction: one binary classifier g(x, k) on the extended examples ((x, e_k), 1[y > k]). A linear g would
 * be the logistic all-threshold model, so here g is a small network whose input is x together with the one-hot e_k,
 * which lets every split bend its own way. The rank is 1 + Σ_k 1[g(x, k) > 0].
 */
function liLinProblem(data: Sample, k: number, seed = 11): Problem {
  const H = HIDDEN
  const inputs = 2 + (k - 1)
  const r = stream(seed)
  const dim = H * inputs + H + H + 1
  const init = [...Array.from({ length: H * inputs }, () => 0.8 * normal(r)), ...Array(2 * H + 1).fill(0)]
  const n = data.y.length
  const W2 = H * inputs + H
  const forward = (p: number[], x: Point, j: number) => {
    const h = Array.from({ length: H }, (_, u) =>
      Math.tanh(p[u * inputs] * x[0] + p[u * inputs + 1] * x[1] + p[u * inputs + 2 + j] + p[H * inputs + u]),
    )
    return { h, g: h.reduce((s, hu, u) => s + p[W2 + u] * hu, 0) + p[W2 + H] }
  }
  const lossGrad: LossGrad = (p, grad) => {
    let total = 0
    const m = n * (k - 1)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < k - 1; j++) {
        const x = data.x[i]
        const { h, g } = forward(p, x, j)
        const [l, dg] = bce(g, data.y[i] > j ? 1 : 0)
        total += l
        const d = dg / m
        for (let u = 0; u < H; u++) {
          grad[W2 + u] += d * h[u]
          const dz = d * p[W2 + u] * (1 - h[u] * h[u])
          grad[u * inputs] += dz * x[0]
          grad[u * inputs + 1] += dz * x[1]
          grad[u * inputs + 2 + j] += dz
          grad[H * inputs + u] += dz
        }
        grad[W2 + H] += d
      }
    let penalty = 0
    for (let q = 0; q < dim; q++) {
      grad[q] += L2 * p[q]
      penalty += p[q] * p[q]
    }
    return total / m + (L2 / 2) * penalty
  }
  return {
    init,
    lossGrad,
    build: (p) => {
      const exceed = (x: Point) => Array.from({ length: k - 1 }, (_, j) => sigmoid(forward(p, x, j).g))
      return { probs: (x) => exceedanceProbs(exceed(x)), predict: (x) => countRule(exceed(x)) }
    },
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Fitting with a cache
// ---------------------------------------------------------------------------------------------------------------

function scoreFitted(
  w: Point,
  theta: number[],
  probsOf: (s: number, th: number[]) => number[],
  byInterval: boolean,
): Fitted {
  const score = (x: Point) => w[0] * x[0] + w[1] * x[1]
  const probs = (x: Point) => probsOf(score(x), theta)
  return {
    probs,
    predict: (x) => (byInterval ? intervalOf(score(x), theta) : argmax(probs(x))),
    score,
    thresholds: theta,
  }
}

const exceedFromScore = (s: number, th: number[]) => exceedanceProbs(th.map((t) => sigmoid(s - t)))

function scoreProblem(
  data: Sample,
  k: number,
  loss: ThresholdLoss,
  ordered: boolean,
  probsOf: (s: number, th: number[]) => number[],
  byInterval: boolean,
): Problem {
  const m = scoreModel(data, k, loss, ordered)
  return {
    init: m.init,
    lossGrad: m.lossGrad,
    build: (p) => {
      const { w, theta } = m.decode(p)
      return scoreFitted(w, theta, probsOf, byInterval)
    },
  }
}

/** The optimisation problem of a model on a dataset, or the fitted model itself when it has a closed form. */
function problemFor(spec: DataSpec, model: ModelId): Problem | Fitted {
  const { train, k } = dataset(spec)
  switch (model) {
    case 'cumulative-logit':
      return scoreProblem(
        train,
        k,
        cumulativeLoss((v: number) => sigmoid(v), logisticPdf),
        true,
        (s, th) => cumulativeProbs((v: number) => sigmoid(v), s, th),
        false,
      )
    case 'cumulative-probit':
      return scoreProblem(
        train,
        k,
        cumulativeLoss(
          (v: number) => normalCdf(v),
          (v: number) => normalPdf(v),
        ),
        true,
        (s, th) => cumulativeProbs((v: number) => normalCdf(v), s, th),
        false,
      )
    case 'continuation-ratio':
      return scoreProblem(train, k, continuationLoss, false, continuationProbs, false)
    case 'adjacent-category':
      return scoreProblem(train, k, adjacentLoss, false, adjacentProbs, false)
    case 'immediate-threshold':
      return scoreProblem(train, k, thresholdLoss(false), true, exceedFromScore, true)
    case 'all-threshold':
      return scoreProblem(train, k, thresholdLoss(true), true, exceedFromScore, true)
    case 'binary-decomposition':
      return binaryDecompositionProblem(train, k)
    case 'storm':
      return stormProblem(train, k)
    case 'storm-poly2':
    case 'storm-poly3':
    case 'storm-nystrom': {
      const features = model.slice('storm-'.length) as Features
      return stormProblem(train, k, featureMap(features, train.x), FEATURE_L2[features])
    }
    case 'multiclass':
      return multiclassProblem(train, k)
    case 'regression':
      return fitRegression(train, k)
    case 'svor':
      return { ...scoreProblem(train, k, hingeAllThreshold, true, exceedFromScore, true), rate: 0.02 }
    case 'li-lin':
      return liLinProblem(train, k)
    case 'orcnn':
    case 'coral':
    case 'corn':
    case 'sord':
      return mlpProblem(train, heads(k)[model])
  }
}

const fitCache = new Map<string, Fitted>()
/** The last solution per curve, K and model: a warm start when only the noise, seed or size changes. */
const warmCache = new Map<string, number[]>()

const fitKey = (spec: DataSpec, model: ModelId) => `${specKey(spec)}/${model}`
const warmKey = (spec: DataSpec, model: ModelId) => `${spec.shape}/${spec.k}/${model}`

/** The cached fit of a model to a spec, if there is one. */
export const cachedFit = (spec: DataSpec, model: ModelId) => fitCache.get(fitKey(spec, model))

/**
 * A fit in progress: `step(ms)` optimises for up to `ms` milliseconds and returns the fitted model once converged.
 * Closed-form models and cached fits are returned at once.
 */
export function startFit(spec: DataSpec, model: ModelId): { step: (ms: number) => Fitted | null; steps: () => number } {
  const hit = cachedFit(spec, model)
  if (hit) return { step: () => hit, steps: () => 0 }
  const problem = problemFor(spec, model)
  if (!('lossGrad' in problem)) {
    remember(fitCache, fitKey(spec, model), problem)
    return { step: () => problem, steps: () => 0 }
  }
  const warm = warmCache.get(warmKey(spec, model))
  const o = new Optimiser(warm ?? problem.init, problem.lossGrad, problem.rate)
  return {
    step: (ms) => {
      if (!o.run(ms)) return null
      warmCache.set(warmKey(spec, model), o.p.slice())
      return remember(fitCache, fitKey(spec, model), problem.build(o.p))
    },
    steps: () => o.steps,
  }
}

/** Fit a model to a dataset's training sample, to convergence, once per spec while it stays in the cache. */
export function fitModel(spec: DataSpec, model: ModelId): Fitted {
  return startFit(spec, model).step(Infinity)!
}

// ---------------------------------------------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------------------------------------------

export type OrdinalMetrics = { accuracy: number; mae: number; macroMae: number; qwk: number }

/** Accuracy, mean absolute error, macro-averaged MAE over the classes present, and quadratic weighted kappa. */
export function ordinalMetrics(y: number[], yhat: number[], k: number): OrdinalMetrics {
  const n = y.length
  const confusion = Array.from({ length: k }, () => Array(k).fill(0) as number[])
  y.forEach((t, i) => confusion[t][yhat[i]]++)
  const accuracy = y.filter((t, i) => t === yhat[i]).length / n
  const mae = y.reduce((s, t, i) => s + Math.abs(t - yhat[i]), 0) / n
  const perClass = confusion
    .map((row, t) => {
      const count = row.reduce((a, b) => a + b, 0)
      return count ? row.reduce((s, c, j) => s + c * Math.abs(t - j), 0) / count : null
    })
    .filter((v): v is number => v !== null)
  const macroMae = perClass.reduce((a, b) => a + b, 0) / perClass.length
  const rows = confusion.map((row) => row.reduce((a, b) => a + b, 0))
  const cols = confusion[0].map((_, j) => confusion.reduce((s, row) => s + row[j], 0))
  let observed = 0
  let expected = 0
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      const w = (i - j) ** 2
      observed += w * confusion[i][j]
      expected += (w * rows[i] * cols[j]) / n
    }
  return { accuracy, mae, macroMae, qwk: expected > 0 ? 1 - observed / expected : 1 }
}

// ---------------------------------------------------------------------------------------------------------------
// One-dimensional helpers
// ---------------------------------------------------------------------------------------------------------------

/**
 * Binary logistic regression in one feature by Newton's method: returns the intercept, slope and their 2 × 2
 * covariance (the inverse Fisher information).
 */
export function logistic1d(x: number[], t: number[]) {
  let a = 0
  let b = 0
  let cov = [
    [0, 0],
    [0, 0],
  ]
  for (let it = 0; it < 30; it++) {
    let g0 = 0
    let g1 = 0
    let h00 = 1e-6
    let h01 = 0
    let h11 = 1e-6
    for (let i = 0; i < x.length; i++) {
      const p = sigmoid(a + b * x[i])
      const r = t[i] - p
      const w = p * (1 - p)
      g0 += r
      g1 += r * x[i]
      h00 += w
      h01 += w * x[i]
      h11 += w * x[i] * x[i]
    }
    const det = h00 * h11 - h01 * h01
    cov = [
      [h11 / det, -h01 / det],
      [-h01 / det, h00 / det],
    ]
    const da = cov[0][0] * g0 + cov[0][1] * g1
    const db = cov[1][0] * g0 + cov[1][1] * g1
    a += da
    b += db
    if (Math.abs(da) + Math.abs(db) < 1e-9) break
  }
  return { a, b, cov }
}

/**
 * One feature, four classes, with a separate slope for each cumulative split: P(y > j | x) = σ(β_j x − θ_j) with
 * β_j = 1.2(1 + δ(j − 1)) for j = 0, 1, 2 and θ = (−2.5, 0, 2.5). With δ = 0 the splits are parallel and proportional
 * odds holds. For |δ| ≤ 0.6 the three curves do not cross on [−3, 3], so a single uniform draw u sets the class as the
 * number of splits with u < P(y > j | x), and every split has exactly its stated probability.
 */
export function nonParallelSample(n: number, delta: number, seed = 5): { x: number[]; y: number[] } {
  const r = stream(seed)
  const x: number[] = []
  const y: number[] = []
  const theta = [-2.5, 0, 2.5]
  for (let i = 0; i < n; i++) {
    const xi = -3 + 6 * uniform(r)
    const u = uniform(r)
    x.push(xi)
    y.push(theta.filter((t, j) => u < sigmoid(1.2 * (1 + delta * (j - 1)) * xi - t)).length)
  }
  return { x, y }
}

/** A cumulative logit or all-threshold (CORAL) fit to one feature: the slope w and the ordered thresholds. */
export function fitShared1d(x: number[], y: number[], k: number, loss: 'cumulative-logit' | 'all-threshold') {
  const sample: Sample = { x: x.map((v) => [v, 0] as Point), y }
  const l = loss === 'cumulative-logit' ? cumulativeLoss((v: number) => sigmoid(v), logisticPdf) : thresholdLoss(true)
  const m = scoreModel(sample, k, l, true)
  const { w, theta } = m.decode(minimise(m.init, m.lossGrad))
  return { w: w[0], theta }
}
