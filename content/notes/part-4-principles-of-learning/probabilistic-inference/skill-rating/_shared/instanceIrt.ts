import { normal, stream, uniform } from 'aifn-compute/foundation/random'
import { logGamma } from 'aifn-compute/numerics/special'
/**
 * Item response theory on classifiers, small enough to recompute on every slider move. A two-class dataset in two
 * features is simulated with some test labels flipped; a population of simple classifiers of varying skill is trained
 * on clean training data; each classifier's predicted probability of every test instance's given label is recorded.
 * Two IRT models are then fitted with test instances as items and classifiers as respondents:
 * - the 2PL on correct / incorrect (probability above one half), by joint maximum a posteriori estimation;
 * - the β³ model on the probabilities themselves, by maximum a posteriori estimation with Adam.
 * Used by the notes on IRT for machine learning and on the β³ model.
 */

export type Point = [number, number]

const sq = (v: number) => v * v

export type Dataset = {
  train: { x: Point[]; y: number[] }
  /** Test instances: features, the label they were generated with, the label given (after flips), and the flip flag. */
  test: { x: Point[]; yTrue: number[]; y: number[]; flipped: boolean[]; bayes: number[] }
}

/** P(class 1 | x) under the generating model: the Bayes-optimal classifier, which no trained model can beat. */
export function bayesPosterior(p: Point): number {
  const logDensity = (c: number) =>
    -0.5 * (sq((p[0] - MEANS[c][0]) / SDS[c][0]) + sq((p[1] - MEANS[c][1]) / SDS[c][1])) -
    Math.log(SDS[c][0] * SDS[c][1])
  return 1 / (1 + Math.exp(logDensity(0) - logDensity(1)))
}

const MEANS: [Point, Point] = [
  [-1.1, 0],
  [1.1, 0],
]
/** Standard deviations per feature: class 0 is round, class 1 is stretched along the first feature. */
const SDS: [Point, Point] = [
  [0.7, 0.7],
  [1.3, 0.45],
]

/** Simulate balanced training and test sets, then flip the labels of a fraction `noise` of the test instances. */
export function simulate(seed: number, noise: number, nTrain = 100, nTest = 70): Dataset {
  const r = stream(seed)
  const draw = (n: number) => {
    const x: Point[] = []
    const y: number[] = []
    for (let i = 0; i < n; i++) {
      const c = i % 2
      x.push([MEANS[c][0] + SDS[c][0] * normal(r), MEANS[c][1] + SDS[c][1] * normal(r)])
      y.push(c)
    }
    return { x, y }
  }
  const train = draw(nTrain)
  const t = draw(nTest)
  // Choose the flipped instances by a seeded shuffle, so a higher noise rate flips a superset of a lower one.
  const order = t.x.map((_, i) => i)
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(uniform(r) * (i + 1))
    ;[order[i], order[j]] = [order[j], order[i]]
  }
  const nFlip = Math.round(noise * nTest)
  const flipped = t.x.map(() => false)
  for (let k = 0; k < nFlip; k++) flipped[order[k]] = true
  const y = t.y.map((c, i) => (flipped[i] ? 1 - c : c))
  // The Bayes-optimal probability of each given label: below one half, the label disagrees with the features.
  const bayes = t.x.map((p, i) => (y[i] === 1 ? bayesPosterior(p) : 1 - bayesPosterior(p)))
  return { train, test: { x: t.x, yTrue: t.y, y, flipped, bayes } }
}

/** A trained classifier: returns P(class 1 | x). */
type Model = (x: Point) => number

const clampP = (p: number) => Math.min(1 - 1e-6, Math.max(1e-6, p))

/** Gaussian class-conditional classifier. `shared` pools the covariance (LDA); `diagonal` drops correlations (NB). */
function gaussianModel(x: Point[], y: number[], shared: boolean, diagonal: boolean): Model {
  const stats = [0, 1].map((c) => {
    const pts = x.filter((_, i) => y[i] === c)
    const m: Point = [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length]
    let sxx = 0
    let syy = 0
    let sxy = 0
    for (const p of pts) {
      sxx += sq(p[0] - m[0])
      syy += sq(p[1] - m[1])
      sxy += (p[0] - m[0]) * (p[1] - m[1])
    }
    const n = Math.max(pts.length - 1, 1)
    return { m, cov: [sxx / n + 0.01, diagonal ? 0 : sxy / n, syy / n + 0.01] as [number, number, number] }
  })
  if (shared) {
    const pooled = [0, 1, 2].map((k) => (stats[0].cov[k] + stats[1].cov[k]) / 2) as [number, number, number]
    stats[0].cov = pooled
    stats[1].cov = pooled
  }
  const logDensity = (p: Point, c: number) => {
    const [a, b, d] = stats[c].cov
    const det = a * d - b * b
    const dx = p[0] - stats[c].m[0]
    const dy = p[1] - stats[c].m[1]
    return -0.5 * ((d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det) - 0.5 * Math.log(det)
  }
  return (p) => clampP(1 / (1 + Math.exp(logDensity(p, 0) - logDensity(p, 1))))
}

/** Nearest centroid, with P(class 1) a softmax of negative half squared distances. */
function centroidModel(x: Point[], y: number[]): Model {
  const m = [0, 1].map((c) => {
    const pts = x.filter((_, i) => y[i] === c)
    return [pts.reduce((a, p) => a + p[0], 0) / pts.length, pts.reduce((a, p) => a + p[1], 0) / pts.length]
  })
  return (p) => {
    const d0 = sq(p[0] - m[0][0]) + sq(p[1] - m[0][1])
    const d1 = sq(p[0] - m[1][0]) + sq(p[1] - m[1][1])
    return clampP(1 / (1 + Math.exp((d1 - d0) / 2)))
  }
}

/** k nearest neighbours, with the class-1 share smoothed as (count + 1/2) / (k + 1). */
function knnModel(x: Point[], y: number[], k: number): Model {
  return (p) => {
    const d = x.map((q, i) => [sq(q[0] - p[0]) + sq(q[1] - p[1]), y[i]]).sort((a, b) => a[0] - b[0])
    const ones = d.slice(0, k).reduce((a, v) => a + v[1], 0)
    return (ones + 0.5) / (k + 1)
  }
}

/** A decision stump on one feature: the threshold with the fewest training errors, Laplace-smoothed leaves. */
function stumpModel(x: Point[], y: number[], f: 0 | 1): Model {
  const vals = [...new Set(x.map((p) => p[f]))].sort((a, b) => a - b)
  let best = { t: 0, err: Infinity }
  for (let i = 0; i + 1 < vals.length; i++) {
    const t = (vals[i] + vals[i + 1]) / 2
    let e = 0
    x.forEach((p, j) => (e += (p[f] > t ? 1 : 0) !== y[j] ? 1 : 0))
    const err = Math.min(e, x.length - e)
    if (err < best.err) best = { t, err }
  }
  const leaf = (side: boolean) => {
    const idx = x.map((p, j) => [p[f] > best.t === side, j] as const).filter(([s]) => s)
    const ones = idx.reduce((a, [, j]) => a + y[j], 0)
    return (ones + 1) / (idx.length + 2)
  }
  const hi = leaf(true)
  const lo = leaf(false)
  return (p) => (p[f] > best.t ? hi : lo)
}

type Family = 'QDA' | 'LDA' | 'NB' | 'centroid' | 'kNN' | 'stump x₁' | 'stump x₂' | 'random'

/**
 * The population, from strong to weak: families × training-set sizes n, three QDA models that see each test point
 * through Gaussian input noise of standard deviation `blur` (a smooth dial on skill), and two random guessers.
 */
const POPULATION: { family: Family; n: number; k?: number; blur?: number }[] = [
  { family: 'QDA', n: 100 },
  { family: 'QDA', n: 12 },
  { family: 'LDA', n: 100 },
  { family: 'NB', n: 8 },
  { family: 'centroid', n: 100 },
  { family: 'centroid', n: 4 },
  { family: 'kNN', n: 100, k: 1 },
  { family: 'kNN', n: 100, k: 15 },
  { family: 'kNN', n: 10, k: 3 },
  { family: 'stump x₁', n: 100 },
  { family: 'stump x₁', n: 6 },
  { family: 'stump x₂', n: 100 },
  { family: 'QDA', n: 100, blur: 0.75 },
  { family: 'QDA', n: 100, blur: 1.5 },
  { family: 'QDA', n: 100, blur: 3 },
  { family: 'random', n: 0 },
  { family: 'random', n: 0 },
]

export type Responses = {
  names: string[]
  /** probs[i][j]: classifier i's probability of test instance j's given label. */
  probs: number[][]
  /** correct[i][j]: 1 when that probability exceeds one half. */
  correct: number[][]
  accuracy: number[]
}

/** Train the population on (balanced) subsamples of the clean training set and score every test instance. */
export function respond(data: Dataset, seed: number): Responses {
  const r = stream(seed + 7919)
  const names: string[] = []
  const probs: number[][] = []
  for (const spec of POPULATION) {
    // A balanced subsample of size n: the first n/2 of a shuffle of each class.
    const pick = [0, 1].flatMap((c) => {
      const idx = data.train.y.flatMap((yy, i) => (yy === c ? [i] : []))
      for (let i = idx.length - 1; i > 0; i--) {
        const j = Math.floor(uniform(r) * (i + 1))
        ;[idx[i], idx[j]] = [idx[j], idx[i]]
      }
      return idx.slice(0, Math.max(2, Math.floor(spec.n / 2)))
    })
    const x = pick.map((i) => data.train.x[i])
    const y = pick.map((i) => data.train.y[i])
    let model: Model
    switch (spec.family) {
      case 'QDA':
        model = gaussianModel(x, y, false, false)
        break
      case 'LDA':
        model = gaussianModel(x, y, true, false)
        break
      case 'NB':
        model = gaussianModel(x, y, false, true)
        break
      case 'centroid':
        model = centroidModel(x, y)
        break
      case 'kNN':
        model = knnModel(x, y, spec.k ?? 1)
        break
      case 'stump x₁':
        model = stumpModel(x, y, 0)
        break
      case 'stump x₂':
        model = stumpModel(x, y, 1)
        break
      case 'random': {
        const u = data.test.x.map(() => uniform(r))
        let k = 0
        model = () => u[k++ % u.length]
        break
      }
    }
    const blur = spec.blur ?? 0
    const seen = data.test.x.map((p): Point => (blur ? [p[0] + blur * normal(r), p[1] + blur * normal(r)] : p))
    const label = spec.family === 'kNN' ? `${spec.k}-NN` : spec.family
    names.push(spec.family === 'random' ? 'random' : blur ? `${label}, input noise ${blur}` : `${label}, n = ${spec.n}`)
    probs.push(seen.map((q, j) => (data.test.y[j] === 1 ? model(q) : 1 - model(q))))
  }
  const correct = probs.map((row) => row.map((p): number => (p > 0.5 ? 1 : 0)))
  const accuracy = correct.map((row) => row.reduce((a, b) => a + b, 0) / row.length)
  return { names, probs, correct, accuracy }
}

/** Instance hardness: one minus the population's mean probability of the given label (Smith et al., 2014). */
export const instanceHardness = (probs: number[][]) =>
  probs[0].map((_, j) => 1 - probs.reduce((a, row) => a + row[j], 0) / probs.length)

// ---------------------------------------------------------------------------------------------------------------
// 2PL: P(correct) = σ(a_j (θ_i − b_j)), fitted in slope–intercept form σ(a_j θ_i + c_j) with b_j = −c_j / a_j.
// ---------------------------------------------------------------------------------------------------------------

export type TwoPL = { theta: number[]; a: number[]; b: number[]; c: number[] }

/** Priors of the joint MAP fit: θ ~ N(0, 1), a ~ N(A_MEAN, A_SD²), c ~ N(0, C_SD²). */
export const TWO_PL_PRIOR = { aMean: 0.5, aSd: 1, cSd: 3 }

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
/** Damped Newton: joint updates can overshoot when an item's responses are nearly separable in θ. */
const clampStep = (s: number) => Math.max(-0.5, Math.min(0.5, s))

/**
 * Joint maximum a posteriori fit of the 2PL by alternating Newton steps: each ability given the items, then each
 * item's (a, c) given the abilities (a two-parameter logistic regression with a Gaussian prior). Abilities start at
 * the standardised logit of accuracy, which fixes the orientation of the scale: able classifiers get high θ.
 */
export function fitTwoPL(x: number[][], rounds = 40): TwoPL {
  const M = x.length
  const N = x[0].length
  const { aMean, aSd, cSd } = TWO_PL_PRIOR
  const acc = x.map((row) => (row.reduce((s, v) => s + v, 0) + 0.5) / (N + 1))
  const lg = acc.map((p) => Math.log(p / (1 - p)))
  const mu = lg.reduce((s, v) => s + v, 0) / M
  const sd = Math.sqrt(lg.reduce((s, v) => s + sq(v - mu), 0) / M) || 1
  const theta = lg.map((v) => (v - mu) / sd)
  const a = new Array<number>(N).fill(1)
  const c = x[0].map((_, j) => {
    const p = (x.reduce((s, row) => s + row[j], 0) + 0.5) / (M + 1)
    return Math.log(p / (1 - p))
  })
  for (let round = 0; round < rounds; round++) {
    for (let j = 0; j < N; j++) {
      for (let step = 0; step < 3; step++) {
        let ga = -(a[j] - aMean) / (aSd * aSd)
        let gc = -c[j] / (cSd * cSd)
        let haa = 1 / (aSd * aSd)
        let hac = 0
        let hcc = 1 / (cSd * cSd)
        for (let i = 0; i < M; i++) {
          const p = sigmoid(a[j] * theta[i] + c[j])
          const w = p * (1 - p)
          const e = x[i][j] - p
          ga += e * theta[i]
          gc += e
          haa += w * theta[i] * theta[i]
          hac += w * theta[i]
          hcc += w
        }
        const det = haa * hcc - hac * hac
        a[j] += clampStep((hcc * ga - hac * gc) / det)
        c[j] += clampStep((haa * gc - hac * ga) / det)
      }
    }
    for (let i = 0; i < M; i++) {
      for (let step = 0; step < 3; step++) {
        let g = -theta[i]
        let h = 1
        for (let j = 0; j < N; j++) {
          const p = sigmoid(a[j] * theta[i] + c[j])
          g += a[j] * (x[i][j] - p)
          h += a[j] * a[j] * p * (1 - p)
        }
        theta[i] += clampStep(g / h)
      }
    }
  }
  const b = a.map((aj, j) => Math.max(-6, Math.min(6, -c[j] / (Math.abs(aj) < 1e-3 ? 1e-3 : aj))))
  return { theta, a, b, c }
}

export const iccTwoPL = (aj: number, cj: number, t: number) => sigmoid(aj * t + cj)

// ---------------------------------------------------------------------------------------------------------------
// β³-IRT (Chen et al., 2019): p_ij ~ Beta(α, β), α = (θ_i/δ_j)^a_j, β = ((1 − θ_i)/(1 − δ_j))^a_j.
// ---------------------------------------------------------------------------------------------------------------

export type BetaThree = { theta: number[]; delta: number[]; a: number[] }

/** Digamma ψ(x) for x > 0: recurrence up to x ≥ 6, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0
  while (x < 6) {
    r -= 1 / x
    x += 1
  }
  const f = 1 / (x * x)
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))
}

/** β³ log-density of one response p given ability, difficulty and discrimination. */
export function betaThreeLogDensity(p: number, theta: number, delta: number, a: number): number {
  const al = Math.pow(theta / delta, a)
  const be = Math.pow((1 - theta) / (1 - delta), a)
  return (al - 1) * Math.log(p) + (be - 1) * Math.log(1 - p) - (logGamma(al) + logGamma(be) - logGamma(al + be))
}

/** Correlation, across respondents, between responses to item j and each respondent's mean response to the others. */
export function itemRestCorrelation(x: number[][], j: number): number {
  const u = x.map((row) => row[j])
  const v = x.map((row) => (row.reduce((s, p) => s + p, 0) - row[j]) / (row.length - 1))
  const mu = u.reduce((s, p) => s + p, 0) / u.length
  const mv = v.reduce((s, p) => s + p, 0) / v.length
  let suv = 0
  let suu = 0
  let svv = 0
  for (let i = 0; i < u.length; i++) {
    suv += (u[i] - mu) * (v[i] - mv)
    suu += sq(u[i] - mu)
    svv += sq(v[i] - mv)
  }
  return suu > 0 && svv > 0 ? suv / Math.sqrt(suu * svv) : 0
}

/** Mean response (the item characteristic curve): σ(a (logit θ − logit δ)). */
export const iccBetaThree = (theta: number, delta: number, a: number) =>
  1 / (1 + Math.pow(delta / (1 - delta), a) * Math.pow(theta / (1 - theta), -a))

/** Responses are clipped into [P_MIN, 1 − P_MIN]: the Beta density is unbounded at 0 and 1. */
export const P_MIN = 0.01
/** Prior on discrimination, as in the paper: a ~ N(1, σ₀²) with σ₀ = 1. Abilities and difficulties ~ Beta(1, 1). */
export const BETA_THREE_A_SD = 1

/**
 * MAP fit of β³ by Adam on unconstrained u = logit θ, v = logit δ and a. The Beta(1, 1) prior on θ becomes the
 * logistic density on u, whose log-gradient is 1 − 2σ(u). As in the paper's coordinate ascent, discrimination is held
 * at its initial value while abilities and difficulties settle, then all three move together. Each discrimination
 * starts at +1 or −1 by the sign of the item's correlation with the classifiers' mean responses. Started at +1, a
 * mislabelled item gets stuck at a local optimum: very hard, with discrimination near zero.
 */
export function fitBetaThree(probs: number[][], iters = 500, warm = 150): BetaThree {
  const M = probs.length
  const N = probs[0].length
  const P = probs.map((row) => row.map((p) => Math.min(1 - P_MIN, Math.max(P_MIN, p))))
  const L1 = P.map((row) => row.map(Math.log))
  const L0 = P.map((row) => row.map((p) => Math.log(1 - p)))
  const logit = (p: number) => Math.log(p / (1 - p))
  const rowMean = P.map((row) => row.reduce((s, v) => s + v, 0) / N)
  const colMean = P[0].map((_, j) => P.reduce((s, row) => s + row[j], 0) / M)
  const params = [
    ...rowMean.map((m) => logit(Math.min(0.95, Math.max(0.05, m)))),
    ...colMean.map((m) => logit(Math.min(0.95, Math.max(0.05, 1 - m)))),
    ...P[0].map((_, j) => (itemRestCorrelation(P, j) < 0 ? -1 : 1)),
  ]
  const m1 = new Array<number>(params.length).fill(0)
  const m2 = new Array<number>(params.length).fill(0)
  const lr = 0.05
  const [b1, b2, eps] = [0.9, 0.999, 1e-8]
  for (let t = 1; t <= iters; t++) {
    const g = new Array<number>(params.length).fill(0)
    const th = params.slice(0, M).map(sigmoid)
    const de = params.slice(M, M + N).map(sigmoid)
    for (let i = 0; i < M; i++) {
      const lt = Math.log(th[i])
      const l1t = Math.log(1 - th[i])
      for (let j = 0; j < N; j++) {
        const aj = params[M + N + j]
        const ld = Math.log(de[j])
        const l1d = Math.log(1 - de[j])
        const al = Math.exp(aj * (lt - ld))
        const be = Math.exp(aj * (l1t - l1d))
        const dsum = digamma(al + be)
        const gAl = L1[i][j] - digamma(al) + dsum
        const gBe = L0[i][j] - digamma(be) + dsum
        // Chain rule through α = exp(a (log θ − log δ)) and β = exp(a (log(1 − θ) − log(1 − δ))).
        g[i] += gAl * al * aj * (1 - th[i]) - gBe * be * aj * th[i]
        g[M + j] += -gAl * al * aj * (1 - de[j]) + gBe * be * aj * de[j]
        g[M + N + j] += gAl * al * (lt - ld) + gBe * be * (l1t - l1d)
      }
    }
    for (let i = 0; i < M; i++) g[i] += 1 - 2 * th[i]
    for (let j = 0; j < N; j++) {
      g[M + j] += 1 - 2 * de[j]
      g[M + N + j] += -(params[M + N + j] - 1) / (BETA_THREE_A_SD * BETA_THREE_A_SD)
    }
    for (let k = 0; k < params.length; k++) {
      if (t <= warm && k >= M + N) continue
      m1[k] = b1 * m1[k] + (1 - b1) * g[k]
      m2[k] = b2 * m2[k] + (1 - b2) * g[k] * g[k]
      const mh = m1[k] / (1 - Math.pow(b1, t))
      const vh = m2[k] / (1 - Math.pow(b2, t))
      params[k] += (lr * mh) / (Math.sqrt(vh) + eps)
    }
  }
  return {
    theta: params.slice(0, M).map(sigmoid),
    delta: params.slice(M, M + N).map(sigmoid),
    a: params.slice(M + N),
  }
}

/** Everything the explorer draws, for one seed and noise rate. */
export function runExperiment(seed: number, noise: number) {
  const data = simulate(seed, noise)
  const resp = respond(data, seed)
  return {
    data,
    resp,
    hardness: instanceHardness(resp.probs),
    twoPL: fitTwoPL(resp.correct),
    betaThree: fitBetaThree(resp.probs),
  }
}
