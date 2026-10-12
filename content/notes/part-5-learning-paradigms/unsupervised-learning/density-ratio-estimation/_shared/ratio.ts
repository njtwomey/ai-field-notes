/**
 * The running problem of the density-ratio notes, and the estimators they compare, composed from engine primitives.
 *
 * The problem: a sample from the numerator p = N(m, s²) and a sample from the denominator q = N(0, 1), so the true
 * ratio r(x) = p(x)/q(x) is known and log r is quadratic in x. The estimators: dividing two kernel density estimates,
 * logistic regression between the samples, KLIEP (projected gradient ascent on the log-likelihood of the p sample
 * over Gaussian-kernel models) and uLSIF (a ridge-regularised linear solve).
 *
 * A candidate for the engine: `aifn-methods/unsupervised/density-ratio` with `kliep(xp, xq, options)` (an Algorithm
 * to trace), `ulsif(xp, xq, options)` with its cross-validation score, and `classifierRatio(xp, xq, options)`, each
 * returning a ratio function.
 */
import { child, normals, stream } from 'aifn-compute/foundation/random'
import {
  add,
  div,
  eye,
  log,
  matmul,
  mean,
  mul,
  neg,
  sum,
  tensor,
  toFlat,
  transpose,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { binaryCrossEntropyWithLogits } from 'aifn-compute/learning/losses'
import { solve } from 'aifn-compute/numerics/linalg'
import { normalLogPdf } from 'aifn-compute/numerics/special'
import { gradientDescent, lbfgs, objectiveFn } from 'aifn-compute/optim'
import { projectedGradient, projectSimplex } from 'aifn-compute/optim/proximal'
import { multivariateKde } from 'aifn-compute/probability/stats'

/** The numerator p = N(mean, sd²); the denominator q is always N(0, 1). */
export type Shift = { mean: number; sd: number }

/** log p(x). */
export const logP = (x: number, s: Shift) => normalLogPdf((x - s.mean) / s.sd) - Math.log(s.sd)
/** log q(x). */
export const logQ = (x: number) => normalLogPdf(x)
/** The true log ratio log r(x) = log p(x) − log q(x). */
export const logRatio = (x: number, s: Shift) => logP(x, s) - logQ(x)

/** A sample from p and a sample from q. A larger n extends a smaller one, so growing samples are nested. */
export function drawSamples(seed: number, np: number, nq: number, s: Shift) {
  const root = stream(seed)
  const xp = toFlat(normals(child(root, 1), [np])).map((z) => s.mean + s.sd * z)
  const xq = toFlat(normals(child(root, 2), [nq]))
  return { xp, xq }
}

/** The Gaussian-kernel design matrix K[i, l] = exp(−(x_i − c_l)²/2σ²), of shape [n, b]. */
export const design = (x: readonly number[], centres: readonly number[], sigma: number): Tensor =>
  gram(rbf({ lengthscale: sigma }), tensor([...x]), tensor([...centres]))

/** At most b members of the p sample, spread through it: the usual kernel centres. */
export function pickCentres(xp: readonly number[], b: number): number[] {
  if (xp.length <= b) return [...xp]
  const step = xp.length / b
  return Array.from({ length: b }, (_, l) => xp[Math.floor(l * step)])
}

/** r̂(x) = Σ α_l K(x, c_l) at the points xs, floored at 0. */
export function kernelRatio(
  alpha: readonly number[],
  centres: readonly number[],
  sigma: number,
  xs: readonly number[],
) {
  return toFlat(matmul(design(xs, centres, sigma), tensor([...alpha]))).map((v) => Math.max(v, 0))
}

/**
 * The log of a Gaussian kernel density estimate of `sample` at xs, with bandwidth `scale` times Scott's rule (the
 * engine's factor is relative to the sample's standard deviation; Scott's is n^(−1/5) in one dimension).
 */
export const kdeLogDensity = (sample: readonly number[], xs: readonly number[], scale = 1) =>
  toFlat(
    multivariateKde(tensor(sample.map((v) => [v])), tensor(xs.map((x) => [x])), {
      bandwidth: scale * sample.length ** -0.2,
    }).logDensity,
  )

/** log p̂(x) − log q̂(x) from two kernel density estimates, the plug-in estimate of log r. */
export function kdeLogRatio(xp: readonly number[], xq: readonly number[], xs: readonly number[], scale = 1) {
  const top = kdeLogDensity(xp, xs, scale)
  const bottom = kdeLogDensity(xq, xs, scale)
  return top.map((t, i) => t - bottom[i])
}

/**
 * KLIEP (Sugiyama et al., 2008): maximise the mean of log r̂ over the p sample subject to the mean of r̂ over the q
 * sample being 1, with r̂ = Σ α_l K(·, c_l) and α ≥ 0. With β_l = α_l b_l, where b_l is the q sample's mean of
 * K(·, c_l), the constraints become the probability simplex, so projected gradient with the simplex projection solves
 * it. Returns α and the objective at every step; `accelerated` uses the FISTA extrapolation.
 */
export function kliepTrace(
  xp: readonly number[],
  xq: readonly number[],
  centres: readonly number[],
  sigma: number,
  steps: number,
  accelerated = false,
) {
  const scale = toFlat(mean(design(xq, centres, sigma), 0) as Tensor)
  const A = div(design(xp, centres, sigma), tensor(scale))
  const At = transpose(A)
  const n = xp.length
  // f(β) = −mean log(Aβ), with gradient −Aᵀ(1/Aβ)/n.
  const f = (beta: Tensor) => {
    const z = matmul(A, beta)
    return { value: -(mean(log(z)) as number), grad: neg(div(matmul(At, div(1, z)), n)) as Tensor }
  }
  const b = centres.length
  const alg = projectedGradient(f, projectSimplex(1), { stepSize: 1, backtracking: true, accelerated })
  const tr = trace(alg, { x0: new Array(b).fill(1 / b) }, steps)
  return {
    alphas: tr.steps.map((s) => toFlat(s.x).map((v, l) => v / scale[l])),
    objective: tr.steps.map((s) => -s.value),
    index: Array.from(tr.index),
  }
}

/** uLSIF (Kanamori et al., 2009): α = max(0, (H + λI)⁻¹h), with H = K_qᵀK_q/n_q and h the mean row of K_p. */
export function ulsif(
  xp: readonly number[],
  xq: readonly number[],
  centres: readonly number[],
  sigma: number,
  lambda: number,
) {
  return ulsifFromDesigns(design(xp, centres, sigma), design(xq, centres, sigma), lambda)
}

function ulsifFromDesigns(Kp: Tensor, Kq: Tensor, lambda: number) {
  const b = Kp.shape[1]
  const H = add(div(matmul(transpose(Kq), Kq), Kq.shape[0]), mul(lambda, eye(b)))
  return toFlat(solve(H, mean(Kp, 0)) as Tensor).map((a) => Math.max(a, 0))
}

/**
 * The cross-validated uLSIF score J = ½ mean_q r̂² − mean_p r̂ on held-out folds, for every (σ, λ) pair; rows follow
 * `sigmas`, columns `lambdas`. J equals ½ E_q[(r̂ − r)²] minus a constant that does not depend on r̂, so lower is
 * better.
 */
export function ulsifScores(
  xp: readonly number[],
  xq: readonly number[],
  centres: readonly number[],
  sigmas: readonly number[],
  lambdas: readonly number[],
  folds = 5,
) {
  const part = (x: readonly number[], k: number) => ({
    fit: x.filter((_, i) => i % folds !== k),
    held: x.filter((_, i) => i % folds === k),
  })
  return sigmas.map((sigma) => {
    const totals = lambdas.map(() => 0)
    for (let k = 0; k < folds; k++) {
      const p = part(xp, k)
      const q = part(xq, k)
      const Kp = design(p.fit, centres, sigma)
      const Kq = design(q.fit, centres, sigma)
      const Hp = design(p.held, centres, sigma)
      const Hq = design(q.held, centres, sigma)
      lambdas.forEach((lambda, j) => {
        const alpha = tensor(ulsifFromDesigns(Kp, Kq, lambda))
        const rq = toFlat(matmul(Hq, alpha))
        const rp = toFlat(matmul(Hp, alpha))
        totals[j] += 0.5 * average(rq.map((v) => v * v)) - average(rp)
      })
    }
    return totals.map((t) => t / folds)
  })
}

const average = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / Math.max(v.length, 1)

/** Feature maps for the classifier: a line, a quadratic (exact for two Gaussians), or Gaussian bumps. */
export type Features = 'linear' | 'quadratic' | 'kernel'

/**
 * The features φ(x) of each point, intercept first. In the quadratic map x² enters as (x² − 1)/√2, which is
 * uncorrelated with 1 and x under q, so gradient descent is not slowed by badly scaled features.
 */
export function featureRows(xs: readonly number[], kind: Features, centres: readonly number[], sigma: number) {
  if (kind === 'kernel') {
    const K = toFlat(design(xs, centres, sigma))
    const b = centres.length
    return xs.map((_, i) => [1, ...K.slice(i * b, (i + 1) * b)])
  }
  return xs.map((x) => (kind === 'linear' ? [1, x] : [1, x, (x * x - 1) / Math.SQRT2]))
}

/** The penalised mean log loss of "from p" (y = 1) against "from q" (y = 0), differentiated by the engine. */
function logisticObjective(pRows: number[][], qRows: number[][], lambda: number) {
  const X = tensor([...pRows, ...qRows])
  const y = tensor([...pRows.map(() => 1), ...qRows.map(() => 0)])
  return objectiveFn({
    kind: 'objective',
    name: 'logistic',
    dim: pRows[0].length,
    value: (w: Tensor) =>
      add(binaryCrossEntropyWithLogits(matmul(X, w), y), mul(0.5 * lambda, sum(mul(w, w)))) as Tensor,
  })
}

/**
 * Logistic regression of "from p" (y = 1) against "from q" (y = 0), with the penalty λ/2 ‖w‖², by gradient descent
 * with a backtracking line search from w = 0. Returns w and the penalised mean log loss at every step.
 */
export function classifierTrace(pRows: number[][], qRows: number[][], lambda: number, steps: number) {
  const f = logisticObjective(pRows, qRows, lambda)
  const d = pRows[0].length
  const tr = trace(gradientDescent(f, { lineSearch: 'backtracking' }), { x0: new Array(d).fill(0) }, steps)
  return {
    weights: tr.steps.map((s) => toFlat(s.x)),
    loss: tr.steps.map((s) => s.value),
    index: Array.from(tr.index),
  }
}

/** The same logistic regression run to convergence by L-BFGS, for when only the fitted weights are needed. */
export function classifierFit(pRows: number[][], qRows: number[][], lambda: number) {
  const f = logisticObjective(pRows, qRows, lambda)
  const d = pRows[0].length
  return toFlat(run(lbfgs(f), { x0: new Array(d).fill(0) }, 200).x)
}

/** log r̂(x) = log(n_q/n_p) + φ(x)ᵀw, the classifier's log odds corrected for the sample sizes. */
export const classifierLogRatio = (rows: number[][], w: readonly number[], np: number, nq: number) =>
  rows.map((phi) => Math.log(nq / np) + phi.reduce((a, v, i) => a + v * w[i], 0))

/**
 * Two errors of an estimate r̂ (given as log r̂ on an even grid xs), by a Riemann sum: the squared error
 * ½ ∫ (r̂ − r)² q dx, which uLSIF minimises, and the generalised Kullback–Leibler divergence
 * ∫ p log(r/r̂) dx + ∫ q r̂ dx − 1 from p to r̂q, which KLIEP minimises.
 */
export function ratioErrors(xs: readonly number[], logEst: readonly number[], s: Shift) {
  const dx = xs[1] - xs[0]
  let squared = 0
  let kl = 0
  xs.forEach((x, i) => {
    const q = Math.exp(logQ(x))
    const p = Math.exp(logP(x, s))
    const est = Math.exp(logEst[i])
    squared += 0.5 * (est - Math.exp(logRatio(x, s))) ** 2 * q * dx
    kl += (p * (logRatio(x, s) - Math.max(logEst[i], -700)) + q * est - p) * dx
  })
  return { squared, kl }
}
