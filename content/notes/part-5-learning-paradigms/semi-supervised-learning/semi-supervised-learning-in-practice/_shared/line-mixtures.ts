/**
 * Figure helpers for two-class problems on the real line whose class-conditional densities are Gaussian mixtures:
 * sampling, the density, and the exact error of a two-Gaussian decision rule. Used to show semi-supervised EM with a
 * correct and a misspecified model (when-semi-supervised-learning-fails).
 */
import { child, normals, uniform, type Stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'

/** One mixture component: weight, mean, standard deviation. */
export type Component = readonly [weight: number, mean: number, sd: number]

/** A two-class problem: equal priors, each class a Gaussian mixture. */
export interface LineMixtureProblem {
  readonly classes: readonly [readonly Component[], readonly Component[]]
}

/** The density of a mixture at x. */
export function mixtureDensity(cs: readonly Component[], x: number): number {
  let p = 0
  for (const [w, m, s] of cs) p += (w * Math.exp(-0.5 * ((x - m) / s) ** 2)) / (s * Math.sqrt(2 * Math.PI))
  return p
}

/** The mass of a mixture on the interval (a, b). */
function mixtureMass(cs: readonly Component[], a: number, b: number): number {
  let p = 0
  for (const [w, m, s] of cs) p += w * (normalCdf((b - m) / s) - normalCdf((a - m) / s))
  return p
}

/** n draws from a mixture. */
export function drawMixture(cs: readonly Component[], s: Stream, n: number): number[] {
  if (n === 0) return []
  const u = toFlat(uniform(child(s, 'component'), 0, 1, { shape: [n] }))
  const z = toFlat(normals(child(s, 'value'), n))
  return u.map((v, i) => {
    let acc = 0
    for (const [w, m, sd] of cs) {
      acc += w
      if (v <= acc) return m + sd * z[i]
    }
    const [, m, sd] = cs[cs.length - 1]
    return m + sd * z[i]
  })
}

/** A sample: m labelled points per class and u unlabelled points with equal class priors. */
export function drawLineSample(
  problem: LineMixtureProblem,
  s: Stream,
  perClass: number,
  u: number,
): { x: number[]; labels: number[]; truth: number[] } {
  const l0 = drawMixture(problem.classes[0], child(s, 'labelled', 0), perClass)
  const l1 = drawMixture(problem.classes[1], child(s, 'labelled', 1), perClass)
  const c = toFlat(uniform(child(s, 'class'), 0, 1, { shape: [Math.max(u, 1)] }))
    .slice(0, u)
    .map((v) => (v < 0.5 ? 0 : 1))
  const u0 = drawMixture(problem.classes[0], child(s, 'unlabelled', 0), u)
  const u1 = drawMixture(problem.classes[1], child(s, 'unlabelled', 1), u)
  const xu = c.map((k, i) => (k === 0 ? u0[i] : u1[i]))
  return {
    x: [...l0, ...l1, ...xu],
    labels: [...l0.map(() => 0), ...l1.map(() => 1), ...xu.map(() => -1)],
    truth: [...l0.map(() => 0), ...l1.map(() => 1), ...c],
  }
}

/**
 * A one-dimensional two-class Gaussian rule in the flat layout of `ClassGaussians` with $d = 1$: the priors
 * `weights[k]`, the means `means[k]` and the variances `covariances[k]` of classes 0 and 1. A state of
 * `semiSupervisedMixtureSteps` on the line is one.
 */
export interface TwoGaussianRule {
  weights: ArrayLike<number>
  means: ArrayLike<number>
  covariances: ArrayLike<number>
}

/**
 * The points where a two-Gaussian rule switches class: the real roots of
 * $\log \pi_1\Gauss(x \mid \mu_1, \sigma_1^2) = \log \pi_0\Gauss(x \mid \mu_0, \sigma_0^2)$, sorted.
 */
export function ruleBoundaries(rule: TwoGaussianRule): number[] {
  const p0 = rule.weights[0]
  const p1 = rule.weights[1]
  const m0 = rule.means[0]
  const m1 = rule.means[1]
  const v0 = rule.covariances[0]
  const v1 = rule.covariances[1]
  // f(x) = log p1 - log p0 = a x^2 + b x + c
  const a = 0.5 / v0 - 0.5 / v1
  const b = m1 / v1 - m0 / v0
  const c = Math.log(p1 / p0) - 0.5 * Math.log(v1 / v0) - (m1 * m1) / (2 * v1) + (m0 * m0) / (2 * v0)
  if (Math.abs(a) < 1e-12) return Math.abs(b) < 1e-12 ? [] : [-c / b]
  const disc = b * b - 4 * a * c
  if (disc < 0) return []
  const r = Math.sqrt(disc)
  return [(-b - r) / (2 * a), (-b + r) / (2 * a)].sort((u, v) => u - v)
}

/** The class a two-Gaussian rule gives at x. */
export function ruleClass(rule: TwoGaussianRule, x: number): number {
  const l = [0, 1].map(
    (k) =>
      Math.log(rule.weights[k]) -
      0.5 * Math.log(rule.covariances[k]) -
      (x - rule.means[k]) ** 2 / (2 * rule.covariances[k]),
  )
  return l[1] > l[0] ? 1 : 0
}

/** The exact error of a two-Gaussian rule on the problem (equal true priors). */
export function ruleError(problem: LineMixtureProblem, rule: TwoGaussianRule): number {
  const cuts = [-Infinity, ...ruleBoundaries(rule), Infinity]
  let e = 0
  for (let i = 0; i + 1 < cuts.length; i++) {
    const a = cuts[i]
    const b = cuts[i + 1]
    const mid =
      Number.isFinite(a) && Number.isFinite(b)
        ? (a + b) / 2
        : Number.isFinite(a)
          ? a + 1
          : Number.isFinite(b)
            ? b - 1
            : 0
    const k = ruleClass(rule, mid)
    e += 0.5 * mixtureMass(problem.classes[1 - k], a, b)
  }
  return e
}

/** The Bayes error of the problem, by a fine grid (the densities are smooth). */
export function bayesError(problem: LineMixtureProblem, range: readonly [number, number] = [-12, 12]): number {
  const n = 6000
  const dx = (range[1] - range[0]) / n
  let e = 0
  for (let i = 0; i < n; i++) {
    const x = range[0] + (i + 0.5) * dx
    e += 0.5 * Math.min(mixtureDensity(problem.classes[0], x), mixtureDensity(problem.classes[1], x)) * dx
  }
  return e
}
