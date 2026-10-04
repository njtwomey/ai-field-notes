import type { Vec2 as Point } from 'aifn/numerics/linalg'
import { initialCentres, type CentreInit } from '../_shared/centres'
import { logsumexp, fromData, toFlat } from 'aifn/foundation/tensor'
import { softmax } from 'aifn/numerics/special'

/** Smallest allowed variance. Without it a component can collapse onto one point and the likelihood diverges. */
const VARIANCE_FLOOR = 1e-3

export type Mixture = { weights: number[]; means: Point[]; variances: Point[] }

export type EmState = {
  /** The mixture after each iteration; entry 0 is the initialisation. */
  mixtures: Mixture[]
  /** Mean log-likelihood per point, one entry per mixture. */
  history: number[]
  /** The iteration on display. Stepping from an earlier cursor discards the later iterations. */
  cursor: number
  /** responsibilities[i][j] = P(component j | point i), under the mixture at the cursor. */
  responsibilities: number[][]
  /** The last iteration met the convergence tolerance. */
  done: boolean
}

function logJoint(p: Point, m: Mixture, j: number): number {
  const [vx, vy] = m.variances[j]
  const dx = p[0] - m.means[j][0]
  const dy = p[1] - m.means[j][1]
  return (
    Math.log(m.weights[j]) -
    0.5 * (Math.log(2 * Math.PI * vx) + Math.log(2 * Math.PI * vy) + (dx * dx) / vx + (dy * dy) / vy)
  )
}

/** Responsibilities and mean log-likelihood under `m`, delegated to aifn tensor and special functions. */
export function eStep(points: Point[], m: Mixture): { responsibilities: number[][]; logLikelihood: number } {
  const n = points.length
  const k = m.weights.length
  const flatJoint = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      flatJoint[i * k + j] = logJoint(points[i], m, j)
    }
  }
  const joint = fromData(flatJoint, [n, k])
  const respTensor = softmax(joint)
  const respFlat = toFlat(respTensor)
  const responsibilities: number[][] = []
  for (let i = 0; i < n; i++) {
    responsibilities.push(respFlat.slice(i * k, (i + 1) * k))
  }
  const logMarginals = toFlat(logsumexp(joint, 1))
  let total = 0
  for (let i = 0; i < n; i++) total += logMarginals[i]
  return { responsibilities, logLikelihood: total / n }
}

/** Weighted maximum-likelihood estimates, each point weighted by its responsibility. */
function mStep(points: Point[], r: number[][]): Mixture {
  const k = r[0].length
  const counts = Array.from({ length: k }, (_, j) => r.reduce((s, ri) => s + ri[j], 0))
  const means = counts.map((c, j): Point => [
    points.reduce((s, p, i) => s + r[i][j] * p[0], 0) / c,
    points.reduce((s, p, i) => s + r[i][j] * p[1], 0) / c,
  ])
  const variances = counts.map((c, j): Point => [
    Math.max(points.reduce((s, p, i) => s + r[i][j] * (p[0] - means[j][0]) ** 2, 0) / c, VARIANCE_FLOOR),
    Math.max(points.reduce((s, p, i) => s + r[i][j] * (p[1] - means[j][1]) ** 2, 0) / c, VARIANCE_FLOOR),
  ])
  return { weights: counts.map((c) => c / points.length), means, variances }
}

/** Means from k-means seeding; every component starts with the overall variance and equal weight. */
export function initialise(points: Point[], k: number, init: CentreInit, seed: number): EmState {
  const mean = (d: 0 | 1) => points.reduce((s, p) => s + p[d], 0) / points.length
  const variance = (d: 0 | 1) => points.reduce((s, p) => s + (p[d] - mean(d)) ** 2, 0) / points.length
  const overall: Point = [variance(0), variance(1)]
  const mixture: Mixture = {
    weights: Array(k).fill(1 / k),
    means: initialCentres(points, k, init, seed),
    variances: Array.from({ length: k }, () => [...overall] as Point),
  }
  const { responsibilities, logLikelihood } = eStep(points, mixture)
  return { mixtures: [mixture], history: [logLikelihood], cursor: 0, responsibilities, done: false }
}

/**
 * One EM iteration from the cursor: M-step from the current responsibilities, then an E-step under the new parameters.
 * Any iterations after the cursor are discarded.
 */
export function step(points: Point[], s: EmState, tol = 1e-6): EmState {
  const mixtures = s.mixtures.slice(0, s.cursor + 1)
  const history = s.history.slice(0, s.cursor + 1)
  const mixture = mStep(points, s.responsibilities)
  const { responsibilities, logLikelihood } = eStep(points, mixture)
  const done = logLikelihood - history[history.length - 1] < tol
  return {
    mixtures: [...mixtures, mixture],
    history: [...history, logLikelihood],
    cursor: mixtures.length,
    responsibilities,
    done,
  }
}

/** Show iteration `i` without discarding anything. */
export function seek(points: Point[], s: EmState, i: number): EmState {
  const cursor = Math.max(0, Math.min(s.mixtures.length - 1, Math.round(i)))
  return { ...s, cursor, responsibilities: eStep(points, s.mixtures[cursor]).responsibilities }
}

/**
 * Move component j's mean in the mixture on display and start a new run from the edited mixture. Earlier iterations
 * are discarded: the edit can lower the likelihood, and the history must stay one monotone EM run.
 */
export function moveMean(points: Point[], s: EmState, j: number, mean: Point): EmState {
  const current = s.mixtures[s.cursor]
  const mixture = { ...current, means: current.means.map((m, i) => (i === j ? mean : m)) }
  const { responsibilities, logLikelihood } = eStep(points, mixture)
  return { mixtures: [mixture], history: [logLikelihood], cursor: 0, responsibilities, done: false }
}

/** True when the cursor is at the last iteration and that iteration converged. */
export const finished = (s: EmState) => s.done && s.cursor === s.mixtures.length - 1

/** Points on the axis-aligned ellipse at `radius` standard deviations around component j. */
export function ellipse(m: Mixture, j: number, radius: number, segments = 64): { x: number[]; y: number[] } {
  const [sx, sy] = m.variances[j].map(Math.sqrt)
  const t = Array.from({ length: segments + 1 }, (_, i) => (2 * Math.PI * i) / segments)
  return {
    x: t.map((a) => m.means[j][0] + radius * sx * Math.cos(a)),
    y: t.map((a) => m.means[j][1] + radius * sy * Math.sin(a)),
  }
}
