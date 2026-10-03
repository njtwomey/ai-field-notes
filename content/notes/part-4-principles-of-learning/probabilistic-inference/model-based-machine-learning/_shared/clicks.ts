/**
 * The Infer.NET click model: a latent relevance score per query–document pair, seen by human judges through noise and
 * ordered thresholds, and by users through a click rate. Parameters are the values Infer.NET learns on its example data.
 */
import { Phi, gaussPdf } from './gaussian.ts'

export const CLICK_MODEL = {
  scoreMean: 0.4611,
  scoreVar: 0.5,
  judgeVar: 1 / 19.7,
  clickVar: 1 / 19.98,
  /** Inner thresholds between labels 0|1 and 1|2. */
  thresholds: [0.2013, 0.8018],
}

/** Moment-matched Gaussian of the Beta(1 + clicks, 1 + non-clicks) posterior over the click rate. */
export function clickObservation(clicks: number, exams: number): { mean: number; variance: number } {
  const a = 1 + clicks
  const b = 1 + exams - clicks
  return { mean: a / (a + b), variance: (a * b) / ((a + b) ** 2 * (a + b + 1)) }
}

/** Posterior over the score from clicks alone: a product of two Gaussians. */
export function scoreFromClicks(clicks: number, exams: number): { mean: number; variance: number } {
  const m = CLICK_MODEL
  const obs = clickObservation(clicks, exams)
  const v = obs.variance + m.clickVar
  const prec = 1 / m.scoreVar + 1 / v
  return { variance: 1 / prec, mean: (m.scoreMean / m.scoreVar + obs.mean / v) / prec }
}

/** P(label = l | score ~ N(mean, variance)); the outer thresholds are ±∞, as in Infer.NET's prediction model. */
export function labelProbabilities(mean: number, variance: number): number[] {
  const sd = Math.sqrt(variance + CLICK_MODEL.judgeVar)
  const cuts = [-Infinity, ...CLICK_MODEL.thresholds, Infinity]
  return [0, 1, 2].map((l) => Phi((cuts[l + 1] - mean) / sd) - Phi((cuts[l] - mean) / sd))
}

/**
 * Likelihood of a judge's label given the score, with the training model's outer thresholds 0 and 1: the judged score
 * N(score, judgeVar) must fall in the label's interval.
 */
export function labelLikelihood(label: number, score: number): number {
  const sd = Math.sqrt(CLICK_MODEL.judgeVar)
  const cuts = [0, ...CLICK_MODEL.thresholds, 1]
  return Phi((cuts[label + 1] - score) / sd) - Phi((cuts[label] - score) / sd)
}

/** Densities on a grid: prior, clicks-only posterior, and the posterior with a judge's label as well. */
export function scoreDensities(grid: number[], clicks: number, exams: number, label: number | null) {
  const m = CLICK_MODEL
  const c = scoreFromClicks(clicks, exams)
  const prior = grid.map((s) => gaussPdf(s, m.scoreMean, m.scoreVar))
  const fromClicks = grid.map((s) => gaussPdf(s, c.mean, c.variance))
  const h = grid[1] - grid[0]
  let both = fromClicks.map((p, i) => (label === null ? p : p * labelLikelihood(label, grid[i])))
  const z = both.reduce((a, b) => a + b, 0) * h
  both = both.map((p) => p / z)
  const mean = both.reduce((a, p, i) => a + p * grid[i], 0) * h
  const sd = Math.sqrt(both.reduce((a, p, i) => a + p * (grid[i] - mean) ** 2, 0) * h)
  return { prior, fromClicks, both, clicks: c, mean, sd }
}
