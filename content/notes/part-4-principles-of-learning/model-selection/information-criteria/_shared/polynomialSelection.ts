/**
 * Shared simulation for the information-criteria figures: polynomials of degree 0 to MAX_DEGREE fitted by maximum
 * likelihood (Gaussian noise of unknown variance) to samples from a known regression function, the degree each
 * criterion chooses, and the excess error of every fit against the true function. Everything composes engine calls:
 * the data come from `drawTrainingSet`, the fits are `glm`, the criteria are `informationCriteria`, and the errors are
 * `meanSquaredError` on a fixed grid.
 */
import { drawTrainingSet, targetFunction, unitGrid, type TargetName } from 'aifn-methods/theory'
import { glm } from 'aifn-methods/learning/generalised/glm'
import { polynomialFeatures } from 'aifn-methods/learning/preprocessing'
import { gaussianFamily } from 'aifn-compute/probability/likelihoods'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { informationCriteria } from 'aifn-compute/learning/selection'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

export const MAX_DEGREE = 8
export const DEGREES = Array.from({ length: MAX_DEGREE + 1 }, (_, d) => d)
export const N_GRID = [30, 60, 125, 250, 500, 1000, 2000]
export const CRITERIA = ['aic', 'aicc', 'bic'] as const
export type Criterion = (typeof CRITERIA)[number]

const GRID = unitGrid(201)
const column = (x: Float64Array) => fromData(x, [x.length, 1])
const argmin = (v: ArrayLike<number>) => {
  let best = 0
  for (let i = 1; i < v.length; i++) if (v[i] < v[best]) best = i
  return best
}

/** A polynomial of degree d with Gaussian noise of unknown variance, fitted by maximum likelihood. */
export function fitPolynomial(degree: number, x: Float64Array, y: Float64Array) {
  const features = polynomialFeatures({ degree, includeBias: true }).fit({ x: column(x) })
  const model = glm({ family: gaussianFamily(), intercept: false }).fit({
    x: features.transform(column(x)),
    y: fromData(y),
  })
  const predict = (at: Float64Array): Tensor => model.decide(features.transform(column(at)))
  return { model, predict }
}

/** One replicate: the degree each criterion chooses, and the excess error of every degree's fit on the grid. */
export function selectOnce(seed: string, target: TargetName, n: number, noise: number) {
  const train = drawTrainingSet(child(stream(seed), 'train'), target, n, noise)
  const truth = Array.from(GRID, targetFunction(target))
  const fits = DEGREES.map((d) => fitPolynomial(d, train.x, train.y))
  const criteria = informationCriteria({
    logLik: fits.map((f) => f.model.logLikelihood),
    k: DEGREES.map((d) => d + 2),
    n,
  })
  const excess = fits.map((f) => meanSquaredError(truth, f.predict(GRID)))
  const chosen = Object.fromEntries(CRITERIA.map((c) => [c, argmin(toFlat(criteria[c]))])) as Record<Criterion, number>
  return { chosen, excess, oracle: Math.min(...excess), best: argmin(excess) }
}

export type SelectionSummary = {
  n: number
  /** Fraction of replicates in which each criterion chose each degree. */
  frequency: Record<Criterion, number[]>
  /** Mean excess error of the chosen fit divided by the mean excess error of the best fit on each sample. */
  lossRatio: Record<Criterion, number>
  /** Mean degree each criterion chose. */
  meanDegree: Record<Criterion, number>
  /** Mean degree of the best fit on each sample, the oracle's choice. */
  bestDegree: number
}

/** The selection study over N_GRID: `replicates` samples at each n. */
export function selectionStudy(name: string, target: TargetName, noise: number, replicates: number) {
  return N_GRID.map((n): SelectionSummary => {
    const runs = Array.from({ length: replicates }, (_, r) =>
      selectOnce(`${name}/${target}/${noise}/${n}/${r}`, target, n, noise),
    )
    const oracle = runs.reduce((a, run) => a + run.oracle, 0)
    const frequency = {} as Record<Criterion, number[]>
    const lossRatio = {} as Record<Criterion, number>
    const meanDegree = {} as Record<Criterion, number>
    for (const c of CRITERIA) {
      frequency[c] = DEGREES.map((d) => runs.filter((run) => run.chosen[c] === d).length / replicates)
      lossRatio[c] = runs.reduce((a, run) => a + run.excess[run.chosen[c]], 0) / oracle
      meanDegree[c] = runs.reduce((a, run) => a + run.chosen[c], 0) / replicates
    }
    const bestDegree = runs.reduce((a, run) => a + run.best, 0) / replicates
    return { n, frequency, lossRatio, meanDegree, bestDegree }
  })
}
