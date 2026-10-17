/**
 * The study behind the EvaluationProtocol figure: Laplacian regularised least squares (a semi-supervised method)
 * against kernel ridge regression on the labelled points alone (its supervised baseline) on two moons, for several
 * label counts and repeats, scored on a test set and on two validation sets, for every hyperparameter setting. The
 * figure then selects settings by validation accuracy, for the semi-supervised method over the whole grid and for the
 * baseline over a random subset of the grid of a given size.
 */
import { moons } from 'aifn-methods/data'
import { pointAffinity } from 'aifn-compute/graph/propagation'
import { gram, rbf } from 'aifn-compute/learning/kernels'
import { child, stream } from 'aifn-compute/foundation/random'
import { matmul, tensor, toArray, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { laplacianRegularisedLeastSquares } from 'aifn-methods/learning/semi-supervised'
import { laplacian } from 'aifn-compute/graph/matrices'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'

export const POINTS = 100
export const NOISE = 0.1
export const TEST_POINTS = 1000
export const LARGE_VALIDATION = 500
export const GAMMA_I = 1e4
export const REPEATS = 5
export const LABEL_COUNTS = [1, 2, 4, 8, 16]
/** The hyperparameter grid: kernel lengthscale ℓ and ambient weight γ_A. */
export const CONFIGS: readonly (readonly [lengthscale: number, gammaA: number])[] = [
  [1.5, 0.1],
  [0.5, 0.1],
  [0.2, 0.1],
  [1.5, 1e-3],
  [0.5, 1e-3],
  [0.2, 1e-3],
]

/** Accuracies of one fit: on the test set, the large validation set and the small one. */
export interface Scores {
  test: number
  large: number
  small: number
}

/** scores[method][label count][repeat][config]. */
export type Study = Record<'supervised' | 'semiSupervised', Scores[][][]>

const signAccuracy = (f: ArrayLike<number>, y: ArrayLike<number>) => {
  let right = 0
  for (let i = 0; i < y.length; i++) if ((f[i] > 0 ? 1 : 0) === y[i]) right++
  return right / y.length
}

/** Run the whole study (about half a second). */
export function runStudy(seed: number): Study {
  const root = stream(seed)
  const test = moons(child(root, 'test'), { n: TEST_POINTS, noise: NOISE, spacing: 'random' })
  const large = moons(child(root, 'validation'), { n: LARGE_VALIDATION, noise: NOISE, spacing: 'random' })
  const yTest = toFlat(test.y)
  const yLarge = toFlat(large.y)
  const study: Study = { supervised: LABEL_COUNTS.map(() => []), semiSupervised: LABEL_COUNTS.map(() => []) }
  for (let r = 0; r < REPEATS; r++) {
    const s = child(root, 'repeat', r)
    const d = moons(child(s, 'data'), { n: POINTS, noise: NOISE, spacing: 'random' })
    const y = toFlat(d.y)
    const targets = y.map((v) => (v === 1 ? 1 : -1))
    const lap = laplacian(pointAffinity(d.x, { gamma: 1 / (2 * 0.2 ** 2), neighbours: 8 }))
    const kernels = new Map<number, { K: Tensor; Kt: Tensor; Kv: Tensor; rows: number[][] }>()
    for (const [ls] of CONFIGS)
      if (!kernels.has(ls)) {
        const k = rbf({ lengthscale: ls })
        const K = gram(k, d.x)
        kernels.set(ls, { K, Kt: gram(k, test.x, d.x), Kv: gram(k, large.x, d.x), rows: toArray(K) as number[][] })
      }
    LABEL_COUNTS.forEach((m, mi) => {
      const split = semiSupervisedSplit(child(s, 'split', m), y, m)
      const small = moons(child(s, 'small-validation', m), { n: 2 * m, noise: NOISE, spacing: 'random' })
      const ySmall = toFlat(small.y)
      const L = split.labelled
      const sup: Scores[] = []
      const ssl: Scores[] = []
      for (const [ls, gammaA] of CONFIGS) {
        const g = kernels.get(ls)!
        const Ks = gram(rbf({ lengthscale: ls }), small.x, d.x)
        const score = (alpha: Tensor): Scores => ({
          test: signAccuracy(toFlat(matmul(g.Kt, alpha)), yTest),
          large: signAccuracy(toFlat(matmul(g.Kv, alpha)), yLarge),
          small: signAccuracy(toFlat(matmul(Ks, alpha)), ySmall),
        })
        // Kernel ridge regression on the labelled points: LapRLS on those points alone with no graph term.
        const KLL = L.map((i) => L.map((j) => g.rows[i][j]))
        const zero = L.map(() => L.map(() => 0))
        const krr = laplacianRegularisedLeastSquares(
          KLL,
          zero,
          L.map((i) => targets[i]),
          L.map((_, q) => q),
          { gammaA, gammaI: 0 },
        )
        const a = toFlat(krr.alpha)
        const full = new Array<number>(POINTS).fill(0)
        L.forEach((i, q) => (full[i] = a[q]))
        sup.push(score(tensor(full)))
        const lapRls = laplacianRegularisedLeastSquares(g.K, lap, targets, L, { gammaA, gammaI: GAMMA_I })
        ssl.push(score(lapRls.alpha))
      }
      study.supervised[mi].push(sup)
      study.semiSupervised[mi].push(ssl)
    })
  }
  return study
}

/** All subsets of {0, …, n − 1} of size b. */
function subsets(n: number, b: number): number[][] {
  const out: number[][] = []
  const pick = (from: number, cur: number[]) => {
    if (cur.length === b) {
      out.push(cur.slice())
      return
    }
    for (let i = from; i < n; i++) {
      cur.push(i)
      pick(i + 1, cur)
      cur.pop()
    }
  }
  pick(0, [])
  return out
}

/** The test accuracy of the setting a validation set picks among `allowed` (ties to the first). */
function selected(rows: Scores[], allowed: readonly number[], by: 'large' | 'small'): number {
  let best = allowed[0]
  for (const c of allowed) if (rows[c][by] > rows[best][by]) best = c
  return rows[best].test
}

/**
 * Mean test accuracy per label count after selection: the semi-supervised method over the whole grid, and the
 * baseline over a random subset of `budget` settings (averaged over every subset of that size), each picked by the
 * chosen validation set.
 */
export function summarise(study: Study, budget: number, by: 'large' | 'small') {
  const all = CONFIGS.map((_, c) => c)
  const tries = subsets(CONFIGS.length, budget)
  const mean = (xs: number[]) => xs.reduce((a, v) => a + v, 0) / xs.length
  return LABEL_COUNTS.map((_, mi) => ({
    semiSupervised: mean(study.semiSupervised[mi].map((rows) => selected(rows, all, by))),
    baseline: mean(study.supervised[mi].map((rows) => mean(tries.map((t) => selected(rows, t, by))))),
    tunedBaseline: mean(study.supervised[mi].map((rows) => selected(rows, all, by))),
  }))
}
