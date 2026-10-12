/**
 * The diabetes data of Efron et al. (2004) as the linear-model notes use it: features standardised with
 * `standardScaler`, target centred. Each `*Data` function computes one figure's paths on the engine and caches it.
 */
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import {
  elasticNet,
  elasticNetPath,
  leastAngleRegression,
  linearRegression,
  linearSgdSteps,
  ridgePath,
} from 'aifn-methods/learning/linear'
import { child, stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dataset } from 'aifn-compute/learning/estimators'

/** Coefficients of a regularised fit along a grid of penalty values. */
export interface CoefficientPath {
  features: string[]
  /** Penalty values, decreasing: from all-zero (or near) to least penalised. */
  penalty: number[]
  /** Indexed [feature][penalty index]. */
  coef: number[][]
}

/** A piecewise-linear path given by its knots; coefficients are linear in ‖w‖₁ between consecutive knots. */
export interface KnotPath {
  l1Norm: number[]
  alphas: number[]
  /** Indexed [feature][knot]. */
  coef: number[][]
  /** Features entering or leaving the active set, in order, with the penalty at that knot. */
  events: { knot: number; alpha: number; feature: number; kind: 'enter' | 'drop' }[]
}

const once = <T>(f: () => T): (() => T) => {
  let value: T | undefined
  return () => (value ??= f())
}

const rows = (t: Tensor) => toRows(t) as number[][]

/** Standardised features (442 × 10), centred target and feature names. */
export const diabetesData = once(() => {
  const raw = diabetes()
  const x = standardScaler().fit({ x: raw.x }).transform(raw.x)
  const target = toFlat(raw.y!)
  const mean = target.reduce((a, b) => a + b, 0) / target.length
  const y = fromData(Float64Array.from(target, (v) => v - mean))
  return { x, y, n: target.length, features: [...raw.meta.featureNames] as string[] }
})

const olsWeights = once(() => {
  const { x, y } = diabetesData()
  return Array.from(toFlat(linearRegression().fit(dataset(x, y)).weights))
})

/** Ridge path over 71 λ from 10⁵ to 10⁻², with the effective degrees of freedom and the OLS solution. */
export const ridgePathData = once(() => {
  const { x, y, features } = diabetesData()
  const lambdas = Array.from({ length: 71 }, (_, k) => 10 ** (5 - (7 * k) / 70))
  const path = ridgePath(x, y, lambdas)
  return {
    path: { features, penalty: [...path.lambdas], coef: rows(path.coef) } satisfies CoefficientPath,
    dof: [...path.dof],
    ols: olsWeights(),
  }
})

/** The elastic-net path for mixing ratio ρ on the default grid: 81 α over four decades below α_max. */
const enetPath = (l1Ratio: number): CoefficientPath & { nonzero: number[] } => {
  const { x, y, features } = diabetesData()
  const path = elasticNetPath(x, y, { l1Ratio })
  return { features, penalty: [...path.alphas], coef: rows(path.coef), nonzero: [...path.nonzero] }
}

/** The lasso path with the OLS solution. */
export const lassoPathData = once(() => ({ path: enetPath(1), ols: olsWeights() }))

/** Elastic-net paths for mixing ratios 0.1, 0.5, 0.9 and 1 (the lasso). */
export const L1_RATIOS = [0.1, 0.5, 0.9, 1]
export const elasticNetPathsData = once(() => L1_RATIOS.map(enetPath))

/** LARS and LARS–lasso paths. */
export const larsPathsData = once(() => {
  const { x, y, features } = diabetesData()
  const knots = (method: 'lar' | 'lasso'): KnotPath => {
    const path = leastAngleRegression(x, y, { method })
    return {
      l1Norm: [...path.l1Norm],
      alphas: [...path.alphas],
      coef: rows(path.coef),
      events: path.events.map((e) => ({ ...e, alpha: path.alphas[e.knot] })),
    }
  }
  return { features, lar: knots('lar'), lasso: knots('lasso') }
})

/** One SGD run of `linearSgdSteps` per epoch, against the exact minimiser of the same objective. */
export interface SgdRun {
  label: string
  /** Indexed [feature][epoch]; epoch 0 is the zero start. */
  coef: number[][]
  exact: number[]
  /** Objective minus the exact optimum at each epoch. */
  gap: number[]
  zeros: number[]
  exactZeros: number
}

export const SGD_ALPHA = 1
export const SGD_L1_RATIO = 0.5
export const SGD_EPOCHS = 60

const SGD_RUNS = [
  { label: 'ridge (l2)', penalty: 'l2', schedule: 'invscaling', l1Handling: 'truncated' },
  { label: 'lasso (l1)', penalty: 'l1', schedule: 'invscaling', l1Handling: 'truncated' },
  { label: 'elastic net', penalty: 'elasticnet', schedule: 'invscaling', l1Handling: 'truncated' },
  { label: 'lasso, constant step', penalty: 'l1', schedule: 'constant', l1Handling: 'truncated' },
  { label: 'lasso, plain subgradient', penalty: 'l1', schedule: 'invscaling', l1Handling: 'subgradient' },
] as const

/** The five SGD runs of the SGD note: ridge, lasso and elastic net, the lasso with a constant step and by subgradient. */
export const sgdData = once(() => {
  const { x, y, features } = diabetesData()
  const runs: SgdRun[] = SGD_RUNS.map((r, k) => {
    // Each penalty's objective has the scaling of elasticNet, so its exact minimiser is elasticNet with ρ = 0 (ridge),
    // 1 (lasso) or the elastic net's ρ.
    const l1Ratio = r.penalty === 'l2' ? 0 : r.penalty === 'l1' ? 1 : SGD_L1_RATIO
    const best = elasticNet({ alpha: SGD_ALPHA, l1Ratio }).fit(dataset(x, y))
    const tr = trace(
      linearSgdSteps({
        x,
        y,
        penalty: r.penalty,
        alpha: SGD_ALPHA,
        l1Ratio,
        schedule: r.schedule,
        l1Handling: r.l1Handling,
      }),
      {},
      SGD_EPOCHS,
      { keep: 'all', stream: child(stream('sgd-for-linear-models'), 'run', k) },
    )
    const weights = tr.steps.map((s) => Array.from(toFlat(s.weights)))
    return {
      label: r.label,
      coef: features.map((_, j) => weights.map((w) => w[j])),
      exact: Array.from(toFlat(best.weights)),
      gap: tr.steps.map((s) => Math.max(s.objective - best.objective, 1e-12)),
      zeros: tr.steps.map((s) => s.zeros),
      exactZeros: features.length - best.nonzero,
    }
  })
  return { features, runs }
})
