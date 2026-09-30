/**
 * Entropic Gromov–Wasserstein transport between two metric-measure spaces (Peyré, Cuturi and Solomon, 2016, "Gromov–
 * Wasserstein averaging of kernel and distance matrices", ICML, Algorithm 1 with the square loss): points are matched by
 * how they relate to the other points of their own space, not by a cross-space cost. Each step linearises the quadratic
 * objective at the current plan and solves the entropic problem with Sinkhorn.
 */

import { fromData, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { run } from 'aifn/trace'
import { readCost, readVector, sinkhorn, type CostInput, type WeightsInput } from './discrete'

/** Options for `gromovWassersteinSteps`. */
export interface GromovOptions {
  /** Intra-space distance (or similarity) matrices, n × n and m × m. */
  cx: CostInput
  cy: CostInput
  a: WeightsInput
  b: WeightsInput
  epsilon: number
  /** Sinkhorn iterations per outer step. Default 200. */
  innerIterations?: number
  /** Stop when the plan changes by less than this (max abs). Default 1e-7. */
  tolerance?: number
}

/** One state of entropic Gromov–Wasserstein. */
export interface GromovState {
  /** The coupling, n × m. */
  plan: Tensor
  /** The GW objective Σ_{ijkl} (Cx_ik − Cy_jl)² T_ij T_kl at the plan. */
  loss: number
  /** The linearised cost the last Sinkhorn solve used, n × m. */
  linearCost: Tensor
  /** Largest change of a plan entry in the last step. */
  change: number
  iteration: number
  converged: boolean
  options: GromovOptions
}

function linearise(Cx: Float64Array, Cy: Float64Array, a: Float64Array, b: Float64Array, T: Float64Array) {
  const n = a.length
  const m = b.length
  // Square loss: L(T) = const − 2 Cx T Cyᵀ with const_ij = Σ_k Cx_ik² a_k + Σ_l Cy_jl² b_l.
  const rowTerm = new Float64Array(n)
  const colTerm = new Float64Array(m)
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) rowTerm[i] += Cx[i * n + k] ** 2 * a[k]
  for (let j = 0; j < m; j++) for (let l = 0; l < m; l++) colTerm[j] += Cy[j * m + l] ** 2 * b[l]
  const CxT = new Float64Array(n * m)
  for (let i = 0; i < n; i++)
    for (let k = 0; k < n; k++) {
      const c = Cx[i * n + k]
      if (c === 0) continue
      for (let l = 0; l < m; l++) CxT[i * m + l] += c * T[k * m + l]
    }
  const L = new Float64Array(n * m)
  let loss = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let cross = 0
      for (let l = 0; l < m; l++) cross += CxT[i * m + l] * Cy[j * m + l]
      L[i * m + j] = rowTerm[i] + colTerm[j] - 2 * cross
      loss += L[i * m + j] * T[i * m + j]
    }
  return { L, loss }
}

function read(o: GromovOptions) {
  const a = readVector(o.a, 'gromovWasserstein a')
  const b = readVector(o.b, 'gromovWasserstein b')
  return {
    a,
    b,
    Cx: readCost(o.cx, a.length, a.length, 'gromovWasserstein cx'),
    Cy: readCost(o.cy, b.length, b.length, 'gromovWasserstein cy'),
  }
}

/** Entropic Gromov–Wasserstein as a traceable algorithm, started from the product coupling a bᵀ. */
export const gromovWassersteinSteps: Algorithm<GromovOptions, GromovState> = {
  name: 'gromov-wasserstein',
  init(options) {
    const { a, b, Cx, Cy } = read(options)
    const T = new Float64Array(a.length * b.length)
    for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) T[i * b.length + j] = a[i] * b[j]
    const { L, loss } = linearise(Cx, Cy, a, b, T)
    return {
      plan: fromData(T, [a.length, b.length]),
      loss,
      linearCost: fromData(L, [a.length, b.length]),
      change: Infinity,
      iteration: 0,
      converged: false,
      options,
    }
  },
  step(s) {
    const { a, b, Cx, Cy } = read(s.options)
    const solved = sinkhorn(a, b, s.linearCost, {
      epsilon: s.options.epsilon,
      maxIterations: s.options.innerIterations ?? 200,
    })
    const T = solved.plan.data as Float64Array
    const prev = s.plan.data
    let change = 0
    for (let k = 0; k < T.length; k++) change = Math.max(change, Math.abs(T[k] - prev[k]))
    const { L, loss } = linearise(Cx, Cy, a, b, T)
    return {
      ...s,
      plan: solved.plan,
      loss,
      linearCost: fromData(L, [a.length, b.length]),
      change,
      iteration: s.iteration + 1,
      converged: change < (s.options.tolerance ?? 1e-7),
    }
  },
  done: (s) => s.converged,
}

/** Entropic Gromov–Wasserstein run to convergence or `maxIterations` (default 50) outer steps. */
export function gromovWasserstein(options: GromovOptions & { maxIterations?: number }): GromovState {
  return run(gromovWassersteinSteps, options, options.maxIterations ?? 50)
}
