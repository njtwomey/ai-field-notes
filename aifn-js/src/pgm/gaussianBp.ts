/**
 * Gaussian belief propagation on a pairwise Gaussian Markov random field in information form,
 * p(x) ∝ exp(−½ xᵀJx + hᵀx), with scalar variables (Weiss & Freeman 2001, "Correctness of belief propagation in
 * Gaussian graphical models of arbitrary topology", Neural Computation 13(10); Bickson 2008, "Gaussian belief
 * propagation: theory and application", §2).
 *
 * The message from i to j is a Gaussian in x_j with precision Λ_{i→j} and potential η_{i→j}:
 * Λ̂ = J_ii + Σ_{k∈N(i)\j} Λ_{k→i}, η̂ = h_i + Σ_{k∈N(i)\j} η_{k→i}, Λ_{i→j} = −J_ij²/Λ̂, η_{i→j} = −J_ij η̂/Λ̂.
 * At a fixed point the means are exact on any graph; the variances are exact on trees.
 */

import { adjacency, fromEdges, type Graph } from 'aifn/graph'
import { fromData, toRows, type Matrix, type Tensor } from 'aifn/tensor'
import { run, type Algorithm } from 'aifn/trace'

/** Options of {@link gaussianBeliefPropagationSteps}. */
export interface GaussianBpOptions {
  /** The precision (information) matrix J, n × n symmetric; its off-diagonal non-zeros are the graph's edges. */
  precision: Matrix | readonly (readonly number[])[]
  /** The potential vector h = J μ, length n. */
  shift: Tensor | readonly number[]
  /** `flooding` (all messages from the previous sweep) or `sequential` (in edge order, newest messages). */
  schedule?: 'flooding' | 'sequential'
  /** Weight of the old message, in [0, 1). Default 0. */
  damping?: number
  tolerance?: number
}

/** The state of Gaussian BP; messages are indexed by directed edge (`from`, `to`). */
export interface GaussianBpState {
  n: number
  J: number[][]
  h: number[]
  graph: Graph
  /** Directed edges: message k travels from[k] → to[k]. */
  from: Int32Array
  to: Int32Array
  schedule: 'flooding' | 'sequential'
  damping: number
  tolerance: number
  /** Λ_{i→j} and η_{i→j} per directed edge. */
  messagePrecision: Tensor
  messageShift: Tensor
  /** Marginal means and variances from the current messages. */
  means: Tensor
  variances: Tensor
  sweep: number
  change: number
  converged: boolean
  /** A non-positive Λ̂ appeared (the model is not walk-summable here): the run stops. */
  diverged: boolean
}

function marginals(s: Pick<GaussianBpState, 'n' | 'J' | 'h' | 'to'>, lambda: Float64Array, eta: Float64Array) {
  const P = s.J.map((row, i) => row[i])
  const H = [...s.h]
  for (let k = 0; k < s.to.length; k++) {
    P[s.to[k]] += lambda[k]
    H[s.to[k]] += eta[k]
  }
  return {
    means: fromData(
      Float64Array.from(H, (hi, i) => hi / P[i]),
      [s.n],
    ),
    variances: fromData(
      Float64Array.from(P, (p) => 1 / p),
      [s.n],
    ),
  }
}

/**
 * Gaussian BP as a traceable algorithm; each step is one sweep over every directed edge. Messages start at zero
 * (Λ = η = 0). The run is done when no message moves by more than `tolerance`, and stops as `diverged` when a
 * cavity precision Λ̂ is not positive.
 */
export const gaussianBeliefPropagationSteps: Algorithm<GaussianBpOptions, GaussianBpState> = {
  name: 'pgm.gaussian-belief-propagation',
  init: (o) => {
    const J = 'shape' in o.precision ? toRows(o.precision as Tensor) : o.precision.map((r) => [...r])
    const h = 'shape' in o.shift ? Array.from((o.shift as Tensor).data) : [...(o.shift as readonly number[])]
    const n = J.length
    const pairs: [number, number][] = []
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (J[i][j] !== 0) pairs.push([i, j])
    const graph = fromEdges(n, pairs, { directed: false })
    const from = new Int32Array(2 * pairs.length)
    const to = new Int32Array(2 * pairs.length)
    pairs.forEach(([i, j], k) => {
      from[2 * k] = i
      to[2 * k] = j
      from[2 * k + 1] = j
      to[2 * k + 1] = i
    })
    const lambda = new Float64Array(from.length)
    const eta = new Float64Array(from.length)
    const base = { n, J, h, to }
    return {
      n,
      J,
      h,
      graph,
      from,
      to,
      schedule: o.schedule ?? 'flooding',
      damping: o.damping ?? 0,
      tolerance: o.tolerance ?? 1e-10,
      messagePrecision: fromData(lambda, [lambda.length]),
      messageShift: fromData(eta, [eta.length]),
      ...marginals(base, lambda, eta),
      sweep: 0,
      change: Infinity,
      converged: false,
      diverged: false,
    }
  },
  step: (s) => {
    if (s.converged || s.diverged) return s
    const old = { lambda: Float64Array.from(s.messagePrecision.data), eta: Float64Array.from(s.messageShift.data) }
    const lambda = Float64Array.from(old.lambda)
    const eta = Float64Array.from(old.eta)
    // Incoming messages of each node, by directed-edge index.
    const incoming = adjacency(s.graph).map((_, i) => {
      const ks: number[] = []
      for (let k = 0; k < s.to.length; k++) if (s.to[k] === i) ks.push(k)
      return ks
    })
    const read = s.schedule === 'flooding' ? old : { lambda, eta }
    let change = 0
    let diverged = false
    for (let k = 0; k < s.from.length; k++) {
      const i = s.from[k]
      const j = s.to[k]
      let L = s.J[i][i]
      let E = s.h[i]
      for (const m of incoming[i]) {
        if (s.from[m] === j) continue
        L += read.lambda[m]
        E += read.eta[m]
      }
      if (!(L > 0)) diverged = true
      const nl = (-s.J[i][j] * s.J[i][j]) / L
      const ne = (-s.J[i][j] * E) / L
      const dl = (1 - s.damping) * nl + s.damping * old.lambda[k]
      const de = (1 - s.damping) * ne + s.damping * old.eta[k]
      change = Math.max(change, Math.abs(dl - old.lambda[k]), Math.abs(de - old.eta[k]))
      lambda[k] = dl
      eta[k] = de
    }
    return {
      ...s,
      messagePrecision: fromData(lambda, [lambda.length]),
      messageShift: fromData(eta, [eta.length]),
      ...marginals(s, lambda, eta),
      sweep: s.sweep + 1,
      change,
      converged: !diverged && change < s.tolerance,
      diverged: diverged || !Number.isFinite(change),
    }
  },
  done: (s) => s.converged,
}

/** Run Gaussian BP for at most `maxSweeps` sweeps (default 500): marginal means and variances, and the flags. */
export function gaussianBeliefPropagation(
  precision: GaussianBpOptions['precision'],
  shift: GaussianBpOptions['shift'],
  options: Omit<GaussianBpOptions, 'precision' | 'shift'> & { maxSweeps?: number } = {},
): { means: Tensor; variances: Tensor; sweeps: number; converged: boolean; diverged: boolean } {
  const { maxSweeps = 500, ...rest } = options
  const s = run(gaussianBeliefPropagationSteps, { ...rest, precision, shift }, maxSweeps)
  return { means: s.means, variances: s.variances, sweeps: s.sweep, converged: s.converged, diverged: s.diverged }
}
