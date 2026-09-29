/**
 * A fully connected Boltzmann machine small enough to enumerate: V binary visible units and H binary hidden units,
 * every pair connected. Learning follows the exact log-likelihood gradient, or a version whose negative phase comes
 * from persistent Gibbs chains, as in the original algorithm.
 */

export type Uniform = () => number

export type Machine = { V: number; H: number; w: Float64Array; b: Float64Array }

/** Units 0…V−1 are visible, V…V+H−1 hidden. State code bit k is unit k (1 = on). */
export const unitOn = (code: number, k: number) => (code >> k) & 1

export function energy(m: Machine, code: number): number {
  const n = m.V + m.H
  let e = 0
  for (let i = 0; i < n; i++) {
    if (!unitOn(code, i)) continue
    e -= m.b[i]
    for (let j = i + 1; j < n; j++) if (unitOn(code, j)) e -= m.w[i * n + j]
  }
  return e
}

/** Probabilities of every joint state at temperature T. */
export function jointProbabilities(m: Machine, T = 1): Float64Array {
  const count = 1 << (m.V + m.H)
  const p = new Float64Array(count)
  let max = -Infinity
  for (let c = 0; c < count; c++) {
    p[c] = -energy(m, c) / T
    max = Math.max(max, p[c])
  }
  let z = 0
  for (let c = 0; c < count; c++) z += p[c] = Math.exp(p[c] - max)
  for (let c = 0; c < count; c++) p[c] /= z
  return p
}

/** Marginal probability of each visible configuration (code over the V visible bits). */
export function visibleMarginal(m: Machine, T = 1): Float64Array {
  const joint = jointProbabilities(m, T)
  const out = new Float64Array(1 << m.V)
  const mask = (1 << m.V) - 1
  for (let c = 0; c < joint.length; c++) out[c & mask] += joint[c]
  return out
}

export function kl(p: ArrayLike<number>, q: ArrayLike<number>): number {
  let d = 0
  for (let i = 0; i < p.length; i++) if (p[i] > 0) d += p[i] * Math.log(p[i] / Math.max(q[i], 1e-300))
  return d
}

/** Pair statistics ⟨s_i s_j⟩ (i < j, stored at i·n + j) and unit means ⟨s_i⟩ (stored at i·n + i). */
function statistics(n: number, weights: Iterable<[number, number]>): Float64Array {
  const s = new Float64Array(n * n)
  for (const [code, p] of weights) {
    if (p === 0) continue
    for (let i = 0; i < n; i++) {
      if (!unitOn(code, i)) continue
      s[i * n + i] += p
      for (let j = i + 1; j < n; j++) if (unitOn(code, j)) s[i * n + j] += p
    }
  }
  return s
}

/** Clamped phase: data visible vectors, hidden units from their exact posterior. */
function positivePhase(m: Machine, data: ArrayLike<number>): Float64Array {
  const n = m.V + m.H
  const pairs: [number, number][] = []
  for (let v = 0; v < data.length; v++) {
    if (data[v] === 0) continue
    const logits: number[] = []
    for (let h = 0; h < 1 << m.H; h++) logits.push(-energy(m, v | (h << m.V)))
    const max = Math.max(...logits)
    const z = logits.reduce((a, l) => a + Math.exp(l - max), 0)
    logits.forEach((l, h) => pairs.push([v | (h << m.V), (data[v] * Math.exp(l - max)) / z]))
  }
  return statistics(n, pairs)
}

function negativeExact(m: Machine): Float64Array {
  const joint = jointProbabilities(m)
  return statistics(
    m.V + m.H,
    Array.from(joint, (p, c): [number, number] => [c, p]),
  )
}

/** One sequential Gibbs sweep over every unit of each chain: s_i = 1 with probability σ(b_i + Σ_j w_ij s_j). */
export function gibbsSweep(m: Machine, chains: number[], uniform: Uniform, T = 1): void {
  const n = m.V + m.H
  for (let k = 0; k < chains.length; k++) {
    let code = chains[k]
    for (let i = 0; i < n; i++) {
      let field = m.b[i]
      for (let j = 0; j < n; j++) if (j !== i && unitOn(code, j)) field += m.w[Math.min(i, j) * n + Math.max(i, j)]
      const on = uniform() < 1 / (1 + Math.exp(-field / T))
      code = on ? code | (1 << i) : code & ~(1 << i)
    }
    chains[k] = code
  }
}

export type Snapshot = { marginal: Float64Array; kl: number; w: Float64Array; b: Float64Array }
export type Negative = 'exact' | 'gibbs'

/**
 * Train by gradient ascent on the log-likelihood: Δw_ij = η(⟨s_i s_j⟩_data − ⟨s_i s_j⟩_model). The negative phase is
 * exact, or estimated from `chains` persistent Gibbs chains advanced one sweep per update. Returns one snapshot per
 * epoch, the first before any update.
 */
export function train(
  data: ArrayLike<number>,
  V: number,
  H: number,
  {
    epochs,
    rate,
    negative,
    uniform,
    chains = 20,
    init = 1,
  }: {
    epochs: number
    rate: number
    negative: Negative
    uniform: Uniform
    chains?: number
    /** Initial weights are uniform on [−init, init]. */
    init?: number
  },
): Snapshot[] {
  const n = V + H
  const m: Machine = { V, H, w: new Float64Array(n * n), b: new Float64Array(n) }
  // Random weights break the symmetry between hidden units; at zero, every hidden unit gets the same gradient.
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m.w[i * n + j] = (uniform() - 0.5) * 2 * init
  const state = Array.from({ length: chains }, () => Math.floor(uniform() * (1 << n)))
  const snapshot = (): Snapshot => {
    const marginal = visibleMarginal(m)
    return { marginal, kl: kl(data, marginal), w: m.w.slice(), b: m.b.slice() }
  }
  const out = [snapshot()]
  for (let e = 0; e < epochs; e++) {
    const pos = positivePhase(m, data)
    let neg: Float64Array
    if (negative === 'exact') neg = negativeExact(m)
    else {
      gibbsSweep(m, state, uniform)
      neg = statistics(
        n,
        state.map((c): [number, number] => [c, 1 / chains]),
      )
    }
    for (let i = 0; i < n; i++) {
      m.b[i] += rate * (pos[i * n + i] - neg[i * n + i])
      for (let j = i + 1; j < n; j++) m.w[i * n + j] += rate * (pos[i * n + j] - neg[i * n + j])
    }
    out.push(snapshot())
  }
  return out
}
