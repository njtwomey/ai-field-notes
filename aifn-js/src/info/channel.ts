/**
 * Channel capacity and the rate–distortion function by the Blahut–Arimoto algorithms (Blahut, 1972, "Computation of
 * channel capacity and rate-distortion functions", IEEE Trans. Inf. Theory 18(4); Arimoto, 1972, same issue). Both are
 * traceable `Algorithm`s whose states carry the current distributions and bounds, so a figure can show them
 * converging. Internally in nats; the convenience functions take a `base`.
 */

import { run, type Algorithm } from 'aifn/trace'
import { fromData, fromRows, isTensor, toFlat, type Tensor } from 'aifn/tensor'

/** A matrix given as a tensor or as rows of numbers. */
export type MatrixInput = Tensor | readonly (readonly number[])[]

function matrixOf(m: MatrixInput, where: string): { rows: number; cols: number; values: Float64Array } {
  const t = isTensor(m) ? m : fromRows(m as number[][])
  if (t.shape.length !== 2) throw new RangeError(`${where}: expected a matrix`)
  return { rows: t.shape[0], cols: t.shape[1], values: Float64Array.from(toFlat(t)) }
}

function vectorOf(v: Tensor | ArrayLike<number>): Float64Array {
  return Float64Array.from(isTensor(v) ? toFlat(v) : Array.from(v))
}

// ── Channel capacity ─────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the capacity algorithm. */
export type CapacityOptions = {
  /** The channel W(y | x): rows are inputs x and sum to 1, columns are outputs y. */
  channel: MatrixInput
  /** Stop when upper − lower bound < tolerance (nats; default 1e-10). */
  tolerance?: number
  /** Starting input distribution (default uniform). */
  initial?: Tensor | ArrayLike<number>
}

/** A Blahut–Arimoto capacity state. All information quantities are in nats. */
export type CapacityState = {
  /** The channel W [|X|, |Y|] (carried so that `step` is pure). */
  channel: Tensor
  /** The current input distribution p(x). */
  input: Tensor
  /** The output distribution q(y) = Σₓ p(x) W(y | x). */
  output: Tensor
  /** D(W(·|x) ‖ q) for each input x. */
  divergences: Tensor
  /** I(p; W) = Σₓ p(x) D_x, the mutual information of the current input. */
  information: number
  /** log Σₓ p(x) e^{D_x} ≤ C (Blahut, 1972, Theorem 2). */
  lower: number
  /** maxₓ D_x ≥ C. */
  upper: number
  tolerance: number
  iteration: number
  converged: boolean
}

function capacityState(
  W: Float64Array,
  nx: number,
  ny: number,
  p: Float64Array,
  tolerance: number,
  iteration: number,
  channel: Tensor,
): CapacityState {
  const q = new Float64Array(ny)
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) q[y] += p[x] * W[x * ny + y]
  const D = new Float64Array(nx)
  for (let x = 0; x < nx; x++) {
    let s = 0
    for (let y = 0; y < ny; y++) {
      const w = W[x * ny + y]
      if (w > 0) s += w * Math.log(w / q[y])
    }
    D[x] = s
  }
  let information = 0
  let upper = -Infinity
  let m = -Infinity
  for (let x = 0; x < nx; x++) {
    information += p[x] * D[x]
    if (p[x] > 0) m = Math.max(m, D[x])
    upper = Math.max(upper, D[x])
  }
  let z = 0
  for (let x = 0; x < nx; x++) z += p[x] * Math.exp(D[x] - m)
  const lower = m + Math.log(z)
  return {
    channel,
    input: fromData(p, [nx]),
    output: fromData(q, [ny]),
    divergences: fromData(D, [nx]),
    information,
    lower,
    upper,
    tolerance,
    iteration,
    converged: upper - lower < tolerance,
  }
}

/**
 * Blahut–Arimoto for the capacity C = max_p I(X; Y) of a discrete memoryless channel. Each step reweights the input,
 * p′(x) ∝ p(x) exp D(W(·|x) ‖ q), which increases I(p; W) monotonically; the state carries the bounds
 * lower ≤ C ≤ upper, and `done` holds once they are within the tolerance.
 */
export const blahutArimotoCapacity: Algorithm<CapacityOptions, CapacityState> = {
  name: 'blahutArimotoCapacity',
  init({ channel, tolerance = 1e-10, initial }) {
    const { rows, cols, values } = matrixOf(channel, 'blahutArimotoCapacity')
    for (let x = 0; x < rows; x++) {
      let s = 0
      for (let y = 0; y < cols; y++) {
        const w = values[x * cols + y]
        if (!(w >= 0)) throw new RangeError('blahutArimotoCapacity: channel entries must be non-negative')
        s += w
      }
      if (Math.abs(s - 1) > 1e-9) throw new RangeError(`blahutArimotoCapacity: row ${x} of the channel sums to ${s}`)
    }
    const p = initial ? vectorOf(initial) : new Float64Array(rows).fill(1 / rows)
    const total = p.reduce((a, b) => a + b, 0)
    return capacityState(
      values,
      rows,
      cols,
      p.map((v) => v / total),
      tolerance,
      0,
      fromData(values, [rows, cols]),
    )
  },
  step(state) {
    const [nx, ny] = state.channel.shape
    const W = Float64Array.from(toFlat(state.channel))
    const p = toFlat(state.input)
    const D = toFlat(state.divergences)
    const m = Math.max(...D)
    const next = new Float64Array(nx)
    let z = 0
    for (let x = 0; x < nx; x++) {
      next[x] = p[x] * Math.exp(D[x] - m)
      z += next[x]
    }
    for (let x = 0; x < nx; x++) next[x] /= z
    return capacityState(W, nx, ny, next, state.tolerance, state.iteration + 1, state.channel)
  },
  done: (state) => state.converged,
}

/** The result of `channelCapacity`, in the requested base. */
export type ChannelCapacity = {
  capacity: number
  /** The capacity-achieving input distribution (approximately, at the tolerance). */
  input: Tensor
  output: Tensor
  lower: number
  upper: number
  iterations: number
  converged: boolean
}

/**
 * The capacity of a discrete memoryless channel W(y | x) (rows x) by Blahut–Arimoto, in nats or the given `base`
 * (2 for bits). Runs until the bounds meet within the tolerance or `maxIterations` (default 10,000) is reached;
 * `converged` says which. The reported capacity is the lower bound.
 *
 * @example channelCapacity([[0.9, 0.1], [0.1, 0.9]], { base: 2 }).capacity // 1 − H₂(0.1) ≈ 0.531
 */
export function channelCapacity(
  channel: MatrixInput,
  { tolerance, maxIterations = 10_000, base }: { tolerance?: number; maxIterations?: number; base?: number } = {},
): ChannelCapacity {
  const s = run(blahutArimotoCapacity, { channel, tolerance }, maxIterations)
  const unit = base === undefined ? 1 : 1 / Math.log(base)
  return {
    capacity: s.lower * unit,
    input: s.input,
    output: s.output,
    lower: s.lower * unit,
    upper: s.upper * unit,
    iterations: s.iteration,
    converged: s.converged,
  }
}

// ── Rate–distortion ──────────────────────────────────────────────────────────────────────────────────────────────────

/** Options of the rate–distortion algorithm. */
export type RateDistortionOptions = {
  /** The source distribution p(x). */
  source: Tensor | ArrayLike<number>
  /** The distortion d(x, x̂) ≥ 0, [|X|, |X̂|]. */
  distortion: MatrixInput
  /** The slope parameter β ≥ 0 (minus the slope of R(D) at the point found): larger β gives lower distortion. */
  beta: number
  /** Stop when the reproduction marginal changes by less than this (max abs; default 1e-12). */
  tolerance?: number
}

/** A Blahut–Arimoto rate–distortion state (nats). */
export type RateDistortionState = {
  source: Tensor
  distortionMatrix: Tensor
  beta: number
  /** The test channel Q(x̂ | x), [|X|, |X̂|]. */
  conditional: Tensor
  /** The reproduction marginal q(x̂) = Σₓ p(x) Q(x̂ | x). */
  marginal: Tensor
  /** I(X; X̂) under p(x) Q(x̂ | x). */
  rate: number
  /** E[d(X, X̂)]. */
  distortion: number
  /** max |q′ − q| in the last step. */
  change: number
  tolerance: number
  iteration: number
  converged: boolean
}

/** The test channel from a reproduction marginal: Q(x̂|x) ∝ q(x̂) e^{−β d(x, x̂)}, and its rate and distortion. */
function rateDistortionState(
  p: Float64Array,
  d: Float64Array,
  nx: number,
  ny: number,
  beta: number,
  q: Float64Array,
  previous: Float64Array | null,
  tolerance: number,
  iteration: number,
  tensors: { source: Tensor; distortionMatrix: Tensor },
): RateDistortionState {
  const Q = new Float64Array(nx * ny)
  for (let x = 0; x < nx; x++) {
    // Normalise with the largest exponent subtracted, so large β does not underflow every entry.
    let m = -Infinity
    for (let y = 0; y < ny; y++) if (q[y] > 0) m = Math.max(m, Math.log(q[y]) - beta * d[x * ny + y])
    let z = 0
    for (let y = 0; y < ny; y++) {
      const v = q[y] > 0 ? Math.exp(Math.log(q[y]) - beta * d[x * ny + y] - m) : 0
      Q[x * ny + y] = v
      z += v
    }
    for (let y = 0; y < ny; y++) Q[x * ny + y] /= z
  }
  const marginal = new Float64Array(ny)
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) marginal[y] += p[x] * Q[x * ny + y]
  let rate = 0
  let distortion = 0
  for (let x = 0; x < nx; x++)
    for (let y = 0; y < ny; y++) {
      const v = Q[x * ny + y]
      if (v > 0 && p[x] > 0) rate += p[x] * v * Math.log(v / marginal[y])
      distortion += p[x] * v * d[x * ny + y]
    }
  let change = Infinity
  if (previous) {
    change = 0
    for (let y = 0; y < ny; y++) change = Math.max(change, Math.abs(marginal[y] - previous[y]))
  }
  return {
    ...tensors,
    beta,
    conditional: fromData(Q, [nx, ny]),
    marginal: fromData(marginal, [ny]),
    rate,
    distortion,
    change,
    tolerance,
    iteration,
    converged: change < tolerance,
  }
}

/**
 * Blahut–Arimoto for one point of the rate–distortion function R(D) = min I(X; X̂) subject to E d(X, X̂) ≤ D. For
 * a slope β it alternates Q(x̂|x) ∝ q(x̂) e^{−β d(x, x̂)} and q(x̂) = Σₓ p(x) Q(x̂|x) (Blahut, 1972, §IV; Cover and
 * Thomas, 2006, §10.8); the state's (distortion, rate) converges to the point of the curve with slope −β.
 */
export const blahutArimotoRateDistortion: Algorithm<RateDistortionOptions, RateDistortionState> = {
  name: 'blahutArimotoRateDistortion',
  init({ source, distortion, beta, tolerance = 1e-12 }) {
    if (!(beta >= 0)) throw new RangeError('blahutArimotoRateDistortion: beta must be non-negative')
    const raw = vectorOf(source)
    const total = raw.reduce((a, b) => a + b, 0)
    const p = raw.map((v) => v / total)
    const { rows, cols, values } = matrixOf(distortion, 'blahutArimotoRateDistortion')
    if (rows !== p.length)
      throw new RangeError('blahutArimotoRateDistortion: distortion needs one row per source symbol')
    const q = new Float64Array(cols).fill(1 / cols)
    const tensors = { source: fromData(p, [rows]), distortionMatrix: fromData(values, [rows, cols]) }
    return rateDistortionState(p, values, rows, cols, beta, q, null, tolerance, 0, tensors)
  },
  step(state) {
    const [nx, ny] = state.distortionMatrix.shape
    const p = Float64Array.from(toFlat(state.source))
    const d = Float64Array.from(toFlat(state.distortionMatrix))
    const q = Float64Array.from(toFlat(state.marginal))
    return rateDistortionState(p, d, nx, ny, state.beta, q, q, state.tolerance, state.iteration + 1, {
      source: state.source,
      distortionMatrix: state.distortionMatrix,
    })
  },
  done: (state) => state.converged,
}

/** One point of R(D). */
export type RateDistortionPoint = {
  rate: number
  distortion: number
  conditional: Tensor
  marginal: Tensor
  iterations: number
  converged: boolean
}

/**
 * The point of the rate–distortion curve with slope −β, by Blahut–Arimoto: rate in nats (or `base`), distortion in
 * the units of `distortion`. Runs to the tolerance or `maxIterations` (default 10,000).
 *
 * @example rateDistortion([0.5, 0.5], [[0, 1], [1, 0]], 3, { base: 2 }) // Hamming distortion: R = 1 − H₂(D)
 */
export function rateDistortion(
  source: Tensor | ArrayLike<number>,
  distortion: MatrixInput,
  beta: number,
  { tolerance, maxIterations = 10_000, base }: { tolerance?: number; maxIterations?: number; base?: number } = {},
): RateDistortionPoint {
  const s = run(blahutArimotoRateDistortion, { source, distortion, beta, tolerance }, maxIterations)
  return {
    rate: base === undefined ? s.rate : s.rate / Math.log(base),
    distortion: s.distortion,
    conditional: s.conditional,
    marginal: s.marginal,
    iterations: s.iteration,
    converged: s.converged,
  }
}

/** R(D) traced out over slopes β: `rate` and `distortion` tensors aligned with `betas`. */
export function rateDistortionCurve(
  source: Tensor | ArrayLike<number>,
  distortion: MatrixInput,
  betas: ArrayLike<number>,
  options: { tolerance?: number; maxIterations?: number; base?: number } = {},
): { rate: Tensor; distortion: Tensor; converged: boolean } {
  const n = betas.length
  const rate = new Float64Array(n)
  const dist = new Float64Array(n)
  let converged = true
  for (let i = 0; i < n; i++) {
    const point = rateDistortion(source, distortion, betas[i], options)
    rate[i] = point.rate
    dist[i] = point.distortion
    converged &&= point.converged
  }
  return { rate: fromData(rate, [n]), distortion: fromData(dist, [n]), converged }
}
