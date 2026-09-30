/**
 * Samplers. Every sampler takes the stream first and draws only from it, so a draw is reproducible from the stream's
 * key and position.
 *
 * Shapes. Parameters are numbers or tensors and broadcast against each other (NumPy rules); `options.shape` sets the
 * shape of the draws, which the parameters must broadcast to. With number parameters and no `shape` a sampler returns
 * a number; otherwise it returns a tensor: float64 for values (counts and Bernoulli outcomes included, so that invalid
 * parameters can still give NaN), int32 for indices (`categorical`, `aliasSample`, `choice`, `permutation`).
 * Vector-valued samplers (`dirichlet`, `multinomial`, `multivariateNormal`) append the event axis to the batch shape.
 *
 * Element order. The elements of a tensor of draws are drawn one after another in row-major order, each exactly as the
 * scalar sampler would draw it. So `normal(s, 0, 1, { shape: [2, 3] })` holds the same six values as six successive
 * `normal(s)` calls, and the same stream with the same shape always gives the same draws.
 *
 * Samplers are not differentiable primitives: they have no derivative rule, and they do not accept traced values. A
 * reparameterised draw is written in terms of differentiable operations on a base draw instead
 * (e.g. `add(mean, mul(sd, normal(s, 0, 1, { shape })))`), which autodiff handles.
 */

import { cholesky, type CholeskyOptions } from 'aifn/linalg'
import { logGamma } from 'aifn/special'
import { arange, broadcastShapes, broadcastTo, fromData, isTensor, reshape, toFlat, type Tensor } from 'aifn/tensor'
import type { Stream } from './stream'

// ── Shapes and broadcasting ──────────────────────────────────────────────────────────────────────────────────────────

/** A sampler parameter: a number, or a tensor broadcast against the other parameters and `shape`. */
export type Param = number | Tensor

/** Options shared by the samplers. */
export type SampleOptions = {
  /**
   * The shape of the draws (for vector-valued samplers, the batch shape; the event axis is appended). The parameters'
   * broadcast shape must broadcast to it. Omitted: the parameters' broadcast shape, or a single number when every
   * parameter is a number.
   */
  shape?: readonly number[]
}

type AnyTensor<P extends readonly unknown[]> = true extends {
  [K in keyof P]: P[K] extends Tensor ? true : false
}[number]
  ? true
  : false

/**
 * The result type of a sampler with parameters `P` and options `O`: a tensor when `O` sets a shape or any parameter
 * is a tensor, a number when every parameter is a number and no shape is set, and `number | Tensor` when that is only
 * known at run time.
 */
export type Drawn<P extends readonly unknown[], O> = O extends { shape: readonly number[] }
  ? Tensor
  : AnyTensor<P> extends true
    ? Tensor
    : [P[number]] extends [number | undefined]
      ? O extends { shape?: undefined }
        ? number
        : number | Tensor
      : number | Tensor

// Box–Muller produces normals in pairs; the second of each pair is kept here, per stream object.
const spareNormal = new WeakMap<Stream, number>()

const showShape = (shape: readonly number[]) => `[${shape.join(', ')}]`

/** Check that a parameter shape broadcasts to the requested shape (the requested shape wins, as NumPy's `size`). */
function checkBroadcast(name: string, from: readonly number[], to: readonly number[]): void {
  let joint: number[] | null = null
  try {
    joint = broadcastShapes(from, to)
  } catch {
    joint = null
  }
  if (joint === null || joint.length !== to.length || joint.some((d, k) => d !== to[k]))
    throw new RangeError(`${name}: parameters of shape ${showShape(from)} do not broadcast to shape ${showShape(to)}`)
}

/** The output shape of an elementwise sampler: `options.shape`, or the parameters' broadcast shape. */
function outputShape(name: string, params: readonly Param[], options: SampleOptions | undefined): readonly number[] {
  const joint = broadcastShapes([], ...params.filter(isTensor).map((t) => t.shape))
  if (options?.shape === undefined) return joint
  checkBroadcast(name, joint, options.shape)
  return options.shape
}

/**
 * Draw elementwise: `draw(...values)` for each element of the output in row-major order, with every parameter
 * broadcast to the output shape. Returns a number when every parameter is a number and no shape is given.
 */
function elementwise(
  name: string,
  params: readonly Param[],
  options: SampleOptions | undefined,
  draw: (...values: number[]) => number,
): number | Tensor {
  if (options?.shape === undefined && params.every((p) => typeof p === 'number')) return draw(...(params as number[]))
  const shape = outputShape(name, params, options)
  const flats = params.map((p) => (typeof p === 'number' ? null : toFlat(broadcastTo(p, shape))))
  const n = shape.reduce((a, b) => a * b, 1)
  const out = new Float64Array(n)
  const values = params.map((p) => (typeof p === 'number' ? p : NaN))
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < flats.length; i++) {
      const f = flats[i]
      if (f) values[i] = f[k]
    }
    out[k] = draw(...values)
  }
  return fromData(out, shape)
}

/** A vector-valued parameter as its batch shape, event length and row-major values. */
function eventRows(x: Tensor | ArrayLike<number>, name: string): { batch: number[]; k: number; values: Float64Array } {
  if (!isTensor(x)) return { batch: [], k: x.length, values: Float64Array.from(x) }
  if (x.shape.length === 0) throw new RangeError(`${name}: needs a vector (or a batch of vectors), not a scalar`)
  return { batch: x.shape.slice(0, -1), k: x.shape[x.shape.length - 1], values: Float64Array.from(toFlat(x)) }
}

/** For each element of `shape` (row-major), the row-major index of the batch element it broadcasts from. */
function batchIndex(batch: readonly number[], shape: readonly number[]): number[] {
  const n = batch.reduce((a, b) => a * b, 1)
  return toFlat(broadcastTo(reshape(arange(n), batch), shape))
}

/** A copy of an array as a plain `ArrayLike` of numbers (tensors flattened; they must be rank 1). */
function vectorValues(x: Tensor | ArrayLike<number>, name: string): ArrayLike<number> {
  if (!isTensor(x)) return x
  if (x.shape.length !== 1) throw new RangeError(`${name}: needs a rank-1 tensor, got shape ${showShape(x.shape)}`)
  return toFlat(x)
}

// ── Scalar draws (internal) ──────────────────────────────────────────────────────────────────────────────────────────

/**
 * A normal draw by Box–Muller (Box and Muller 1958), using both outputs of each pair, with no clamp on the tail: the
 * radius uses 1 − u in (0, 1], so |z| can reach about 8.57 (the limit set by 53-bit uniforms, probability 1e-17).
 */
function normalDraw(s: Stream, mean: number, sd: number): number {
  const spare = spareNormal.get(s)
  if (spare !== undefined) {
    spareNormal.delete(s)
    return mean + sd * spare
  }
  const r = Math.sqrt(-2 * Math.log(1 - s.uniform()))
  const theta = 2 * Math.PI * s.uniform()
  spareNormal.set(s, r * Math.sin(theta))
  return mean + sd * r * Math.cos(theta)
}

/**
 * log of a Gamma(shape, 1) draw. Marsaglia and Tsang (2000), "A simple method for generating gamma variables", ACM TOMS
 * 26(3); for shape < 1 the boost G(a) = G(a + 1) · U^{1/a} (their §6), kept in log space so that tiny shapes do not
 * underflow to 0.
 */
function logGammaDraw(s: Stream, shape: number): number {
  if (!(shape > 0)) return NaN
  if (shape < 1) return logGammaDraw(s, shape + 1) + Math.log(1 - s.uniform()) / shape
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  for (;;) {
    const x = normalDraw(s, 0, 1)
    const t = 1 + c * x
    if (t <= 0) continue
    const v = t * t * t
    const u = s.uniform()
    // The squeeze accepts about 98% of proposals without a logarithm.
    if (u < 1 - 0.0331 * x * x * x * x) return Math.log(d * v)
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return Math.log(d * v)
  }
}

/** A Beta(a, b) draw as a sigmoid of the difference of two log-gamma draws (no 0/0 for small a or b). */
function betaDraw(s: Stream, a: number, b: number): number {
  const la = logGammaDraw(s, a)
  const lb = logGammaDraw(s, b)
  const d = lb - la
  return d > 0 ? Math.exp(-d) / (1 + Math.exp(-d)) : 1 / (1 + Math.exp(d))
}

/** A Poisson(λ) draw: inversion for λ < 10, Hörmann's PTRS for λ ≥ 10 (see `poisson`). */
function poissonDraw(s: Stream, lambda: number): number {
  if (!(lambda >= 0)) return NaN
  if (lambda === 0) return 0
  if (lambda < 10) {
    const u = s.uniform()
    let k = 0
    let p = Math.exp(-lambda)
    let cdf = p
    while (u >= cdf && k < 1000) {
      k++
      p *= lambda / k
      cdf += p
    }
    return k
  }
  const slam = Math.sqrt(lambda)
  const logLam = Math.log(lambda)
  const b = 0.931 + 2.53 * slam
  const a = -0.059 + 0.02483 * b
  const invAlpha = 1.1239 + 1.1328 / (b - 3.4)
  const vr = 0.9277 - 3.6224 / (b - 2)
  for (;;) {
    const u = s.uniform() - 0.5
    const v = s.uniform()
    const us = 0.5 - Math.abs(u)
    const k = Math.floor(((2 * a) / us + b) * u + lambda + 0.43)
    if (us >= 0.07 && v <= vr) return k
    if (k < 0 || (us < 0.013 && v > us)) continue
    if (Math.log(v) + Math.log(invAlpha) - Math.log(a / (us * us) + b) <= -lambda + k * logLam - logGamma(k + 1))
      return k
  }
}

/** A Binomial(n, p) draw: BINV inversion for small n·min(p, 1 − p), else the order-statistic recursion. */
function binomialDraw(s: Stream, n: number, p: number): number {
  if (!(Number.isInteger(n) && n >= 0 && p >= 0 && p <= 1)) return NaN
  if (n === 0 || p === 0) return 0
  if (p === 1) return n
  if (p > 0.5) return n - binomialDraw(s, n, 1 - p)
  if (n * p < 30) {
    const q = 1 - p
    const ratio = p / q
    for (;;) {
      let u = s.uniform()
      let r = Math.pow(q, n)
      let k = 0
      while (u > r && k <= n) {
        u -= r
        k++
        r *= ((n - k + 1) / k) * ratio
      }
      // Rounding can leave u above the whole mass; redraw rather than return an impossible k.
      if (k <= n) return k
    }
  }
  const a = 1 + Math.floor(n / 2)
  const b = n + 1 - a
  const x = betaDraw(s, a, b)
  if (x >= p) return binomialDraw(s, a - 1, p / x)
  return a + binomialDraw(s, b - 1, (p - x) / (1 - x))
}

/** A chi-square draw: 2 · Gamma(k/2, 1). */
function chiSquareDraw(s: Stream, df: number): number {
  return 2 * Math.exp(logGammaDraw(s, df / 2))
}

/** One categorical index from non-negative weights w[offset … offset + k − 1] (one uniform, a linear scan). */
function categoricalDraw(s: Stream, w: ArrayLike<number>, offset: number, k: number): number {
  let total = 0
  for (let j = 0; j < k; j++) total += w[offset + j]
  if (!(total > 0) || !Number.isFinite(total))
    throw new RangeError('categorical needs finite weights with a positive sum')
  const u = s.uniform() * total
  let acc = 0
  let last = -1
  for (let j = 0; j < k; j++) {
    const wj = w[offset + j]
    if (wj <= 0) continue
    acc += wj
    last = j
    if (u < acc) return j
  }
  // Rounding in the running sum can leave u just above it; the draw then belongs to the last positive weight.
  return last
}

// ── Continuous samplers ──────────────────────────────────────────────────────────────────────────────────────────────

/** Uniform draws on [a, b) (default [0, 1)), elementwise over broadcast a and b. */
export function uniform<A extends Param = number, B extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  a?: A,
  b?: B,
  options?: O,
): Drawn<[A, B], O> {
  return elementwise('uniform', [a ?? 0, b ?? 1], options, (lo, hi) => lo + (hi - lo) * s.uniform()) as Drawn<[A, B], O>
}

/**
 * Normal draws with the given mean and standard deviation (default N(0, 1)), elementwise over broadcast parameters.
 * Box–Muller (Box and Muller 1958), using both outputs of each pair, with no clamp on the tail: |z| can reach about
 * 8.57 (the limit set by 53-bit uniforms, probability 1e-17).
 */
export function normal<M extends Param = number, D extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  mean?: M,
  sd?: D,
  options?: O,
): Drawn<[M, D], O> {
  return elementwise('normal', [mean ?? 0, sd ?? 1], options, (m, d) => normalDraw(s, m, d)) as Drawn<[M, D], O>
}

/**
 * Normal draws as a tensor of shape `[n]` (or the given shape), with mean and standard deviation broadcast to it:
 * `normal(s, mean, sd, { shape })` with the shape spelled as a length.
 */
export function normals(s: Stream, n: number | readonly number[], mean: Param = 0, sd: Param = 1): Tensor {
  return normal(s, mean, sd, { shape: typeof n === 'number' ? [n] : n })
}

/** Exponential draws with rate λ > 0 (mean 1/λ), by inversion −log(1 − u)/λ, elementwise over broadcast λ. */
export function exponential<R extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  rate?: R,
  options?: O,
): Drawn<[R], O> {
  return elementwise('exponential', [rate ?? 1], options, (r) => -Math.log1p(-s.uniform()) / r) as Drawn<[R], O>
}

/**
 * log of Gamma(shape, 1) draws, elementwise (NaN for shape ≤ 0). Marsaglia and Tsang (2000), "A simple method for
 * generating gamma variables", ACM TOMS 26(3); for shape < 1 the boost G(a) = G(a + 1) · U^{1/a} (their §6), kept in
 * log space so that tiny shapes do not underflow to 0.
 */
export function logGammaVariate<A extends Param, O extends SampleOptions = object>(
  s: Stream,
  shape: A,
  options?: O,
): Drawn<[A], O> {
  return elementwise('logGammaVariate', [shape], options, (a) => logGammaDraw(s, a)) as Drawn<[A], O>
}

/**
 * Gamma(shape, scale) draws (mean shape · scale), Marsaglia–Tsang, including shape < 1, elementwise over broadcast
 * parameters. For very small shapes a value can underflow to 0; use `logGammaVariate` when that matters.
 */
export function gamma<A extends Param, C extends Param = number, O extends SampleOptions = object>(
  s: Stream,
  shape: A,
  scale?: C,
  options?: O,
): Drawn<[A, C], O> {
  return elementwise('gamma', [shape, scale ?? 1], options, (a, c) => c * Math.exp(logGammaDraw(s, a))) as Drawn<
    [A, C],
    O
  >
}

/**
 * Beta(a, b) draws as G_a / (G_a + G_b) with independent gamma draws, formed from their logarithms (a sigmoid of the
 * difference) so that small a or b do not produce 0/0. Elementwise over broadcast a and b.
 */
export function beta<A extends Param, B extends Param, O extends SampleOptions = object>(
  s: Stream,
  a: A,
  b: B,
  options?: O,
): Drawn<[A, B], O> {
  return elementwise('beta', [a, b], options, (x, y) => betaDraw(s, x, y)) as Drawn<[A, B], O>
}

/** Chi-square draws with k > 0 degrees of freedom, 2 · Gamma(k/2, 1), elementwise over broadcast k. */
export function chiSquare<K extends Param, O extends SampleOptions = object>(
  s: Stream,
  df: K,
  options?: O,
): Drawn<[K], O> {
  return elementwise('chiSquare', [df], options, (k) => chiSquareDraw(s, k)) as Drawn<[K], O>
}

/**
 * Student t draws with ν > 0 degrees of freedom (ν = ∞ gives a normal), location and scale, Z / √(χ²_ν / ν),
 * elementwise over broadcast parameters.
 */
export function studentT<
  K extends Param,
  L extends Param = number,
  C extends Param = number,
  O extends SampleOptions = object,
>(s: Stream, df: K, loc?: L, scale?: C, options?: O): Drawn<[K, L, C], O> {
  return elementwise('studentT', [df, loc ?? 0, scale ?? 1], options, (nu, m, c) => {
    const z = normalDraw(s, 0, 1)
    if (nu === Infinity) return m + c * z
    return m + (c * z) / Math.sqrt(chiSquareDraw(s, nu) / nu)
  }) as Drawn<[K, L, C], O>
}

/**
 * Dirichlet(α) draws for concentrations α (length K, all > 0; or a batch of shape [..., K]), as a float64 tensor of
 * shape `[...shape, K]` on the simplex (`[K]` for one vector and no shape). Normalised gamma draws, formed in log space
 * so that small αₖ give tiny components rather than NaN.
 */
export function dirichlet(s: Stream, alpha: Tensor | ArrayLike<number>, options?: SampleOptions): Tensor {
  const { batch, k, values } = eventRows(alpha, 'dirichlet')
  const shape = options?.shape ?? batch
  checkBroadcast('dirichlet', batch, shape)
  const rows = batchIndex(batch, shape)
  const out = new Float64Array(rows.length * k)
  for (let r = 0; r < rows.length; r++) {
    const base = r * k
    let m = -Infinity
    for (let j = 0; j < k; j++) {
      out[base + j] = logGammaDraw(s, values[rows[r] * k + j])
      if (out[base + j] > m) m = out[base + j]
    }
    // Normalise by log-sum-exp (max-shifted), so tiny log-gamma values do not underflow to 0/0.
    let total = 0
    for (let j = 0; j < k; j++) total += Math.exp(out[base + j] - m)
    const z = m + Math.log(total)
    for (let j = 0; j < k; j++) out[base + j] = Math.exp(out[base + j] - z)
  }
  return fromData(out, [...shape, k])
}

// ── Discrete samplers ────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Bernoulli(p) draws: 1 with probability p, else 0 (a number, or float64 tensor), elementwise over broadcast p.
 */
export function bernoulli<P extends Param, O extends SampleOptions = object>(
  s: Stream,
  p: P,
  options?: O,
): Drawn<[P], O> {
  return elementwise('bernoulli', [p], options, (q) => (s.uniform() < q ? 1 : 0)) as Drawn<[P], O>
}

/**
 * Poisson(λ) draws (NaN for invalid λ), elementwise over broadcast λ. For λ < 10, inversion by sequential search (one
 * uniform); for λ ≥ 10, the transformed rejection method PTRS of Hörmann (1993), "The transformed rejection method for
 * generating Poisson random variables", Insurance: Mathematics and Economics 12.
 */
export function poisson<L extends Param, O extends SampleOptions = object>(
  s: Stream,
  lambda: L,
  options?: O,
): Drawn<[L], O> {
  return elementwise('poisson', [lambda], options, (l) => poissonDraw(s, l)) as Drawn<[L], O>
}

/**
 * Binomial(n, p) draws for integer n ≥ 0 (NaN for invalid parameters), elementwise over broadcast n and p. For
 * n·min(p, 1 − p) < 30, inversion by sequential search (Kachitvichyanukul and Schmeiser 1988, BINV). Otherwise the
 * order-statistic recursion (Knuth, TAOCP vol. 2, §3.4.1): the a-th smallest of n uniforms is X ~ Beta(a, n + 1 − a);
 * the count below p splits into a binomial below X or above it, halving n at each level, so a draw costs O(log n) beta
 * draws and is exact.
 */
export function binomial<N extends Param, P extends Param, O extends SampleOptions = object>(
  s: Stream,
  n: N,
  p: P,
  options?: O,
): Drawn<[N, P], O> {
  return elementwise('binomial', [n, p], options, (m, q) => binomialDraw(s, m, q)) as Drawn<[N, P], O>
}

/**
 * Multinomial(n, p) draws for integer n ≥ 0 and probabilities p (length K, normalised internally; or a batch of shape
 * [..., K]), as a float64 tensor of counts of shape `[...shape, K]`, each row summing to n. n may be a tensor, broadcast
 * with p's batch shape. Sequential conditional binomials: countₖ ~ Binomial(n − Σ_{j<k} countⱼ, pₖ / Σ_{j≥k} pⱼ).
 */
export function multinomial(s: Stream, n: Param, p: Tensor | ArrayLike<number>, options?: SampleOptions): Tensor {
  const { batch: pBatch, k, values } = eventRows(p, 'multinomial')
  const nShape = typeof n === 'number' ? [] : n.shape
  const joint = broadcastShapes(nShape, pBatch)
  const shape = options?.shape ?? joint
  checkBroadcast('multinomial', joint, shape)
  const rows = batchIndex(pBatch, shape)
  const ns = typeof n === 'number' ? null : toFlat(broadcastTo(n, shape))
  const out = new Float64Array(rows.length * k)
  for (let r = 0; r < rows.length; r++) {
    const base = r * k
    const offset = rows[r] * k
    let rest = 0
    for (let j = 0; j < k; j++) rest += values[offset + j]
    let left = ns ? ns[r] : (n as number)
    for (let j = 0; j < k && left > 0; j++) {
      if (j === k - 1) {
        out[base + j] = left
        break
      }
      const share = rest > 0 ? Math.min(1, values[offset + j] / rest) : 0
      out[base + j] = binomialDraw(s, left, share)
      left -= out[base + j]
      rest -= values[offset + j]
    }
  }
  return fromData(out, [...shape, k])
}

/**
 * Categorical draws: index k with probability wₖ / Σ w, for non-negative weights (unnormalised is fine) of length K,
 * or a batch of shape [..., K]. Returns a number for one weight vector and no shape, else an int32 tensor of indices
 * of the batch (or requested) shape. One uniform and a linear scan per draw; for many draws from the same weights use
 * {@link aliasTable} and {@link aliasSample}.
 */
export function categorical(s: Stream, weights: ArrayLike<number>): number
export function categorical(s: Stream, weights: Tensor | ArrayLike<number>, options?: SampleOptions): number | Tensor
export function categorical(s: Stream, weights: Tensor | ArrayLike<number>, options?: SampleOptions): number | Tensor {
  const { batch, k, values } = eventRows(weights, 'categorical')
  if (batch.length === 0 && options?.shape === undefined) return categoricalDraw(s, values, 0, k)
  const shape = options?.shape ?? batch
  checkBroadcast('categorical', batch, shape)
  const rows = batchIndex(batch, shape)
  const out = Int32Array.from(rows, (row) => categoricalDraw(s, values, row * k, k))
  return fromData(out, shape)
}

/** Walker's alias table for a categorical distribution (Vose's construction). */
export type AliasTable = {
  /** Probability of keeping column k rather than taking its alias, length K. */
  readonly probability: Float64Array
  /** The alias of column k, length K. */
  readonly alias: Int32Array
}

/**
 * Build Walker's alias table for non-negative weights (length K, an array or a rank-1 tensor), in O(K) (Vose 1991, "A
 * linear algorithm for generating random numbers with a given distribution", IEEE TSE 17(9)). Each
 * {@link aliasSample} is then O(1).
 */
export function aliasTable(weights: Tensor | ArrayLike<number>): AliasTable {
  const w = vectorValues(weights, 'aliasTable')
  const n = w.length
  let total = 0
  for (let k = 0; k < n; k++) total += w[k]
  if (!(total > 0) || !Number.isFinite(total))
    throw new RangeError('aliasTable needs finite weights with a positive sum')
  const scaled = new Float64Array(n)
  for (let k = 0; k < n; k++) scaled[k] = (w[k] * n) / total
  const probability = new Float64Array(n)
  const alias = new Int32Array(n)
  const small: number[] = []
  const large: number[] = []
  for (let k = n - 1; k >= 0; k--) (scaled[k] < 1 ? small : large).push(k)
  while (small.length && large.length) {
    const l = small.pop()!
    const g = large.pop()!
    probability[l] = scaled[l]
    alias[l] = g
    scaled[g] = scaled[g] + scaled[l] - 1
    ;(scaled[g] < 1 ? small : large).push(g)
  }
  // Leftovers are 1 up to rounding.
  for (const k of large) probability[k] = 1
  for (const k of small) probability[k] = 1
  for (let k = 0; k < n; k++) if (probability[k] === 1) alias[k] = k
  return { probability, alias }
}

/**
 * Categorical draws from an alias table: a uniform column, then keep it or take its alias. A number without
 * `shape`, else an int32 tensor of that shape.
 */
export function aliasSample(s: Stream, table: AliasTable): number
export function aliasSample(s: Stream, table: AliasTable, options: SampleOptions & { shape: readonly number[] }): Tensor
export function aliasSample(s: Stream, table: AliasTable, options?: SampleOptions): number | Tensor
export function aliasSample(s: Stream, table: AliasTable, options?: SampleOptions): number | Tensor {
  const one = () => {
    const k = s.int(table.probability.length)
    return s.uniform() < table.probability[k] ? k : table.alias[k]
  }
  if (options?.shape === undefined) return one()
  const n = options.shape.reduce((a, b) => a * b, 1)
  return fromData(Int32Array.from({ length: n }, one), options.shape)
}

/** Shuffle an array (or typed array) in place, uniformly over permutations (Fisher–Yates, Durstenfeld). Returns it. */
export function shuffle<A extends { length: number; [i: number]: unknown }>(s: Stream, array: A): A {
  for (let i = array.length - 1; i > 0; i--) {
    const j = s.int(i + 1)
    const tmp = array[i]
    array[i] = array[j]
    array[j] = tmp
  }
  return array
}

/** A uniformly random permutation of 0, …, n − 1, as an int32 tensor of shape [n]. */
export function permutation(s: Stream, n: number): Tensor {
  const out = new Int32Array(n)
  for (let i = 0; i < n; i++) out[i] = i
  return fromData(shuffle(s, out))
}

/** Options for {@link choice}. */
export type ChoiceOptions = {
  /** Sample with replacement (default true). */
  replace?: boolean
  /** Non-negative weights (length n, an array or a rank-1 tensor); uniform when omitted. */
  weights?: Tensor | ArrayLike<number>
}

/**
 * Indices drawn from {0, …, n − 1}, as an int32 tensor of shape `[size]` (or the given shape; filled in row-major
 * order). With replacement: uniform, or weighted through an alias table. Without replacement: a partial Fisher–Yates
 * shuffle, or for weights the exponential-keys method of Efraimidis and Spirakis (2006), "Weighted random sampling
 * with a reservoir", IPL 97(5) (keys log(u)/wᵢ, largest first), which returns indices in the order successive
 * weighted draws would pick them. Throws if more indices are asked for than there are available items.
 */
export function choice(s: Stream, n: number, size: number | readonly number[], options: ChoiceOptions = {}): Tensor {
  const { replace = true } = options
  const weights = options.weights === undefined ? undefined : vectorValues(options.weights, 'choice')
  const shape = typeof size === 'number' ? [size] : [...size]
  const count = shape.reduce((a, b) => a * b, 1)
  const out = new Int32Array(count)
  if (weights && weights.length !== n) throw new RangeError('choice: weights must have length n')
  if (replace) {
    if (count > 0 && n < 1) throw new RangeError('choice: nothing to choose from')
    if (!weights) for (let i = 0; i < count; i++) out[i] = s.int(n)
    else {
      const table = aliasTable(weights)
      for (let i = 0; i < count; i++) out[i] = aliasSample(s, table)
    }
    return fromData(out, shape)
  }
  if (!weights) {
    if (count > n) throw new RangeError('choice: size exceeds n without replacement')
    const pool = new Int32Array(n)
    for (let i = 0; i < n; i++) pool[i] = i
    for (let i = 0; i < count; i++) {
      const j = i + s.int(n - i)
      const tmp = pool[i]
      pool[i] = pool[j]
      pool[j] = tmp
      out[i] = pool[i]
    }
    return fromData(out, shape)
  }
  const keyed: { k: number; key: number }[] = []
  for (let k = 0; k < n; k++) {
    const u = s.uniform()
    if (weights[k] > 0) keyed.push({ k, key: Math.log1p(-u) / weights[k] })
  }
  if (count > keyed.length) throw new RangeError('choice: size exceeds the number of positive weights')
  keyed.sort((x, y) => y.key - x.key)
  for (let i = 0; i < count; i++) out[i] = keyed[i].k
  return fromData(out, shape)
}

// ── Multivariate normal ──────────────────────────────────────────────────────────────────────────────────────────────

/**
 * The spread of a multivariate normal: its covariance Σ (d × d, symmetric positive definite), or a lower-triangular
 * Cholesky factor L with Σ = L Lᵀ (entries above the diagonal are ignored).
 */
export type Spread = { covariance: Tensor } | { choleskyFactor: Tensor }

/**
 * Multivariate normal draws μ + L z with z ~ N(0, I), for a mean μ (length d, an array or tensor; or a batch of shape
 * [..., d]) and a covariance or Cholesky factor (d × d). Returns a float64 tensor of shape `[...shape, d]` (`[d]` for
 * one mean and no shape); each draw uses d successive standard normals.
 *
 * A covariance is factored with `aifn/linalg`'s `cholesky`, without jitter unless `options.jitter` allows it (the
 * `cholesky` options). A covariance that does not factor is an error rather than a silently regularised draw: pass
 * `jitter`, or a Cholesky factor of your own.
 */
export function multivariateNormal(
  s: Stream,
  mean: Tensor | ArrayLike<number>,
  spread: Spread,
  options: SampleOptions & { jitter?: CholeskyOptions['jitter'] } = {},
): Tensor {
  const { batch, k: d, values } = eventRows(mean, 'multivariateNormal')
  let factor: Tensor
  if ('covariance' in spread) {
    const c = cholesky(spread.covariance, { jitter: options.jitter ?? false })
    if (c.failed)
      throw new RangeError(
        `multivariateNormal: the covariance is not positive definite (pivot ${c.failedAt}); pass jitter or a Cholesky factor`,
      )
    factor = c.L
  } else factor = spread.choleskyFactor
  if (factor.shape.length !== 2 || factor.shape[0] !== d || factor.shape[1] !== d)
    throw new RangeError(`multivariateNormal: needs a ${d} × ${d} factor, got shape ${showShape(factor.shape)}`)
  const L = toFlat(factor)
  const shape = options.shape ?? batch
  checkBroadcast('multivariateNormal', batch, shape)
  const rows = batchIndex(batch, shape)
  const out = new Float64Array(rows.length * d)
  const z = new Float64Array(d)
  for (let r = 0; r < rows.length; r++) {
    for (let j = 0; j < d; j++) z[j] = normalDraw(s, 0, 1)
    for (let i = 0; i < d; i++) {
      let v = values[rows[r] * d + i]
      for (let j = 0; j <= i; j++) v += L[i * d + j] * z[j]
      out[r * d + i] = v
    }
  }
  return fromData(out, [...shape, d])
}
