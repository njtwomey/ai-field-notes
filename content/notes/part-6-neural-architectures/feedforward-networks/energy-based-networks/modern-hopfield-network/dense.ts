/**
 * Dense associative memories. The energy E(s) = −Σ_μ F(ξ^μ · s) with F(x) = x² is the classical Hopfield network (up
 * to a constant and a factor 1/N); F(x) = x^n with n > 2 is the dense associative memory of Krotov and Hopfield;
 * F(x) = exp(x) is the exponential memory of Demircigil et al. The continuous modern Hopfield network of Ramsauer et
 * al. updates a real vector in one step: ξ ← X softmax(β Xᵀ ξ).
 */

export type Uniform = () => number
export type Separation = { kind: 'power'; n: number } | { kind: 'exp' }

/** Separation function F. The exponential is shifted by N so large overlaps stay finite; a common factor cancels. */
function separation(f: Separation, N: number): (x: number) => number {
  if (f.kind === 'exp') return (x) => Math.exp(x - N)
  const n = f.n
  return (x) => Math.pow(x, n)
}

/**
 * Asynchronous updates until a sweep changes nothing: neuron i takes the sign that gives the lower energy, comparing
 * Σ_μ F(ξ^μ·s with s_i = +1) with Σ_μ F(ξ^μ·s with s_i = −1). Overlaps are kept up to date, so a sweep costs O(NK).
 */
export function denseRecall(
  patterns: ArrayLike<number>[],
  start: ArrayLike<number>,
  f: Separation,
  uniform: Uniform,
  maxSweeps = 30,
): Int8Array {
  const N = start.length
  const F = separation(f, N)
  const s = Int8Array.from(start)
  const a = patterns.map((xi) => {
    let m = 0
    for (let i = 0; i < N; i++) m += xi[i] * s[i]
    return m
  })
  const order = Array.from({ length: N }, (_, i) => i)
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(uniform() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    let changes = 0
    for (const i of order) {
      let diff = 0
      for (let mu = 0; mu < patterns.length; mu++) {
        const rest = a[mu] - patterns[mu][i] * s[i]
        diff += F(rest + patterns[mu][i]) - F(rest - patterns[mu][i])
      }
      if (diff === 0) continue
      const next = diff > 0 ? 1 : -1
      if (next === s[i]) continue
      for (let mu = 0; mu < patterns.length; mu++) a[mu] += 2 * next * patterns[mu][i]
      s[i] = next
      changes++
    }
    if (changes === 0) break
  }
  return s
}

/** Attention weights p = softmax(β Xᵀ ξ) over the stored patterns. */
export function attentionWeights(patterns: ArrayLike<number>[], query: ArrayLike<number>, beta: number): number[] {
  const scores = patterns.map((xi) => {
    let d = 0
    for (let i = 0; i < query.length; i++) d += xi[i] * query[i]
    return beta * d
  })
  const max = Math.max(...scores)
  const e = scores.map((x) => Math.exp(x - max))
  const z = e.reduce((s, x) => s + x, 0)
  return e.map((x) => x / z)
}

/** One update of the continuous modern Hopfield network: ξ_new = X p, with p the attention weights. */
export function softmaxUpdate(patterns: ArrayLike<number>[], query: ArrayLike<number>, beta: number): Float64Array {
  const p = attentionWeights(patterns, query, beta)
  const out = new Float64Array(query.length)
  patterns.forEach((xi, mu) => {
    for (let i = 0; i < out.length; i++) out[i] += p[mu] * xi[i]
  })
  return out
}
