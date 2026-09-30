/**
 * Attention: scaled dot-product attention with masks (Vaswani et al., 2017, §3.2.1), multi-head attention (§3.2.2)
 * and a transformer block (pre- or post-norm, §3.1; Xiong et al., 2020, for pre-norm). Attention weights are returned
 * and tapped, so figures can draw them.
 */

import { child, type Stream } from 'aifn/foundation/random'
import { softmax } from 'aifn/numerics/special'
import {
  fromData,
  matmul,
  mul,
  permute,
  reshape,
  shapeOfValue,
  transpose,
  where,
  type Tensor,
  type Value,
} from 'aifn/foundation/tensor'
import { xavierUniform, zerosInit } from 'aifn/nn/init'
import { Linear, linear, tap, type Layer, type LinearParams } from './layers'

/** A lower-triangular mask of ones, [T, T] (or [Tq, Tk] aligned at the end): position i sees positions ≤ i. */
export function causalMask(tq: number, tk: number = tq): Tensor {
  const out = new Float64Array(tq * tk)
  const shift = tk - tq
  for (let i = 0; i < tq; i++) for (let j = 0; j <= Math.min(tk - 1, i + shift); j++) out[i * tk + j] = 1
  return fromData(out, [tq, tk])
}

/** Options of `scaledDotProductAttention`. */
export type AttentionOptions = {
  /** 1 where a query may attend to a key, 0 where not; broadcast to [..., Tq, Tk]. */
  mask?: Tensor
  /** Apply the causal mask (query i sees keys ≤ i). */
  causal?: boolean
  /** The scale of the scores (default 1/√d_k). */
  scale?: number
  /** Called with the weights before they multiply V; returns the weights to use (for recording and probing). */
  tapWeights?: (weights: Value) => Value
}

/** The output of attention and the weights that produced it. */
export type AttentionResult = {
  /** softmax(QKᵀ·scale)·V, shape [..., Tq, d_v]. */
  output: Value
  /** The attention weights, shape [..., Tq, Tk], each row summing to 1 over the keys it may see. */
  weights: Value
}

/**
 * Scaled dot-product attention softmax(QKᵀ/√d_k)·V for queries [..., Tq, d_k], keys [..., Tk, d_k] and values
 * [..., Tk, d_v] (leading axes broadcast). Masked scores are set to −∞ before the softmax, so they get weight 0 and no
 * gradient. A query that may see no key gets NaN weights.
 */
export function scaledDotProductAttention(
  q: Value,
  k: Value,
  v: Value,
  options: AttentionOptions = {},
): AttentionResult {
  const qs = shapeOfValue(q)
  const ks = shapeOfValue(k)
  const dk = qs[qs.length - 1]
  const scale = options.scale ?? 1 / Math.sqrt(dk)
  const rank = ks.length
  const axes = ks.map((_, i) => i)
  ;[axes[rank - 2], axes[rank - 1]] = [axes[rank - 1], axes[rank - 2]]
  let scores = mul(matmul(q, transpose(k, axes)), scale)
  const tq = qs[qs.length - 2]
  const tk = ks[ks.length - 2]
  if (options.causal) scores = where(causalMask(tq, tk), scores, -Infinity)
  if (options.mask) scores = where(options.mask, scores, -Infinity)
  const computed = softmax(scores, { axis: -1 })
  const weights = options.tapWeights ? options.tapWeights(computed) : computed
  return { output: matmul(weights, v), weights }
}

/** Parameters of `MultiHeadAttention`: the query, key, value and output projections. */
export type MultiHeadAttentionParams = {
  query: LinearParams
  key: LinearParams
  value: LinearParams
  output: LinearParams
}

/** Options of `multiHeadAttention` and `MultiHeadAttention`. */
export type MultiHeadOptions = {
  heads: number
  causal?: boolean
  mask?: Tensor
  /** See `AttentionOptions.tapWeights`. */
  tapWeights?: (weights: Value) => Value
}

/** [..., T, h·d] → [..., h, T, d]. */
function splitHeads(x: Value, heads: number): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -2)
  const [t, d] = s.slice(-2)
  const r = lead.length
  const split = reshape(x, [...lead, t, heads, d / heads])
  return permute(split, [...lead.map((_, i) => i), r + 1, r, r + 2])
}

/** [..., h, T, d] → [..., T, h·d]. */
function mergeHeads(x: Value): Value {
  const s = shapeOfValue(x)
  const lead = s.slice(0, -3)
  const [h, t, d] = s.slice(-3)
  const r = lead.length
  return reshape(permute(x, [...lead.map((_, i) => i), r + 1, r, r + 2]), [...lead, t, h * d])
}

/**
 * Multi-head attention (Vaswani et al., 2017, §3.2.2): queries from `xq` [..., Tq, d_model] and keys and values from
 * `xkv` [..., Tk, d_model] (the same tensor for self-attention), projected into h heads of d_model/h, attended
 * separately, concatenated and projected back. Returns the output [..., Tq, d_model] and the weights [..., h, Tq, Tk].
 */
export function multiHeadAttention(
  params: MultiHeadAttentionParams,
  xq: Value,
  xkv: Value,
  { heads, causal, mask, tapWeights }: MultiHeadOptions,
): AttentionResult {
  const q = splitHeads(linear(xq, params.query.weight, params.query.bias), heads)
  const k = splitHeads(linear(xkv, params.key.weight, params.key.bias), heads)
  const v = splitHeads(linear(xkv, params.value.weight, params.value.bias), heads)
  const { output, weights } = scaledDotProductAttention(q, k, v, { causal, mask, tapWeights })
  return { output: linear(mergeHeads(output), params.output.weight, params.output.bias), weights }
}

/**
 * Multi-head self-attention as a layer over [..., T, d_model]. The weights are tapped at `<path>.weights` and the
 * output at `<path>`.
 */
export function MultiHeadAttention(dModel: number, options: MultiHeadOptions): Layer<MultiHeadAttentionParams> {
  if (dModel % options.heads !== 0)
    throw new Error(`MultiHeadAttention: ${options.heads} heads do not divide ${dModel}`)
  const projection = Linear(dModel, dModel, { init: xavierUniform(), biasInit: zerosInit() })
  return {
    kind: 'MultiHeadAttention',
    label: `MultiHeadAttention(${dModel}, ${options.heads} heads${options.causal ? ', causal' : ''})`,
    init: (s: Stream) => ({
      query: projection.init(child(s, 'query')),
      key: projection.init(child(s, 'key')),
      value: projection.init(child(s, 'value')),
      output: projection.init(child(s, 'output')),
    }),
    apply: (p, x, ctx) => {
      const { output } = multiHeadAttention(p, x, x, { ...options, tapWeights: (w) => tap(ctx, w, 'weights') })
      return tap(ctx, output)
    },
  }
}
