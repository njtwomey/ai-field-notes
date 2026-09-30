/**
 * Attention: scaled dot-product attention with masks (Vaswani et al., 2017, §3.2.1), multi-head attention (§3.2.2)
 * and a transformer block (pre- or post-norm, §3.1; Xiong et al., 2020, for pre-norm). Attention weights are returned
 * and tapped, so figures can draw them.
 */

import type { Stream } from 'aifn/random'
import { softmax } from 'aifn/special'
import {
  add,
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
} from 'aifn/tensor'
import { activationFn, type Activation } from './activations'
import { xavierUniform, zerosInit } from './init'
import {
  child,
  Dropout,
  layerNorm,
  LayerNorm,
  Linear,
  linear,
  rmsNorm,
  tap,
  type Context,
  type Layer,
  type LinearParams,
  type NormParams,
} from './layers'

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
      query: projection.init(s.child('query')),
      key: projection.init(s.child('key')),
      value: projection.init(s.child('value')),
      output: projection.init(s.child('output')),
    }),
    apply: (p, x, ctx) => {
      const { output } = multiHeadAttention(p, x, x, { ...options, tapWeights: (w) => tap(ctx, w, 'weights') })
      return tap(ctx, output)
    },
  }
}

/** Parameters of `TransformerBlock`. */
export type TransformerBlockParams = {
  attentionNorm: NormParams
  attention: MultiHeadAttentionParams
  feedForwardNorm: NormParams
  feedForward: { up: LinearParams; down: LinearParams }
}

/** Options of `TransformerBlock`. */
export type TransformerBlockOptions = MultiHeadOptions & {
  /** Width of the feed-forward layer (default 4·d_model). */
  hidden?: number
  /** `pre` (default): x + f(norm(x)), as modern decoders; `post`: norm(x + f(x)), as the original transformer. */
  placement?: 'pre' | 'post'
  /** `layer` (default) or `rms` normalisation. */
  norm?: 'layer' | 'rms'
  activation?: Activation
  /** Dropout on each residual branch while training (default 0). */
  dropout?: number
}

/**
 * A transformer block over [..., T, d_model]: multi-head self-attention and a two-layer feed-forward network, each in a
 * residual branch with normalisation before (`pre`) or after (`post`) it.
 */
export function TransformerBlock(dModel: number, options: TransformerBlockOptions): Layer<TransformerBlockParams> {
  const { hidden = 4 * dModel, placement = 'pre', norm = 'layer', activation = 'gelu', dropout: p = 0 } = options
  const attention = MultiHeadAttention(dModel, options)
  const up = Linear(dModel, hidden)
  const down = Linear(hidden, dModel)
  const normLayer = LayerNorm(dModel)
  const drop = Dropout(p)
  const act = activationFn(activation)
  const normalise = (np: NormParams, x: Value) =>
    norm === 'rms' ? rmsNorm(x, np.gamma) : layerNorm(x, np.gamma, np.beta)
  return {
    kind: 'TransformerBlock',
    label: `TransformerBlock(${dModel}, ${options.heads} heads, ${placement}-norm)`,
    init: (s) => ({
      attentionNorm: normLayer.init(s.child('attentionNorm')),
      attention: attention.init(s.child('attention')),
      feedForwardNorm: normLayer.init(s.child('feedForwardNorm')),
      feedForward: { up: up.init(s.child('up')), down: down.init(s.child('down')) },
    }),
    apply: (params, x, ctx) => {
      const branch = (name: string, f: (h: Value, c?: Context) => Value, np: NormParams, h: Value) => {
        const c = child(ctx, name)
        if (placement === 'pre') return add(h, drop.apply({}, f(normalise(np, h), c), child(ctx, `${name}Dropout`)))
        return normalise(np, add(h, drop.apply({}, f(h, c), child(ctx, `${name}Dropout`))))
      }
      const h = branch('attention', (u, c) => attention.apply(params.attention, u, c), params.attentionNorm, x)
      const ff = (u: Value, c?: Context) =>
        tap(
          c,
          linear(
            act(linear(u, params.feedForward.up.weight, params.feedForward.up.bias)),
            params.feedForward.down.weight,
            params.feedForward.down.bias,
          ),
        )
      return tap(ctx, branch('feedForward', ff, params.feedForwardNorm, h))
    },
  }
}
