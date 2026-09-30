import { add, type Value } from 'aifn/foundation/tensor'
import { activationFn, type Activation } from 'aifn/nn/functional'
import {
  childContext,
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
} from 'aifn/nn/layers'
import { type MultiHeadAttentionParams, type MultiHeadOptions, MultiHeadAttention } from 'aifn/nn/layers'
import { child } from 'aifn/foundation/random'

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
      attentionNorm: normLayer.init(child(s, 'attentionNorm')),
      attention: attention.init(child(s, 'attention')),
      feedForwardNorm: normLayer.init(child(s, 'feedForwardNorm')),
      feedForward: { up: up.init(child(s, 'up')), down: down.init(child(s, 'down')) },
    }),
    apply: (params, x, ctx) => {
      const branch = (name: string, f: (h: Value, c?: Context) => Value, np: NormParams, h: Value) => {
        const c = childContext(ctx, name)
        if (placement === 'pre')
          return add(h, drop.apply({}, f(normalise(np, h), c), childContext(ctx, `${name}Dropout`)))
        return normalise(np, add(h, drop.apply({}, f(h, c), childContext(ctx, `${name}Dropout`))))
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
