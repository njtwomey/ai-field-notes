import type { Tensor } from 'aifn-compute/foundation/tensor'

/**
 * Attention weights over a token sequence, as returned by `aifn/nn/attention` (`multiHeadAttention`'s `weights`), with
 * the tokens they are over: the object the attention view draws.
 */
export type AttentionPattern = {
  readonly kind: 'attention'
  /** Key tokens (the columns). */
  readonly keys: readonly string[]
  /** Query tokens (the rows); default the keys (self-attention). */
  readonly queries?: readonly string[]
  /** Weights [h, Tq, Tk] or [Tq, Tk]; each row sums to one over the keys it may see. */
  readonly weights: Tensor
  /** The mask [Tq, Tk]: 1 where a query may attend, 0 where not (drawn as empty, crossed cells). */
  readonly mask?: Tensor | null
  /** The first `cached` keys come from a key–value cache (marked in the token strip). */
  readonly cached?: number
}

/** An attention pattern for the view. */
export function attentionPattern(p: Omit<AttentionPattern, 'kind'>): AttentionPattern {
  return { kind: 'attention', ...p }
}
