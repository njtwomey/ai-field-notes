/** The bandit environment protocol, part of `aifn-applied/decisions/bandits`: named environments are in `aifn-applied/data/environments`. */

import { type Stream } from 'aifn/foundation/random'

/** A bandit environment: its arms, their contexts and reward draws. */
export interface BanditEnvironment {
  name: string
  /** Number of arms. */
  arms: number
  /** Dimension of the arms' feature vectors (0 for a context-free bandit). */
  dim: number
  /** The arms' feature vectors in a round (arms × dim), or null for a context-free bandit. */
  context(s: Stream): Float64Array[] | null
  /** The reward every arm would pay in a round; only the pulled arm's is revealed to the policy. */
  rewards(s: Stream, context: Float64Array[] | null): Float64Array
  /** The expected reward of every arm (given the round's context). */
  means(context: Float64Array[] | null): Float64Array
  /** Rewards are known to lie in [0, 1] (needed by EXP3 and KL-UCB). */
  bounded: boolean
}
