import type { Stream } from 'aifn/random'
import type { Vector, VectorLike } from 'aifn/tensor'

export type { VectorLike } from 'aifn/tensor'

/** The target of a sampler: an unnormalised log-density, defined in `aifn/distributions` (re-exported here). */
export type { Target } from 'aifn/distributions'

/** The start of a chain. */
export type ChainStart = { x0: VectorLike }

/**
 * Fields every Metropolis-type sampler state carries. `x` is the current point and `logDensity` log π(x). Step t draws
 * from `stream.child(t)`, so a step is a pure function of the state.
 */
export type ChainState = {
  /** Steps taken (0 in the initial state). */
  t: number
  x: Vector
  logDensity: number
  /** The stream all draws come from. */
  stream: Stream
  /** Set when log π(x) is NaN or +Infinity, or x is not finite; stops the runners. */
  diverged: boolean
}

/** Fields of a sampler with an accept/reject step. */
export type AcceptRejectState = ChainState & {
  /** The point proposed on the last step and log π there (the start point at t = 0). */
  proposal: Vector
  proposalLogDensity: number
  /** The log Metropolis–Hastings ratio of the last proposal; the acceptance probability is min(1, exp of it). */
  logAcceptanceRatio: number
  /** min(1, exp(logAcceptanceRatio)): the acceptance probability of the last proposal (NaN at t = 0). */
  acceptance: number
  accepted: boolean
  /** Accepted proposals so far and their fraction of steps (NaN at t = 0). */
  acceptedCount: number
  acceptanceRate: number
}
