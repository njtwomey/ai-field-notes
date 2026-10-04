/** The seeded random source the bandit figures pass to `bandits.ts`: an aifn stream behind the `Rand` interface. */
import { normal, stream, uniform } from 'aifn/foundation/random'
import type { Rand } from './bandits'

/** A `Rand` drawing from `stream(seed)`: the same seed gives the same draws on every device. */
export function seededRand(seed: number): Rand {
  const s = stream(seed)
  return { uniform: () => uniform(s), normal: () => normal(s) }
}
