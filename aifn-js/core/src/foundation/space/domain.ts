/**
 * Domains of observations and actions (`Domain`, docs/aifn-environments.md §3): a constructor and the helpers every
 * agent and rollout needs, `domainContains`, `domainSize` and a uniform draw `sampleDomain`.
 */

import { DomainError } from 'aifn/foundation/errors'
import type { Domain, Size } from 'aifn/foundation/contracts'
import { integers, type Stream } from 'aifn/foundation/random'

/** The discrete domain 0 … n − 1, with optional display names (one per value). */
export function discreteDomain(n: Size, names?: readonly string[]): Domain {
  if (!(Number.isInteger(n) && n >= 1))
    throw new DomainError('discreteDomain', `discreteDomain: n must be a positive integer, got ${n}`)
  if (names && names.length !== n)
    throw new DomainError('discreteDomain', `discreteDomain: ${names.length} names for ${n} values`)
  return names ? { kind: 'discrete', n, names } : { kind: 'discrete', n }
}

/** True when `x` is a value of the domain. */
export function domainContains(domain: Domain, x: unknown): boolean {
  return typeof x === 'number' && Number.isInteger(x) && x >= 0 && x < domain.n
}

/** The number of values of a discrete domain. */
export function domainSize(domain: Domain): Size {
  return domain.n
}

/** A uniform draw from the domain. */
export function sampleDomain(s: Stream, domain: Domain): number {
  return integers(s, domain.n)
}
