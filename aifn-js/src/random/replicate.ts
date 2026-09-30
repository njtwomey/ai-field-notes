/**
 * Replicates: run a stochastic function on child streams 0, 1, …, n − 1 of a stream. Replicate k always uses
 * `s.child(k)`, so its result depends only on (s.key, k): raising n keeps the first replicates (prefix reuse), and
 * running two methods over the same replicates gives common random numbers.
 */

import type { Stream } from './stream'

/** A cache of replicate results keyed by the child stream's key. */
export type ReplicateCache<T> = Map<string, T>

// Default caches, one per function object. A new function (e.g. a closure over new parameters) gets a fresh cache.
const caches = new WeakMap<object, ReplicateCache<unknown>>()

/**
 * Run `fn(s.child(k), k)` for k = 0, …, n − 1 and return the results in order. Results are cached by the child's key:
 * by default in a cache attached to `fn` itself, so calling again with the same function and a larger n computes only
 * the new replicates. `fn` must therefore be a pure function of its stream and index; when its inputs change, pass a
 * new function (as a React `useCallback` with dependencies does) or an explicit `cache`. `cache: false` disables caching.
 */
export function replicate<T>(
  n: number,
  s: Stream,
  fn: (s: Stream, k: number) => T,
  options: { cache?: ReplicateCache<T> | false } = {},
): T[] {
  let cache: ReplicateCache<T> | undefined
  if (options.cache === false) cache = undefined
  else if (options.cache) cache = options.cache
  else {
    cache = caches.get(fn) as ReplicateCache<T> | undefined
    if (!cache) {
      cache = new Map()
      caches.set(fn, cache as ReplicateCache<unknown>)
    }
  }
  const out: T[] = []
  for (let k = 0; k < n; k++) {
    const child = s.child(k)
    if (cache?.has(child.key)) {
      out.push(cache.get(child.key) as T)
      continue
    }
    const value = fn(child, k)
    cache?.set(child.key, value)
    out.push(value)
  }
  return out
}
