import type { Univariate } from 'aifn-compute/probability/distributions'
import { toFlat, unwrap, type Value } from 'aifn-compute/foundation/tensor'

const numbers = (v: Value): number[] => {
  const r = unwrap(v)
  return typeof r === 'number' ? [r] : toFlat(r)
}

/** The x range a univariate view shows by default: its 0.002–0.998 quantiles, or its moments, padded. */
export function distributionRange(d: Univariate): [number, number] {
  const ends = (() => {
    try {
      const lo = Math.min(...numbers(d.quantile(0.002)))
      const hi = Math.max(...numbers(d.quantile(0.998)))
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi] as [number, number]
    } catch {
      // No quantile (or it failed): fall back to the moments.
    }
    try {
      const m = numbers(d.mean())
      const s = numbers(d.stddev())
      const lo = Math.min(...m.map((v, i) => v - 5 * s[i]))
      const hi = Math.max(...m.map((v, i) => v + 5 * s[i]))
      if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return [lo, hi] as [number, number]
    } catch {
      // No moments either.
    }
    return [-5, 5] as [number, number]
  })()
  if (!d.discrete) {
    const pad = 0.05 * (ends[1] - ends[0])
    return [ends[0] - pad, ends[1] + pad]
  }
  return [Math.floor(ends[0]) - 1, Math.ceil(ends[1]) + 1]
}
