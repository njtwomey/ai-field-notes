/**
 * What the two "model built up step by step" showcases share (sparse GPs grown inducing point by inducing point, the
 * RVM basis by basis): the predictive band, a played position that restarts with each new run, and the axes models.
 */
import { useState } from 'react'
import { formatValue } from '@lab/views'

export const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : String(v))

/** The mean and the ± 2 sd edges of a Gaussian prediction. */
export function band(mean: ArrayLike<number>, variance: ArrayLike<number>) {
  const m = Array.from(mean)
  const sd = Array.from(variance, (v) => Math.sqrt(Math.max(v, 0)))
  return { mean: m, upper: m.map((v, i) => v + 2 * sd[i]), lower: m.map((v, i) => v - 2 * sd[i]) }
}

/**
 * The played step of a run: it starts at 0 whenever `key` (the run's settings) changes, and is clamped to the run's
 * length while a newer run is on its way.
 */
export function usePlayed(key: string, count: number): [number, (p: number) => void] {
  const [played, setPlayed] = useState({ key, pos: 0 })
  const pos = played.key === key ? played.pos : 0
  return [Math.max(0, Math.min(pos, count - 1)), (p: number) => setPlayed({ key, pos: p })]
}
