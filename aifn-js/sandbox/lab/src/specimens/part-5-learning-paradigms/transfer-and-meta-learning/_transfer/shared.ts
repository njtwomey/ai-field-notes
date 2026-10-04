/** Helpers shared by the transfer figures: number formatting and the picked checkpoint of a trained run. */
import { useState } from 'react'
import { formatNumber } from '@lab/viz'

export const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** The checkpoint of a run picked on it; a new run opens at checkpoint 0. */
export function usePicked<S>(trained: S | null, count: number) {
  const [picked, setPicked] = useState<{ run: S | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained ? picked.index : 0, Math.max(0, count - 1))
  return [index, (i: number) => setPicked({ run: trained, index: i })] as const
}
