import { useMemo, useState } from 'react'
import { DEFAULT_DATA, type DataSpec } from './ordinal'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

export type Resolution = 'low' | 'medium' | 'high'

/** What a filled map shows: the model's decision (its predicted class) or its expected class E[y | x]. */
export type FillMode = 'decision' | 'expectation'

/** Cells per side of a filled map over the dataset's range. Fitted models are cached apart from the grid, so this never refits. */
export const CELLS: Record<Resolution, number> = { low: 40, medium: 80, high: 160 }

/**
 * The one source of 2-D ordinal data for every widget in the category: a spec (curve, classes, noise, seed, training
 * size) that starts from the shared defaults, with the display settings of any filled map. The data itself comes with
 * the fit (see useFit), so that points and fill always belong together.
 */
export function useOrdinalData(initial: Partial<DataSpec> = {}) {
  const [spec, setSpec] = useState<DataSpec>({ ...DEFAULT_DATA, ...initial })
  const [resolution, setResolution] = useState<Resolution>('medium')
  const [fill, setFill] = useState<FillMode>('expectation')
  return { spec, setSpec, resolution, setResolution, fill, setFill }
}

/** The cells of a filled map over a dataset's square range at a resolution. */
export function useGrid(range: [number, number], resolution: Resolution): number[] {
  const [lo, hi] = range
  return useMemo(() => toFlat(linspace(lo, hi, CELLS[resolution])), [lo, hi, resolution])
}
