import { toFlat } from 'aifn-compute/foundation/tensor'
import type { Histogram } from 'aifn-compute/probability/stats'

/** A histogram as plain arrays for drawing: bar centres, counts and densities, and the edges. */
export type HistogramBars = { x: number[]; counts: number[]; density: number[]; edges: number[] }

/** The bars of an `aifn/probability/stats` histogram: one per bin, placed at the bin's centre. */
export function histogramBars(h: Histogram): HistogramBars {
  const edges = toFlat(h.edges)
  return {
    x: edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2),
    counts: toFlat(h.counts),
    density: toFlat(h.density),
    edges,
  }
}
