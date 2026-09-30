import { useMemo, type ReactNode } from 'react'
import { classProbabilities, hasPredictive, type Distribution } from 'aifn/estimators'
import { grid2d } from 'aifn/geometry'
import { fromData, toFlat, type Tensor } from 'aifn/tensor'
import { Figure } from '@lab/layout'
import { Heatmap, Readout, type HeatmapOverlay, type Vec2 } from '@lab/viz'
import type { FrameProps } from './frame'
import { formatValue } from './format'

export type DecisionRegionViewProps = FrameProps & {
  /** A fitted classifier on two features: anything with `decide(x [m, 2]) → labels [m]`. */
  model: { decide(x: Tensor): Tensor }
  /** Training data: inputs [n, 2] and labels [n]. */
  data: { x: Tensor; y: Tensor }
  /** Class names for the legend, tooltip and readouts. */
  classNames?: readonly string[]
  /** The plotted ranges (default: the data's, padded by 10%). */
  xRange?: [number, number]
  yRange?: [number, number]
  /** Grid cells per axis (default 90). */
  resolution?: number
  /** A query point, drawn as a handle; dragging it calls `onQuery`. */
  query?: Vec2
  onQuery?: (p: Vec2) => void
  /** Extra marks over the regions, e.g. the query's neighbours or the support vectors. */
  overlay?: readonly HeatmapOverlay[]
  /** Readouts about the query supplied by the specimen (shown before the generic ones). */
  queryReadouts?: ReactNode
  xLabel?: string
  yLabel?: string
}

function padded(v: number[], pad = 0.1): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const u of v) {
    lo = Math.min(lo, u)
    hi = Math.max(hi, u)
  }
  const p = pad * (hi - lo || 1)
  return [lo - p, hi + p]
}

/**
 * A 2-D classifier's decision regions: `decide` evaluated on a grid (one colour per class), the training points by
 * class, and an optional draggable query point with its prediction (and class probabilities when the model has a
 * predictive).
 */
export function DecisionRegionView({
  model,
  data,
  classNames,
  xRange,
  yRange,
  resolution = 90,
  query,
  onQuery,
  overlay,
  queryReadouts,
  xLabel = 'x₀',
  yLabel = 'x₁',
  title,
  controls,
  readouts,
  ...frame
}: DecisionRegionViewProps) {
  const cols = useMemo(() => {
    const flat = toFlat(data.x)
    return { x0: flat.filter((_, i) => i % 2 === 0), x1: flat.filter((_, i) => i % 2 === 1) }
  }, [data.x])
  const xr = useMemo(() => xRange ?? padded(cols.x0), [xRange, cols])
  const yr = useMemo(() => yRange ?? padded(cols.x1), [yRange, cols])
  const labels = useMemo(() => toFlat(data.y), [data.y])
  const K = useMemo(() => Math.max(2, ...labels.map((l) => l + 1)), [labels])
  const names = classNames ?? Array.from({ length: K }, (_, k) => `class ${k}`)
  const grid = useMemo(() => {
    const g = grid2d(xr, yr, resolution)
    const d = toFlat(model.decide(g.points))
    const [ny, nx] = g.shape
    const z = Array.from({ length: ny }, (_, i) => d.slice(i * nx, (i + 1) * nx))
    return { x: toFlat(g.x), y: toFlat(g.y), z }
  }, [model, xr, yr, resolution])
  const marks = useMemo(
    (): HeatmapOverlay[] => [
      { name: 'training rows', type: 'scatter', x: cols.x0, y: cols.x1, group: labels, groupNames: names },
      ...(overlay ?? []),
    ],
    [cols, labels, names, overlay],
  )
  const prediction = useMemo(() => {
    if (!query) return null
    const q = fromData(Float64Array.from(query), [1, 2])
    const label = toFlat(model.decide(q))[0]
    let probs: number[] | null = null
    if (hasPredictive(model)) {
      const d = (model as unknown as { predictive(x: Tensor): Distribution }).predictive(q)
      probs = Array.from(classProbabilities(d).probs)
    }
    return { label, probs }
  }, [model, query])
  return (
    <Figure
      title={title ?? 'Decision regions'}
      defaultSize="L"
      {...frame}
      controls={controls}
      readouts={
        <>
          {queryReadouts}
          {prediction && <Readout label="decide(query)" value={names[prediction.label] ?? String(prediction.label)} />}
          {prediction?.probs?.map((p, k) => (
            <Readout key={k} label={`P(${names[k] ?? k} | query)`} value={formatValue(p)} />
          ))}
          {readouts}
        </>
      }
    >
      <Heatmap
        x={grid.x}
        y={grid.y}
        z={grid.z}
        scale="categorical"
        categoryNames={names}
        fillOpacity={0.35}
        overlay={marks}
        equalAspect
        xLabel={xLabel}
        yLabel={yLabel}
        valueLabel="decision"
        handles={query && onQuery ? [{ kind: 'point', at: query, onDrag: onQuery, label: 'query' }] : undefined}
        marker={query}
      />
    </Figure>
  )
}
