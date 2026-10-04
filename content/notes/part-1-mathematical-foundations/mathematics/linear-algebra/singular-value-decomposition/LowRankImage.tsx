import { useMemo } from 'react'
import { Bars, Figure, Handle, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import type { ImageSvd } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'

/** Rank-k reconstructions of a test image from its SVD, precomputed by python/mlc/figures/svd.py. */
export function LowRankImage() {
  const { data, error } = useFigure<ImageSvd>('singular-value-decomposition/image')
  // The test image is 64 × 64, so its rank is at most 64.
  const state = useFigureState({
    k: int(4, { min: 1, max: 64, step: 1, suggestions: [1, 4, 16, 64], label: 'rank k' }),
  })
  const size = data?.s.length ?? 64
  const k = Math.min(state.k, size)
  const xAxis = useAxis({})
  const yAxis = useAxis({})
  const xAxis2 = useAxis({})
  const yAxis2 = useAxis({})
  const xAxis3 = useAxis({ label: 'i', hold: 'union' })
  const yAxis3 = useAxis({ label: 'σᵢ', hold: 'union', log: true })

  const r = useMemo(() => {
    if (!data) return undefined
    const size = data.s.length
    // A_k = Σ_{i<k} σᵢ uᵢ vᵢᵀ; rows are reversed so the first image row is drawn at the top.
    const approx = Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, col) => {
        let v = 0
        for (let i = 0; i < k; i++) v += data.s[i] * data.u[row][i] * data.vt[i][col]
        return v
      }),
    ).reverse()
    const total = data.s.reduce((acc, s) => acc + s * s, 0)
    const kept = data.s.slice(0, k).reduce((acc, s) => acc + s * s, 0)
    const axis = Array.from({ length: size }, (_, i) => i)
    const spectrum = [
      { name: 'kept', x: axis.slice(0, k).map((i) => i + 1), y: data.s.slice(0, k), slot: 0 },
      { name: 'discarded', x: axis.slice(k).map((i) => i + 1), y: data.s.slice(k), muted: true },
    ] as const
    return {
      size,
      axis,
      approx,
      original: [...data.image].reverse(),
      energy: kept / total,
      error: Math.sqrt((total - kept) / total),
      storage: (k * (2 * size + 1)) / (size * size),
      spectrum,
    }
  }, [data, k])

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!r) return null
  // The cut sits between bar k (kept) and bar k + 1 (discarded); dragging it picks the nearest such gap.
  return (
    <Figure
      title="Keeping the k largest singular values"
      caption="The rank-k approximation keeps the first k terms σᵢuᵢvᵢᵀ. The smooth gradient, the disc and the bar appear within a few terms; the diagonal band and the checkerboard need many, because they are not aligned with rows and columns. No other rank-k matrix is closer to the image. Drag the dashed cut on the spectrum to choose k."
      state={state}
      readouts={
        <>
          <Readout label="energy kept Σσᵢ² (i ≤ k) / Σσᵢ²" value={`${(100 * r.energy).toFixed(1)}%`} />
          <Readout label="relative Frobenius error" value={`${(100 * r.error).toFixed(1)}%`} />
          <Readout label="numbers stored vs full image" value={`${(100 * r.storage).toFixed(0)}%`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-3">
        <div>
          <p className="mb-1 text-center text-sm font-medium">Original</p>
          <Plot x={xAxis} y={yAxis} height={260}>
            <Raster x={r.axis} y={r.axis} z={r.original} range={[0, 1]} valueLabel={'intensity'} />
          </Plot>
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Rank {k}</p>
          <Plot x={xAxis2} y={yAxis2} height={260}>
            <Raster x={r.axis} y={r.axis} z={r.approx} range={[0, 1]} valueLabel={'intensity'} />
          </Plot>
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Singular values</p>
          <Plot x={xAxis3} y={yAxis3} height={260}>
            <Bars {...r.spectrum[0]} />
            <Bars {...r.spectrum[1]} />
            <Handle
              kind="x"
              at={k + 0.5}
              label="k"
              onDrag={(x) => state.set('k', Math.min(Math.max(Math.round(x - 0.5), 1), r.size))}
            />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
