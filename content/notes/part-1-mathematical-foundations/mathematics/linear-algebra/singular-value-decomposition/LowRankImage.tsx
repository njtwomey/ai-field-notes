import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, XYChart, type Handle, type XYSeries } from 'aifn-render'
import type { ImageSvd } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'

/** Rank-k reconstructions of a test image from its SVD, precomputed by python/mlc/figures/svd.py. */
export function LowRankImage() {
  const { data, error } = useFigure<ImageSvd>('singular-value-decomposition/image')
  const [k, setK] = useState(4)

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
    const spectrum: XYSeries[] = [
      { name: 'kept', type: 'bar', x: axis.slice(0, k).map((i) => i + 1), y: data.s.slice(0, k), slot: 0 },
      { name: 'discarded', type: 'bar', x: axis.slice(k).map((i) => i + 1), y: data.s.slice(k), muted: true },
    ]
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
  const cut: Handle[] = [
    { kind: 'x', at: k + 0.5, label: 'k', onDrag: (x) => setK(Math.min(Math.max(Math.round(x - 0.5), 1), r.size)) },
  ]
  return (
    <Interactive
      title="Keeping the k largest singular values"
      caption="The rank-k approximation keeps the first k terms σᵢuᵢvᵢᵀ. The smooth gradient, the disc and the bar appear within a few terms; the diagonal band and the checkerboard need many, because they are not aligned with rows and columns. No other rank-k matrix is closer to the image. Drag the dashed cut on the spectrum to choose k."
      controls={
        <ParamSlider label="rank k" value={k} onChange={setK} min={1} max={r.size} step={1} format={(v) => String(v)} />
      }
      readout={
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
          <Heatmap x={r.axis} y={r.axis} z={r.original} range={[0, 1]} valueLabel="intensity" height={260} />
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Rank {k}</p>
          <Heatmap x={r.axis} y={r.axis} z={r.approx} range={[0, 1]} valueLabel="intensity" height={260} />
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Singular values</p>
          <XYChart height={260} series={r.spectrum} yLog xLabel="i" yLabel="σᵢ" handles={cut} />
        </div>
      </div>
    </Interactive>
  )
}
