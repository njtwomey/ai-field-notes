import { useContext, useMemo, type ReactNode } from 'react'
import {
  Curve,
  Figure,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import { testImage } from './testImage'

const PANEL = 300
const SPECTRUM = 220

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** The 64 × 64 test image and its thin SVD from the engine: U (columns), the singular values S, and V (columns). */
function imageSvd() {
  const image = testImage(64)
  const { U, S, V } = svd(tensor(image))
  return { image, u: toRows(U), s: toFlat(S), v: toRows(V) }
}

/** Rank-k reconstructions of a test image from its SVD. */
export function LowRankImage() {
  const data = useComputed(imageSvd, [], { mode: 'release' }).value
  // The test image is 64 × 64, so its rank is at most 64.
  const state = useFigureState({
    k: int(4, { ge: 1, le: 64, suggestions: [1, 4, 16, 64], label: 'rank k' }),
  })
  const size = data.s.length
  const k = Math.min(state.k, size)
  // Square images: equal pixel length per unit on both axes.
  const xAxis = useAxis({ range: [-0.5, size - 0.5] })
  const yAxis = useAxis({ range: [-0.5, size - 0.5], equal: xAxis })
  const xAxis2 = useAxis({ range: [-0.5, size - 0.5] })
  const yAxis2 = useAxis({ range: [-0.5, size - 0.5], equal: xAxis2 })
  const xAxis3 = useAxis({ label: 'i', range: [0, size + 1] })
  const yAxis3 = useAxis({ label: 'σᵢ', log: true, range: [0.01, 100] })

  const r = useMemo(() => {
    // A_k = Σ_{i<k} σᵢ uᵢ vᵢᵀ; rows are reversed so the first image row is drawn at the top.
    const approx = Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, col) => {
        let v = 0
        for (let i = 0; i < k; i++) v += data.s[i] * data.u[row][i] * data.v[col][i]
        return v
      }),
    ).reverse()
    const total = data.s.reduce((acc, s) => acc + s * s, 0)
    const kept = data.s.slice(0, k).reduce((acc, s) => acc + s * s, 0)
    const axis = Array.from({ length: size }, (_, i) => i)
    // A log axis cannot show the exact zeros of a rank-deficient image, so the spectrum stops at the last nonzero value.
    const rank = data.s.filter((v) => v > 1e-9).length
    const index = axis.slice(0, rank).map((i) => i + 1)
    const sv = data.s.slice(0, rank)
    const spectrum = {
      kept: { x: index.slice(0, k), y: sv.slice(0, k) },
      discarded: { x: index.slice(k), y: sv.slice(k) },
      all: { x: index, y: sv },
    }
    return {
      size,
      axis,
      approx,
      original: [...data.image].reverse(),
      energy: kept / total,
      error: Math.sqrt((total - kept) / total),
      storage: (k * (2 * size + 1)) / (size * size),
      rank,
      spectrum,
    }
  }, [data, k, size])

  // The cut sits between bar k (kept) and bar k + 1 (discarded); dragging it picks the nearest such gap.
  return (
    <Figure
      title="Keeping the k largest singular values"
      caption={`The rank-k approximation keeps the first k terms σᵢuᵢvᵢᵀ. The smooth gradient, the disc and the bar appear within a few terms; the diagonal band and the checkerboard need many, because they are not aligned with rows and columns. No other rank-k matrix is closer to the image. Below: the singular values on a log scale, the kept ones in colour. The image has rank ${r.rank}, so its last ${r.size - r.rank} singular values are exactly zero and are not drawn. Drag the dashed cut on the spectrum to choose k.`}
      state={state}
      readouts={
        <>
          <Readout label="energy kept Σσᵢ² (i ≤ k) / Σσᵢ²" value={`${(100 * r.energy).toFixed(1)}%`} />
          <Readout label="relative Frobenius error" value={`${(100 * r.error).toFixed(1)}%`} />
          <Readout label="numbers stored vs full image" value={`${(100 * r.storage).toFixed(0)}%`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <div>
          <p className="mb-1 text-center text-sm font-medium">Original</p>
          <FixedHeight height={PANEL}>
            <Plot x={xAxis} y={yAxis}>
              <Raster x={r.axis} y={r.axis} z={r.original} range={[0, 1]} colorBar={false} valueLabel="intensity" />
            </Plot>
          </FixedHeight>
        </div>
        <div>
          <p className="mb-1 text-center text-sm font-medium">Rank {k}</p>
          <FixedHeight height={PANEL}>
            <Plot x={xAxis2} y={yAxis2}>
              <Raster x={r.axis} y={r.axis} z={r.approx} range={[0, 1]} colorBar={false} valueLabel="intensity" />
            </Plot>
          </FixedHeight>
        </div>
        <div className="md:col-span-2">
          <p className="mb-1 text-center text-sm font-medium">Singular values</p>
          <FixedHeight height={SPECTRUM}>
            <Plot x={xAxis3} y={yAxis3}>
              <Curve name="spectrum" x={r.spectrum.all.x} y={r.spectrum.all.y} muted silent />
              <Points name="kept" x={r.spectrum.kept.x} y={r.spectrum.kept.y} slot={0} size={6} />
              {r.spectrum.discarded.x.length > 0 && (
                <Points name="discarded" x={r.spectrum.discarded.x} y={r.spectrum.discarded.y} muted size={5} />
              )}
              <Handle
                kind="x"
                at={k + 0.5}
                label="k"
                onDrag={(x) => state.set('k', Math.min(Math.max(Math.round(x - 0.5), 1), r.size))}
              />
            </Plot>
          </FixedHeight>
        </div>
      </div>
    </Figure>
  )
}
