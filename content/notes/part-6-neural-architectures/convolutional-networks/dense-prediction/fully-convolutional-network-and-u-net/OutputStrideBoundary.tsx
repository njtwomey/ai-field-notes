import { useMemo } from 'react'
import { choice, Figure, formatNumber, Plot, Raster, Readout, Segments, useAxis, useFigureState } from 'aifn-render'
import { avgPool2d } from 'aifn-compute/nn/functional'
import { convTranspose } from 'aifn-compute/foundation/convolution'
import { tensor, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { boundaryF1, maskIou } from 'aifn-methods/evaluation/detection'

/** Image side in pixels: divisible by every stride offered. */
const N = 128
const AXIS = Array.from({ length: N }, (_, i) => i)

/** The true mask: a disc, a 4-pixel-wide pole and a 12-pixel square. */
const MASK = (() => {
  const m = new Float64Array(N * N)
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++) {
      const disc = (r - 64) ** 2 + (c - 44) ** 2 <= 28 ** 2
      const pole = c >= 94 && c < 98 && r >= 16 && r < 112
      const square = r >= 20 && r < 32 && c >= 104 && c < 116
      m[r * N + c] = disc || pole || square ? 1 : 0
    }
  return m
})()

/** Pixel edges where the true mask changes, drawn as the true outline. */
const OUTLINE = (() => {
  const segs: { from: [number, number]; to: [number, number] }[] = []
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++) {
      const v = MASK[r * N + c]
      if (c + 1 < N && MASK[r * N + c + 1] !== v) segs.push({ from: [c + 0.5, r - 0.5], to: [c + 0.5, r + 0.5] })
      if (r + 1 < N && MASK[(r + 1) * N + c] !== v) segs.push({ from: [c - 0.5, r + 0.5], to: [c + 0.5, r + 0.5] })
    }
  return segs
})()

const ROWS = (flat: ArrayLike<number>) =>
  Array.from({ length: N }, (_, r) => Array.from(flat).slice(r * N, (r + 1) * N))

/**
 * The upsampling kernel of a stride-s transposed convolution: bilinear (FCN's initialisation, size 2s, a tent centred
 * between the middle taps) or nearest (an s × s block of ones).
 */
function upsamplingKernel(s: number, mode: 'bilinear' | 'nearest'): Tensor {
  const k = mode === 'bilinear' ? 2 * s : s
  const centre = (k - 1) / 2
  const w1 = Array.from({ length: k }, (_, i) => (mode === 'bilinear' ? 1 - Math.abs(i - centre) / s : 1))
  const w = new Float64Array(k * k)
  for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) w[i * k + j] = w1[i] * w1[j]
  return tensor(w, [1, 1, k, k])
}

/**
 * The best a stride-s score map can do: each coarse cell holds the true fraction of foreground in it (average pooling
 * of the mask), and a transposed convolution brings the map back to full resolution.
 */
function scoreAtStride(s: number, mode: 'bilinear' | 'nearest'): number[] {
  if (s === 1) return Array.from(MASK)
  const coarse = avgPool2d(tensor(MASK, [1, 1, N, N]), s)
  const pad = mode === 'bilinear' ? s / 2 : 0
  return toFlat(convTranspose(coarse, upsamplingKernel(s, mode), { stride: s, padding: pad }) as Tensor)
}

/** A perfect classifier read out at output stride s: how much of the mask survives the coarse map. */
export function OutputStrideBoundary() {
  const state = useFigureState({
    stride: choice([1, 2, 4, 8, 16, 32], 8, { label: 'output stride' }),
    upsampling: choice(['bilinear', 'nearest'] as const, 'bilinear', { label: 'upsampling' }),
  })
  const r = useMemo(() => {
    const score = scoreAtStride(state.stride, state.upsampling)
    const pred = score.map((v) => (v >= 0.5 ? 1 : 0))
    const grid: { from: [number, number]; to: [number, number] }[] = []
    if (state.stride >= 4)
      for (let e = state.stride; e < N; e += state.stride) {
        grid.push({ from: [e - 0.5, -0.5], to: [e - 0.5, N - 0.5] })
        grid.push({ from: [-0.5, e - 0.5], to: [N - 0.5, e - 0.5] })
      }
    const poleRow = 64 * N
    const poleKept = pred.slice(poleRow + 94, poleRow + 98).some((v) => v === 1)
    return {
      z: ROWS(score),
      grid,
      iou: maskIou(MASK, pred),
      f1: boundaryF1(ROWS(MASK), ROWS(pred)),
      poleKept,
    }
  }, [state.stride, state.upsampling])

  const x = useAxis({ label: 'column', range: [-0.5, N - 0.5] })
  const y = useAxis({ label: 'row', range: [-0.5, N - 0.5], inverse: true, equal: x })
  return (
    <Figure
      title="What a coarse score map can place"
      state={state}
      aspect={1}
      caption={`A 128 × 128 mask (outlined in ink): a disc, a pole 4 pixels wide and a 12-pixel square. Each cell of a stride-s score map is given the true share of foreground in its s × s block, the best any classifier at that stride can do; the map is then upsampled to full resolution by a transposed convolution and thresholded at 0.5 (the contour). Choose the output stride and the upsampling. At stride 4 every shape survives. At stride 8 the pole, which straddles two cells at a quarter each, disappears; at 16 the square goes too; at 32, the stride of FCN-32s, the disc is a blob. Nearest upsampling gives blocky outlines and a lower IoU than bilinear at every stride. Thin and small structures are lost before large boundaries are, which is why decoders bring back fine encoder maps.`}
      readouts={
        <>
          <Readout label="score map" value={`${N / state.stride} × ${N / state.stride} cells`} />
          <Readout label="mask IoU" value={formatNumber(r.iou)} />
          <Readout label="boundary F1 (2-pixel tolerance)" value={formatNumber(r.f1)} />
          <Readout label="pole found" value={r.poleKept ? 'yes' : 'no'} />
        </>
      }
    >
      <Plot x={x} y={y} ariaLabel="Upsampled foreground score with the predicted and true outlines">
        <Raster
          x={AXIS}
          y={AXIS}
          z={r.z}
          range={[0, 1]}
          valueLabel="foreground score"
          boundary={0.5}
          fillOpacity={0.75}
        />
        {r.grid.length > 0 && <Segments name="coarse cells" segments={r.grid} muted />}
        <Segments name="true outline" segments={OUTLINE} emphasis width={1.5} />
      </Plot>
    </Figure>
  )
}
