import { useMemo } from 'react'
import { Bars, choice, Figure, formatNumber, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { PEAK, SIZE, distort, psnr, ssim, testImage, type Distortion, type Image } from '../_shared/image'

const REFERENCE = testImage()
const AXIS = Array.from({ length: SIZE }, (_, i) => i)
const KINDS: { value: Distortion; label: string }[] = [
  { value: 'noise', label: 'Gaussian noise' },
  { value: 'blur', label: 'blur' },
  { value: 'salt-and-pepper', label: 'salt and pepper' },
  { value: 'contrast', label: 'lower contrast' },
  { value: 'shift', label: 'brighter' },
]

/** Row-major image to heatmap rows, flipped so row 0 is at the top. */
const rows = (img: Float64Array | Image) =>
  AXIS.map((r) => Array.from(img.subarray((SIZE - 1 - r) * SIZE, (SIZE - r) * SIZE))).map((row) =>
    row.map((v) => (Number.isNaN(v) ? 0 : v)),
  )

const REFERENCE_ROWS = rows(REFERENCE)
const IMAGE_RANGE: [number, number] = [0, PEAK]
const SSIM_RANGE: [number, number] = [-1, 1]

/** One image as a square raster, row 0 at the top. */
function ImagePanel(props: { title: string; z: number[][]; range: [number, number]; scale?: 'diverging' }) {
  const xAxis = useAxis({ range: [-0.5, SIZE - 0.5] })
  const yAxis = useAxis({ range: [-0.5, SIZE - 0.5], equal: xAxis })
  return (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{props.title}</div>
      <Plot x={xAxis} y={yAxis} height={240}>
        <Raster x={AXIS} y={AXIS} z={props.z} range={props.range} scale={props.scale} valueLabel="value" />
      </Plot>
    </div>
  )
}

/**
 * Five distortions of one image, each tuned to the same mean squared error and therefore the same PSNR. SSIM tells
 * them apart: it barely penalises a brightness or contrast change that preserves structure, and heavily penalises
 * noise, which destroys it.
 */
export function SameMseDifferentSsim() {
  const state = useFigureState({
    kind: choice<Distortion>(KINDS, 'noise', { label: 'distortion' }),
    target: int(225, { min: 25, max: 900, step: 25, label: 'target mean squared error', format: (v) => String(v) }),
  })

  const all = useMemo(
    () =>
      KINDS.map((k) => {
        const img = distort(REFERENCE, k.value, state.target)
        return { ...k, img, psnr: psnr(REFERENCE, img), ssim: ssim(REFERENCE, img) }
      }),
    [state.target],
  )
  const current = all.find((a) => a.value === state.kind)!

  const maps = useMemo(
    () => ({ distorted: rows(current.img), ssim: rows(current.ssim.map) }),
    [current.img, current.ssim.map],
  )

  const xAxis = useAxis({ label: 'distortion', hold: 'union' })
  const yAxis = useAxis({ label: 'mean SSIM', range: [0, 1] })
  return (
    <Figure
      title="Same PSNR, different SSIM"
      state={state}
      caption="Each distortion is tuned so that its mean squared error against the reference equals the target, so all five have the same PSNR. SSIM ranks them very differently: a uniform brightening or a contrast reduction keeps the image's structure and scores near 1, while noise at the same error destroys local structure and scores far lower. The right-hand map shows SSIM in each 7 × 7 window."
      readouts={
        <>
          <Readout label="PSNR" value={`${formatNumber(current.psnr)} dB`} />
          <Readout label="SSIM" value={formatNumber(current.ssim.mean)} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3">
          <ImagePanel title="reference" z={REFERENCE_ROWS} range={IMAGE_RANGE} />
          <ImagePanel title={`distorted: ${current.label}`} z={maps.distorted} range={IMAGE_RANGE} />
          <ImagePanel title="SSIM map" z={maps.ssim} range={SSIM_RANGE} scale="diverging" />
        </div>
        <Plot
          x={xAxis}
          y={yAxis}
          height={220}
          ariaLabel={`SSIM for ${all.map((a) => `${a.label} ${a.ssim.mean.toFixed(2)}`).join(', ')}`}
        >
          <Bars name="SSIM at equal PSNR" x={all.map((_, i) => i + 1)} y={all.map((a) => a.ssim.mean)} slot={0} />
        </Plot>
        <p className="text-center text-xs text-muted-foreground">
          {all.map((a, i) => `${i + 1}: ${a.label} (${a.ssim.mean.toFixed(3)})`).join(' · ')}
        </p>
      </div>
    </Figure>
  )
}
