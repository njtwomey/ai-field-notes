import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
} from 'aifn-render'
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

/**
 * Five distortions of one image, each tuned to the same mean squared error and therefore the same PSNR. SSIM tells
 * them apart: it barely penalises a brightness or contrast change that preserves structure, and heavily penalises
 * noise, which destroys it.
 */
export function SameMseDifferentSsim() {
  const [kind, setKind] = useState<Distortion>('noise')
  const target = useParam(225, { min: 25, max: 900, step: 25 })

  const all = useMemo(
    () =>
      KINDS.map((k) => {
        const img = distort(REFERENCE, k.value, target.value)
        return { ...k, img, psnr: psnr(REFERENCE, img), ssim: ssim(REFERENCE, img) }
      }),
    [target.value],
  )
  const current = all.find((a) => a.value === kind)!

  const panel = (title: string, z: number[][], range: [number, number], scale?: 'diverging') => (
    <div className="min-w-0 space-y-1">
      <div className="text-center text-xs text-muted-foreground">{title}</div>
      <Heatmap
        x={AXIS}
        y={AXIS}
        z={z}
        range={range}
        scale={scale}
        xLabel=""
        yLabel=""
        valueLabel="value"
        height={240}
      />
    </div>
  )

  return (
    <Interactive
      title="Same PSNR, different SSIM"
      caption="Each distortion is tuned so that its mean squared error against the reference equals the target, so all five have the same PSNR. SSIM ranks them very differently: a uniform brightening or a contrast reduction keeps the image's structure and scores near 1, while noise at the same error destroys local structure and scores far lower. The right-hand map shows SSIM in each 7 × 7 window."
      controls={
        <>
          <ParamChoice label="distortion" value={kind} onChange={setKind} options={KINDS} />
          <ParamSlider label="target mean squared error" param={target} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="PSNR" value={`${formatNumber(current.psnr)} dB`} />
          <Readout label="SSIM" value={formatNumber(current.ssim.mean)} />
        </>
      }
    >
      <div className="space-y-4">
        <div className="grid gap-3 md:grid-cols-3">
          {panel('reference', rows(REFERENCE), [0, PEAK])}
          {panel(`distorted: ${current.label}`, rows(current.img), [0, PEAK])}
          {panel('SSIM map', rows(current.ssim.map), [-1, 1], 'diverging')}
        </div>
        <XYChart
          height={220}
          xLabel="distortion"
          yLabel="mean SSIM"
          yRange={[0, 1]}
          series={[
            {
              name: 'SSIM at equal PSNR',
              type: 'bar',
              x: all.map((_, i) => i + 1),
              y: all.map((a) => a.ssim.mean),
              slot: 0,
            },
          ]}
          ariaLabel={`SSIM for ${all.map((a) => `${a.label} ${a.ssim.mean.toFixed(2)}`).join(', ')}`}
        />
        <p className="text-center text-xs text-muted-foreground">
          {all.map((a, i) => `${i + 1}: ${a.label} (${a.ssim.mean.toFixed(3)})`).join(' · ')}
        </p>
      </div>
    </Interactive>
  )
}
