import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSlider, Readout } from '@/components/viz'
import { AXIS, SIZE, gradients, testImage, type Image } from '../_shared/image'

type Stage = 'magnitude' | 'nms' | 'edges'

const INPUT = testImage(0.06)

function magnitudeAndDirection(sigma: number): { mag: Image; dir: number[][] } {
  const { gx, gy } = gradients(INPUT, sigma)
  const mag = gx.map((row, r) => row.map((v, c) => Math.hypot(v, gy[r][c])))
  // Gradient direction quantised to 0°, 45°, 90° or 135° (index 0 to 3).
  const dir = gx.map((row, r) =>
    row.map((v, c) => {
      const deg = ((Math.atan2(gy[r][c], v) * 180) / Math.PI + 180) % 180
      return Math.round(deg / 45) % 4
    }),
  )
  return { mag, dir }
}

// Neighbour offsets (row, column) along each quantised gradient direction.
const STEP: [number, number][] = [
  [0, 1],
  [1, 1],
  [1, 0],
  [1, -1],
]

function nonMaxSuppress(mag: Image, dir: number[][]): Image {
  const at = (r: number, c: number) => (r >= 0 && r < SIZE && c >= 0 && c < SIZE ? mag[r][c] : 0)
  return mag.map((row, r) =>
    row.map((m, c) => {
      const [dr, dc] = STEP[dir[r][c]]
      return m >= at(r + dr, c + dc) && m > at(r - dr, c - dc) ? m : 0
    }),
  )
}

/** Keep weak pixels (≥ low) only if 8-connected to a strong pixel (≥ high). */
function hysteresis(thin: Image, low: number, high: number): Image {
  const out: Image = thin.map((row) => row.map(() => 0))
  const stack: [number, number][] = []
  thin.forEach((row, r) =>
    row.forEach((m, c) => {
      if (m >= high) {
        out[r][c] = 1
        stack.push([r, c])
      }
    }),
  )
  while (stack.length) {
    const [r, c] = stack.pop()!
    for (let dr = -1; dr <= 1; dr++)
      for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr
        const cc = c + dc
        if (rr < 0 || rr >= SIZE || cc < 0 || cc >= SIZE || out[rr][cc]) continue
        if (thin[rr][cc] >= low) {
          out[rr][cc] = 1
          stack.push([rr, cc])
        }
      }
  }
  return out
}

const count = (img: Image) => img.reduce((a, row) => a + row.filter((v) => v > 0).length, 0)

/** The stages of the Canny detector on a noisy test image. */
export function CannyExplorer() {
  const [stage, setStage] = useState<Stage>('edges')
  const [sigma, setSigma] = useState(1.2)
  const [high, setHigh] = useState(0.1)
  const [ratio, setRatio] = useState(0.4)
  const low = high * ratio

  const { mag, dir } = useMemo(() => magnitudeAndDirection(sigma), [sigma])
  const thin = useMemo(() => nonMaxSuppress(mag, dir), [mag, dir])
  const edges = useMemo(() => hysteresis(thin, low, high), [thin, low, high])
  const strongOnly = useMemo(() => count(thin.map((row) => row.map((m) => (m >= high ? 1 : 0)))), [thin, high])
  const shown = stage === 'magnitude' ? mag : stage === 'nms' ? thin : edges

  return (
    <Interactive
      title="The stages of the Canny edge detector"
      caption="Left: a test image with Gaussian noise of standard deviation 0.06. Right: one stage of the detector. The gradient magnitude marks edges as ridges several pixels wide. Non-maximum suppression thins each ridge to one pixel. Hysteresis keeps a pixel above the low threshold only if it connects to a pixel above the high threshold. A small σ lets noise through; a large σ rounds corners and merges nearby edges."
      controls={
        <>
          <ParamChoice
            label="stage"
            value={stage}
            onChange={setStage}
            options={[
              { value: 'magnitude', label: '1. gradient' },
              { value: 'nms', label: '2. suppression' },
              { value: 'edges', label: '3. hysteresis' },
            ]}
          />
          <ParamSlider
            label="smoothing σ (pixels)"
            value={sigma}
            onChange={setSigma}
            min={0}
            max={3}
            step={0.1}
            format={(v) => v.toFixed(1)}
          />
          <ParamSlider
            label="high threshold"
            value={high}
            onChange={setHigh}
            min={0.02}
            max={0.3}
            step={0.01}
            format={(v) => v.toFixed(2)}
          />
          <ParamSlider
            label="low / high"
            value={ratio}
            onChange={setRatio}
            min={0.1}
            max={1}
            step={0.05}
            format={(v) => v.toFixed(2)}
          />
        </>
      }
      readout={
        <>
          <Readout label="low threshold" value={low.toFixed(3)} />
          <Readout label="pixels above high" value={String(strongOnly)} />
          <Readout label="edge pixels after hysteresis" value={String(count(edges))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={INPUT}
          range={[0, 1]}
          xLabel="column"
          yLabel="row"
          valueLabel="intensity"
          height={320}
          ariaLabel="Noisy input image"
        />
        <Heatmap
          x={AXIS}
          y={AXIS}
          z={shown}
          range={stage === 'edges' ? [0, 1] : [0, 0.4]}
          xLabel="column"
          yLabel="row"
          valueLabel={stage === 'edges' ? 'edge' : 'gradient magnitude'}
          height={320}
          ariaLabel="Canny detector stage"
        />
      </div>
    </Interactive>
  )
}
