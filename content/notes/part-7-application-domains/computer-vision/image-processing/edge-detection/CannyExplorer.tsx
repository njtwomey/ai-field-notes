import { useMemo } from 'react'
import { choice, Figure, float, Plot, Raster, Readout, slider, useAxis, useFigureState } from 'aifn-render'
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
  const state = useFigureState({
    stage: choice<Stage>(
      [
        { value: 'magnitude', label: '1. gradient' },
        { value: 'nms', label: '2. suppression' },
        { value: 'edges', label: '3. hysteresis' },
      ],
      'edges',
      { label: 'stage' },
    ),
    sigma: float(1.2, { min: 0, max: 3, step: 0.1, label: 'smoothing σ (pixels)', format: (v) => v.toFixed(1) }),
    high: slider(0.02, 0.3, 0.1, { step: 0.01, label: 'high threshold', format: (v) => v.toFixed(2) }),
    ratio: float(0.4, { min: 0.1, max: 1, step: 0.05, label: 'low / high', format: (v) => v.toFixed(2) }),
  })
  const low = state.high * state.ratio

  const { mag, dir } = useMemo(() => magnitudeAndDirection(state.sigma), [state.sigma])
  const thin = useMemo(() => nonMaxSuppress(mag, dir), [mag, dir])
  const edges = useMemo(() => hysteresis(thin, low, state.high), [thin, low, state.high])
  const strongOnly = useMemo(
    () => count(thin.map((row) => row.map((m) => (m >= state.high ? 1 : 0)))),
    [thin, state.high],
  )
  const shown = state.stage === 'magnitude' ? mag : state.stage === 'nms' ? thin : edges

  const xAxis = useAxis({ label: 'column' })
  const yAxis = useAxis({ label: 'row' })
  const xAxis2 = useAxis({ label: 'column' })
  const yAxis2 = useAxis({ label: 'row' })
  return (
    <Figure
      title="The stages of the Canny edge detector"
      state={state}
      caption="Left: a test image with Gaussian noise of standard deviation 0.06. Right: one stage of the detector. The gradient magnitude marks edges as ridges several pixels wide. Non-maximum suppression thins each ridge to one pixel. Hysteresis keeps a pixel above the low threshold only if it connects to a pixel above the high threshold. A small σ lets noise through; a large σ rounds corners and merges nearby edges."

      readouts={
        <>
          <Readout label="low threshold" value={low.toFixed(3)} />
          <Readout label="pixels above high" value={String(strongOnly)} />
          <Readout label="edge pixels after hysteresis" value={String(count(edges))} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320} ariaLabel={'Noisy input image'}>
          <Raster x={AXIS} y={AXIS} z={INPUT} range={[0, 1]} valueLabel={'intensity'} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320} ariaLabel={'Canny detector stage'}>
          <Raster
            x={AXIS}
            y={AXIS}
            z={shown}
            range={state.stage === 'edges' ? [0, 1] : [0, 0.4]}
            valueLabel={state.stage === 'edges' ? 'edge' : 'gradient magnitude'}
          />
        </Plot>
      </div>
    </Figure>
  )
}
