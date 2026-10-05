import { useMemo } from 'react'
import {
  Figure,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

// Minimal sample sizes: a line, a homography, the seven- and eight-point fundamental matrix.
const MODELS = [
  { s: 2, name: 's = 2 (line)' },
  { s: 4, name: 's = 4 (homography)' },
  { s: 7, name: 's = 7 (fundamental, 7-point)' },
  { s: 8, name: 's = 8 (fundamental, 8-point)' },
]

const iterations = (p: number, w: number, s: number) => Math.log(1 - p) / Math.log(1 - w ** s)
const W = toFlat(linspace(0.1, 0.95, 120))

/** RANSAC iterations needed to draw one all-inlier sample with probability p, against the inlier ratio w. */
export function RansacIterations() {
  const state = useFigureState({
    w: slider(0.1, 0.95, 0.5, { step: 0.01, label: 'inlier ratio w', format: (v) => v.toFixed(2) }),
    p: slider(0.9, 0.999, 0.99, { step: 0.001, label: 'success probability p', format: (v) => v.toFixed(3) }),
  })

  const series = useMemo<SeriesSpec[]>(
    () =>
      MODELS.map((m, i) => ({
        name: m.name,
        type: 'line',
        x: W,
        y: W.map((wi) => Math.max(1, iterations(state.p, wi, m.s))),
        slot: i,
      })),
    [state.p],
  )

  const xAxis = useAxis({ label: 'inlier ratio w', range: [0.1, 0.95] })
  const yAxis = useAxis({ label: 'samples N', hold: 'union', log: true })
  return (
    <Figure
      title="How many RANSAC samples are enough"
      state={state}
      caption="The number of random minimal samples N = log(1 − p) / log(1 − wˢ) needed to draw at least one sample of s inliers with probability p, when a fraction w of the correspondences are inliers. Drag the vertical line to set w. The cost explodes as w falls, and faster for larger s: at w = 0.3 a homography needs about 566 samples and an 8-point fundamental matrix about 70,000."

      readouts={
        <>
          {MODELS.map((m) => (
            <Readout
              key={m.s}
              label={`N for s = ${m.s}`}
              value={Math.ceil(iterations(state.p, state.w, m.s)).toLocaleString('en-GB')}
            />
          ))}
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel={'RANSAC iterations against inlier ratio'}>
        {seriesLayers(series)}
        <Handle {...state.handle('w', { label: 'w' })} />
      </Plot>
    </Figure>
  )
}
