import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, useParam, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'

// Minimal sample sizes: a line, a homography, the seven- and eight-point fundamental matrix.
const MODELS = [
  { s: 2, name: 's = 2 (line)' },
  { s: 4, name: 's = 4 (homography)' },
  { s: 7, name: 's = 7 (fundamental, 7-point)' },
  { s: 8, name: 's = 8 (fundamental, 8-point)' },
]

const iterations = (p: number, w: number, s: number) => Math.log(1 - p) / Math.log(1 - w ** s)
const W = linspace(0.1, 0.95, 120)

/** RANSAC iterations needed to draw one all-inlier sample with probability p, against the inlier ratio w. */
export function RansacIterations() {
  const [p, setP] = useState(0.99)
  const w = useParam(0.5, { min: 0.1, max: 0.95, step: 0.01 })

  const series = useMemo<XYSeries[]>(
    () =>
      MODELS.map((m, i) => ({
        name: m.name,
        type: 'line',
        x: W,
        y: W.map((wi) => Math.max(1, iterations(p, wi, m.s))),
        slot: i,
      })),
    [p],
  )

  return (
    <Interactive
      title="How many RANSAC samples are enough"
      caption="The number of random minimal samples N = log(1 − p) / log(1 − wˢ) needed to draw at least one sample of s inliers with probability p, when a fraction w of the correspondences are inliers. Drag the vertical line to set w. The cost explodes as w falls, and faster for larger s: at w = 0.3 a homography needs about 566 samples and an 8-point fundamental matrix about 70,000."
      controls={
        <>
          <ParamSlider label="inlier ratio w" param={w} format={(v) => v.toFixed(2)} />
          <ParamSlider
            label="success probability p"
            value={p}
            onChange={setP}
            min={0.9}
            max={0.999}
            step={0.001}
            format={(v) => v.toFixed(3)}
          />
        </>
      }
      readout={
        <>
          {MODELS.map((m) => (
            <Readout
              key={m.s}
              label={`N for s = ${m.s}`}
              value={Math.ceil(iterations(p, w.value, m.s)).toLocaleString('en-GB')}
            />
          ))}
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="inlier ratio w"
        yLabel="samples N"
        xRange={[0.1, 0.95]}
        yLog
        handles={[{ kind: 'x', at: w.value, onDrag: w.set, label: 'w' }]}
        ariaLabel="RANSAC iterations against inlier ratio"
      />
    </Interactive>
  )
}
