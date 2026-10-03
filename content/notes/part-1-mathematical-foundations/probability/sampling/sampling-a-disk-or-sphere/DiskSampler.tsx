import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, type XYSeries } from 'aifn-render'
import { linspace, rng } from '@/lib/math'

/** The circle that splits the unit disc into two halves of equal area. */
const HALF = Math.SQRT1_2

const circle = (r: number, name: string): XYSeries => {
  const t = linspace(0, 2 * Math.PI, 129)
  return { name, type: 'line', x: t.map((a) => r * Math.cos(a)), y: t.map((a) => r * Math.sin(a)), muted: true }
}

type Panel = { points: XYSeries; inside: number }

function sample(n: number, radius: (u: number) => number, name: string): Panel {
  const r = rng(3)
  const x: number[] = []
  const y: number[] = []
  let inside = 0
  for (let i = 0; i < n; i++) {
    const rad = radius(r.uniform())
    const angle = 2 * Math.PI * r.uniform()
    x.push(rad * Math.cos(angle))
    y.push(rad * Math.sin(angle))
    if (rad <= HALF) inside++
  }
  return { points: { name, type: 'scatter', x, y, slot: 0 }, inside: inside / n }
}

/** The same uniform angles with two radius rules: R = U crowds the centre, R = √U is uniform over the area. */
export function DiskSampler() {
  const [n, setN] = useState(1500)
  const { naive, correct } = useMemo(
    () => ({ naive: sample(n, (u) => u, 'R = U'), correct: sample(n, Math.sqrt, 'R = √U') }),
    [n],
  )
  const guides = useMemo(() => [circle(1, 'unit circle'), circle(HALF, 'r = 1/√2, half the area')], [])
  const pct = (v: number) => `${(100 * v).toFixed(1)}%`

  return (
    <Interactive
      title="Uniform radius is not uniform area"
      caption="Both panels draw the angle uniformly. On the left the radius is a uniform number U; on the right it is √U. The inner circle, of radius 1/√2, encloses exactly half of the disc's area, so a uniform sampler puts half its points inside it. R = U puts 1/√2 ≈ 71% of its points there and crowds the centre."
      controls={<ParamSlider label="points" value={n} onChange={setN} min={200} max={4000} step={100} />}
      readout={
        <>
          <Readout label="inside r = 1/√2, R = U" value={pct(naive.inside)} />
          <Readout label="inside r = 1/√2, R = √U" value={pct(correct.inside)} />
          <Readout label="uniform target" value="50%" />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={[naive.points, ...guides]}
          xRange={[-1.05, 1.05]}
          yRange={[-1.05, 1.05]}
          equalAspect
          xLabel="x"
          yLabel="y"
        />
        <XYChart
          series={[correct.points, ...guides]}
          xRange={[-1.05, 1.05]}
          yRange={[-1.05, 1.05]}
          equalAspect
          xLabel="x"
          yLabel="y"
        />
      </div>
    </Interactive>
  )
}
