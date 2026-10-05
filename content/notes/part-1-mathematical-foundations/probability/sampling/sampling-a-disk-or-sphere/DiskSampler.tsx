import { useMemo } from 'react'
import { Figure, int, Plot, Readout, seriesLayers, type SeriesSpec, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'

/** The circle that splits the unit disc into two halves of equal area. */
const HALF = Math.SQRT1_2

const circle = (r: number, name: string): SeriesSpec => {
  const t = toFlat(linspace(0, 2 * Math.PI, 129))
  return { name, type: 'line', x: t.map((a) => r * Math.cos(a)), y: t.map((a) => r * Math.sin(a)), muted: true }
}

type Panel = { points: SeriesSpec; inside: number }

function sample(n: number, radius: (u: number) => number, name: string): Panel {
  const r = stream(3)
  const x: number[] = []
  const y: number[] = []
  let inside = 0
  for (let i = 0; i < n; i++) {
    const rad = radius(uniform(r))
    const angle = 2 * Math.PI * uniform(r)
    x.push(rad * Math.cos(angle))
    y.push(rad * Math.sin(angle))
    if (rad <= HALF) inside++
  }
  return { points: { name, type: 'scatter', x, y, slot: 0 }, inside: inside / n }
}

/** The same uniform angles with two radius rules: R = U crowds the centre, R = √U is uniform over the area. */
export function DiskSampler() {
  const state = useFigureState({
    n: int(1500, { min: 200, max: 4000, step: 100, label: 'points' }),
  })
  const { naive, correct } = useMemo(
    () => ({ naive: sample(state.n, (u) => u, 'R = U'), correct: sample(state.n, Math.sqrt, 'R = √U') }),
    [state.n],
  )
  const guides = useMemo(() => [circle(1, 'unit circle'), circle(HALF, 'r = 1/√2, half the area')], [])
  const pct = (v: number) => `${(100 * v).toFixed(1)}%`

  const xAxis = useAxis({ label: 'x', range: [-1.05, 1.05] })
  const yAxis = useAxis({ label: 'y', range: [-1.05, 1.05], equal: xAxis })
  const xAxis2 = useAxis({ label: 'x', range: [-1.05, 1.05] })
  const yAxis2 = useAxis({ label: 'y', range: [-1.05, 1.05], equal: xAxis2 })
  return (
    <Figure
      title="Uniform radius is not uniform area"
      state={state}
      caption="Both panels draw the angle uniformly. On the left the radius is a uniform number U; on the right it is √U. The inner circle, of radius 1/√2, encloses exactly half of the disc's area, so a uniform sampler puts half its points inside it. R = U puts 1/√2 ≈ 71% of its points there and crowds the centre."

      readouts={
        <>
          <Readout label="inside r = 1/√2, R = U" value={pct(naive.inside)} />
          <Readout label="inside r = 1/√2, R = √U" value={pct(correct.inside)} />
          <Readout label="uniform target" value="50%" />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers([naive.points, ...guides])}
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers([correct.points, ...guides])}
        </Plot>
      </div>
    </Figure>
  )
}
