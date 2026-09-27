import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'

const T = linspace(0, 4, 161)
const FAMILY = [0.5, 1.5, 2.5, 3.5]

/** x(t) = 0 until time c, then (t − c)²/4: every such curve solves ẋ = √x with x(0) = 0. */
const solution = (c: number) => T.map((t) => (t <= c ? 0 : (t - c) ** 2 / 4))

export function NonUniqueness() {
  const c = useParam(1, { min: 0, max: 4, step: 0.05 })
  const series = useMemo<XYSeries[]>(
    () => [
      ...FAMILY.map((f) => ({ name: 'other solutions', type: 'line' as const, x: T, y: solution(f), muted: true })),
      { name: 'x(t) = 0', type: 'line', x: T, y: T.map(() => 0), slot: 1, dashed: true },
      { name: 'solution leaving 0 at t = c', type: 'line', x: T, y: solution(c.value), slot: 0 },
    ],
    [c.value],
  )
  const handles = useMemo<Handle[]>(() => [{ kind: 'x', at: c.value, label: 'c', onDrag: c.set }], [c])
  const t = 3
  const x = t <= c.value ? 0 : (t - c.value) ** 2 / 4
  return (
    <Interactive
      title="Many solutions from one initial value"
      caption="Every curve here satisfies ẋ = √x and starts at x(0) = 0. The solution may rest at 0 for any length of time c and then leave along (t − c)²/4. Drag the vertical line, or use the slider, to choose c. The right side √x has an infinite slope at x = 0, so it is not Lipschitz there and uniqueness fails."
      controls={<ParamSlider label="departure time c" param={c} />}
      readout={
        <>
          <Readout label="x(3)" value={formatNumber(x)} />
          <Readout label="ẋ(3)" value={formatNumber(t <= c.value ? 0 : (t - c.value) / 2)} />
          <Readout label="√x(3)" value={formatNumber(Math.sqrt(x))} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="t"
        yLabel="x"
        series={series}
        handles={handles}
        xRange={[0, 4]}
        yRange={[-0.2, 4]}
      />
    </Interactive>
  )
}
