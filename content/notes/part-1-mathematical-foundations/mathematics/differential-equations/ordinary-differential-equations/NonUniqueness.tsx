import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const T = toFlat(linspace(0, 4, 161))
const FAMILY = [0.5, 1.5, 2.5, 3.5]

/** x(t) = 0 until time c, then (t − c)²/4: every such curve solves ẋ = √x with x(0) = 0. */
const solution = (c: number) => T.map((t) => (t <= c ? 0 : (t - c) ** 2 / 4))

export function NonUniqueness() {
  const state = useFigureState({
    c: slider(0, 4, 1, { step: 0.05, label: 'departure time c' }),
  })
  const { c, set } = state
  const series = useMemo<SeriesSpec[]>(
    () => [
      ...FAMILY.map((f) => ({ name: 'other solutions', type: 'line' as const, x: T, y: solution(f), muted: true })),
      { name: 'x(t) = 0', type: 'line', x: T, y: T.map(() => 0), slot: 1, dashed: true },
      { name: 'solution leaving 0 at t = c', type: 'line', x: T, y: solution(c), slot: 0 },
    ],
    [c],
  )
  const handles = useMemo<Handle[]>(
    () => [{ kind: 'x', at: c, label: 'c', onDrag: (v: number) => set('c', v) }],
    [c, set],
  )
  const t = 3
  const x = t <= c ? 0 : (t - c) ** 2 / 4
  const xAxis = useAxis({ label: 't', range: [0, 4] })
  const yAxis = useAxis({ label: 'x', range: [-0.2, 4] })
  return (
    <Figure
      title="Many solutions from one initial value"
      state={state}
      caption="Every curve here satisfies ẋ = √x and starts at x(0) = 0. The solution may rest at 0 for any length of time c and then leave along (t − c)²/4. Drag the vertical line, or use the slider, to choose c. The right side √x has an infinite slope at x = 0, so it is not Lipschitz there and uniqueness fails."

      readouts={
        <>
          <Readout label="x(3)" value={formatNumber(x)} />
          <Readout label="ẋ(3)" value={formatNumber(t <= c ? 0 : (t - c) / 2)} />
          <Readout label="√x(3)" value={formatNumber(Math.sqrt(x))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
