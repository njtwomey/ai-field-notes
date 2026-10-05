import { useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const GRID = toFlat(linspace(0, 1, 201))
const START: [number, number][] = [
  [0.7, 0.2],
  [0.35, 0.65],
  [0.15, 0.55],
]
type Contexts = 'uniform' | 'low' | 'high'
/** Context densities on [0, 1]: uniform, Beta(1, 3) and Beta(3, 1). */
const DENSITY: Record<Contexts, (x: number) => number> = {
  uniform: () => 1,
  low: (x) => 3 * (1 - x) ** 2,
  high: (x) => 3 * x * x,
}

/**
 * Three arms whose mean reward is linear in a scalar context x; their ends are draggable. The upper envelope is the
 * best policy; the value lost by any single arm is what a context-free bandit gives up.
 */
export function ContextValue() {
  const [ends, setEnds] = useState(START)
  const state = useFigureState({
    contexts: choice<Contexts>(
      [
        { value: 'uniform', label: 'uniform' },
        { value: 'low', label: 'mostly low x' },
        { value: 'high', label: 'mostly high x' },
      ],
      'uniform',
      { label: 'context distribution' },
    ),
  })

  const mu = (a: number, x: number) => ends[a][0] + (ends[a][1] - ends[a][0]) * x
  const weights = GRID.map(DENSITY[state.contexts])
  const total = weights.reduce((acc, w) => acc + w, 0)
  const expect = (f: (x: number) => number) => GRID.reduce((acc, x, i) => acc + weights[i] * f(x), 0) / total
  const armValues = ends.map((_, a) => expect((x) => mu(a, x)))
  const bestArm = armValues.indexOf(Math.max(...armValues))
  const envelope = (x: number) => Math.max(...ends.map((_, a) => mu(a, x)))
  const policyValue = expect(envelope)

  const series: SeriesSpec[] = [
    {
      name: 'context density (scaled)',
      type: 'line',
      x: GRID,
      y: weights.map((w) => (0.15 * w) / Math.max(...weights)),
      muted: true,
      area: true,
    },
    ...ends.map((_, a): SeriesSpec => ({
      name: `arm ${a + 1}${a === bestArm ? ' (best single arm)' : ''}`,
      type: 'line',
      x: [0, 1],
      y: [ends[a][0], ends[a][1]],
      slot: a,
    })),
    { name: 'best policy π*(x)', type: 'line', x: GRID, y: GRID.map(envelope), emphasis: true, dashed: true },
  ]
  const handles: Handle[] = ends.flatMap((e, a) =>
    [0, 1].map((side): Handle => ({
      kind: 'point',
      at: [side, e[side]],
      label: `arm ${a + 1}`,
      onDrag: ([, y]) =>
        setEnds((prev) =>
          prev.map((p, b) =>
            b === a
              ? side === 0
                ? [Math.min(1, Math.max(0, Math.round(y * 100) / 100)), p[1]]
                : [p[0], Math.min(1, Math.max(0, Math.round(y * 100) / 100))]
              : p,
          ),
        ),
    })),
  )

  const xAxis = useAxis({ label: 'context x', range: [-0.05, 1.05] })
  const yAxis = useAxis({ label: 'expected reward μ(x, a)', range: [0, 1] })
  return (
    <Figure
      title="What the context is worth"
      state={state}
      caption="Three arms whose expected reward depends on a context x in [0, 1], for example a user's position on some scale. Drag the end points of each line. The dashed upper envelope is the best policy π*(x), which picks the best arm for each context. A bandit that ignores x can at best learn the single arm with the highest average, and loses the difference between the two values in every round, however long it runs. When one arm is best for every x, the context is worthless."

      readouts={
        <>
          {armValues.map((v, a) => (
            <Readout key={a} label={`E[μ(x, ${a + 1})]`} value={formatNumber(v)} />
          ))}
          <Readout label="best single arm, max E[μ]" value={formatNumber(armValues[bestArm])} />
          <Readout label="best policy, E[max μ]" value={formatNumber(policyValue)} />
          <Readout label="lost per round without context" value={formatNumber(policyValue - armValues[bestArm])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
