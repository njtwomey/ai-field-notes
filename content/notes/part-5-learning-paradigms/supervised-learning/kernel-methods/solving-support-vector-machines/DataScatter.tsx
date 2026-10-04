import { useMemo } from 'react'
import { Plot, Points, useAxis } from 'aifn-render'
import { BOX } from './plot'
import { X, Y } from './solver'

/** The six training points, coloured and shaped by class, with no fitted model. */
export function DataScatter() {
  const series = useMemo(
    () =>
      [
        {
          name: 'points',
          x: X.map((p) => p[0]),
          y: X.map((p) => p[1]),
          group: Y.map((v) => (v > 0 ? 1 : 0)),
          groupNames: ['class −1', 'class +1'],
        },
      ] as const,
    [],
  )
  const xAxis = useAxis({ label: 'x₁', range: BOX.x })
  const yAxis = useAxis({ label: 'x₂', range: BOX.y, equal: xAxis })
  return (
    <Plot
      x={xAxis}
      y={yAxis}
      ariaLabel={
        'Scatter plot of the six training points: class −1 at (1, 1), (2, 1), (1, 2); class +1 at (2, 2), (4, 3), (4, 1)'
      }
    >
      <Points {...series[0]} />
    </Plot>
  )
}
