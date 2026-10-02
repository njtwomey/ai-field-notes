import { useMemo } from 'react'
import { XYChart, type XYSeries } from 'aifn-render'
import { BOX } from './plot'
import { X, Y } from './solver'

/** The six training points, coloured and shaped by class, with no fitted model. */
export function DataScatter() {
  const series = useMemo(
    (): XYSeries[] => [
      {
        name: 'points',
        type: 'scatter',
        x: X.map((p) => p[0]),
        y: X.map((p) => p[1]),
        group: Y.map((v) => (v > 0 ? 1 : 0)),
        groupNames: ['class −1', 'class +1'],
      },
    ],
    [],
  )
  return (
    <XYChart
      series={series}
      xRange={BOX.x}
      yRange={BOX.y}
      equalAspect
      xLabel="x₁"
      yLabel="x₂"
      ariaLabel="Scatter plot of the six training points: class −1 at (1, 1), (2, 1), (1, 2); class +1 at (2, 2), (4, 3), (4, 1)"
    />
  )
}
