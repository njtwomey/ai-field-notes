import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
  type Vec2,
} from 'aifn-render'
import { moons, type Point } from '../../_shared/datasets'
import { clusterSeries } from '../../_shared/groups'
import { dbscan } from './dbscan'
import { stream, uniform } from 'aifn-compute/foundation/random'

/** Two moons plus a sprinkling of uniform background noise. */
function data(): Point[] {
  const { points } = moons(160, 0.07, 7)
  const r = stream(21)
  const noise: Point[] = Array.from({ length: 14 }, () => [-1.3 + 3.6 * uniform(r), -0.9 + 2.3 * uniform(r)])
  return [...points, ...noise]
}

const CIRCLE = Array.from({ length: 65 }, (_, i) => (2 * Math.PI * i) / 64)

export function DbscanExplorer() {
  const points = useMemo(() => data(), [])
  const state = useFigureState({
    eps: float(0.25, { min: 0.05, max: 0.6, step: 0.01, label: 'ε (neighbourhood radius)' }),
    minPts: int(5, { min: 2, max: 15, step: 1, label: 'minPts' }),
  })
  const [probe, setProbe] = useState<Vec2>([0.5, 0.35])
  const result = useMemo(() => dbscan(points, state.eps, state.minPts), [points, state.eps, state.minPts])

  const coreCount = result.core.filter(Boolean).length
  const noiseCount = result.labels.filter((l) => l < 0).length
  const inBall = points.filter((p) => (p[0] - probe[0]) ** 2 + (p[1] - probe[1]) ** 2 <= state.eps ** 2).length
  const series: SeriesSpec[] = [
    ...clusterSeries(points, result.labels, 'noise'),
    {
      name: 'ε-ball',
      type: 'line',
      x: CIRCLE.map((t) => probe[0] + state.eps * Math.cos(t)),
      y: CIRCLE.map((t) => probe[1] + state.eps * Math.sin(t)),
      emphasis: true,
      dashed: true,
    },
  ]

  const xAxis = useAxis({ label: 'x₁', range: [-1.5, 2.5] })
  const yAxis = useAxis({ label: 'x₂', range: [-1, 1.5], equal: xAxis })
  return (
    <Figure
      title="DBSCAN on two moons with background noise"
      state={state}
      caption="A point is a core point when its ε-ball holds at least minPts points, itself included. Clusters are the core points linked through overlapping balls, plus the border points they reach; everything else is noise. Drag the dashed ε-ball to count the points around any location. Too small an ε breaks the moons into fragments and labels most points noise; too large an ε merges the moons."

      readouts={
        <>
          <Readout label="clusters" value={result.clusters} />
          <Readout label="core points" value={coreCount} />
          <Readout label="border points" value={points.length - coreCount - noiseCount} />
          <Readout label="noise" value={noiseCount} />
          <Readout label="points in the ε-ball" value={`${inBall} (ε = ${formatNumber(state.eps)})`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {seriesLayers(series)}
        <Handle kind="point" at={probe} label="ε-ball centre" onDrag={setProbe} />
      </Plot>
    </Figure>
  )
}
