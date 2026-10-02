import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type Vec2,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { moons, type Point } from '../../_shared/datasets'
import { clusterSeries } from '../../_shared/groups'
import { dbscan } from './dbscan'

/** Two moons plus a sprinkling of uniform background noise. */
function data(): Point[] {
  const { points } = moons(160, 0.07, 7)
  const r = rng(21)
  const noise: Point[] = Array.from({ length: 14 }, () => [-1.3 + 3.6 * r.uniform(), -0.9 + 2.3 * r.uniform()])
  return [...points, ...noise]
}

const CIRCLE = Array.from({ length: 65 }, (_, i) => (2 * Math.PI * i) / 64)

export function DbscanExplorer() {
  const points = useMemo(() => data(), [])
  const eps = useParam(0.25, { min: 0.05, max: 0.6, step: 0.01 })
  const [minPts, setMinPts] = useState(5)
  const [probe, setProbe] = useState<Vec2>([0.5, 0.35])
  const result = useMemo(() => dbscan(points, eps.value, minPts), [points, eps.value, minPts])

  const coreCount = result.core.filter(Boolean).length
  const noiseCount = result.labels.filter((l) => l < 0).length
  const inBall = points.filter((p) => (p[0] - probe[0]) ** 2 + (p[1] - probe[1]) ** 2 <= eps.value ** 2).length
  const series: XYSeries[] = [
    ...clusterSeries(points, result.labels, 'noise'),
    {
      name: 'ε-ball',
      type: 'line',
      x: CIRCLE.map((t) => probe[0] + eps.value * Math.cos(t)),
      y: CIRCLE.map((t) => probe[1] + eps.value * Math.sin(t)),
      emphasis: true,
      dashed: true,
    },
  ]
  const handles: Handle[] = [{ kind: 'point', at: probe, label: 'ε-ball centre', onDrag: setProbe }]

  return (
    <Interactive
      title="DBSCAN on two moons with background noise"
      caption="A point is a core point when its ε-ball holds at least minPts points, itself included. Clusters are the core points linked through overlapping balls, plus the border points they reach; everything else is noise. Drag the dashed ε-ball to count the points around any location. Too small an ε breaks the moons into fragments and labels most points noise; too large an ε merges the moons."
      controls={
        <>
          <ParamSlider label="ε (neighbourhood radius)" param={eps} />
          <ParamSlider label="minPts" value={minPts} onChange={setMinPts} min={2} max={15} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="clusters" value={result.clusters} />
          <Readout label="core points" value={coreCount} />
          <Readout label="border points" value={points.length - coreCount - noiseCount} />
          <Readout label="noise" value={noiseCount} />
          <Readout label="points in the ε-ball" value={`${inBall} (ε = ${formatNumber(eps.value)})`} />
        </>
      }
    >
      <XYChart
        equalAspect
        xRange={[-1.5, 2.5]}
        yRange={[-1, 1.5]}
        xLabel="x₁"
        yLabel="x₂"
        handles={handles}
        series={series}
      />
    </Interactive>
  )
}
