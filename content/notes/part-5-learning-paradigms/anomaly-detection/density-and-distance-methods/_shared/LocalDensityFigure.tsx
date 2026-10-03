import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { linspace } from '@/lib/math'
import { fitDensity, scorePoint, twoDensityData, type Pt } from './localDensity'

type Score = 'knn' | 'lof'

const X = linspace(-3, 10, 40)
const Y = linspace(-3.5, 8, 36)
const DATA = twoDensityData()
const PLANTED = new Set(DATA.planted)
const SCORE_OPTIONS = [
  { value: 'knn', label: 'k-NN distance' },
  { value: 'lof', label: 'LOF' },
] as const

/**
 * Anomaly scores over the plane for a tight and a diffuse cluster. The k-NN distance marks the whole diffuse cluster
 * as unusual; LOF, which divides by the neighbours' own density, is near 1 inside both clusters.
 */
export function LocalDensityFigure({ initialScore = 'lof' }: { initialScore?: Score }) {
  const [score, setScore] = useState<Score>(initialScore)
  const k = useParam(10, { min: 2, max: 40, step: 1 })
  const px = useParam(1.6, { min: -3, max: 10, step: 0.05 })
  const py = useParam(1.4, { min: -3.5, max: 8, step: 0.05 })

  const model = useMemo(() => fitDensity(DATA.points, k.value), [k.value])

  const grid = useMemo(() => {
    const knn: number[][] = []
    const lof: number[][] = []
    for (const y of Y) {
      const rk: number[] = []
      const rl: number[] = []
      for (const x of X) {
        const s = scorePoint(model, [x, y])
        rk.push(s.knn)
        rl.push(s.lof)
      }
      knn.push(rk)
      lof.push(rl)
    }
    // The k-NN colour scale stops at the 90th percentile of the grid so that the clusters' interiors keep contrast.
    const sorted = knn.flat().sort((a, b) => a - b)
    return { knn, lof, knnTop: sorted[Math.floor(0.9 * sorted.length)] }
  }, [model])

  const trainScores = score === 'knn' ? model.knn : model.lof
  const top = useMemo(
    () =>
      trainScores
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s)
        .slice(0, PLANTED.size)
        .map((o) => o.i),
    [trainScores],
  )
  const hits = top.filter((i) => PLANTED.has(i)).length

  const overlay = useMemo((): HeatmapOverlay[] => {
    const pts = DATA.points
    return [
      {
        name: 'points',
        type: 'scatter',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        group: pts.map((_, i) => (PLANTED.has(i) ? 1 : 0)),
        groupNames: ['data', 'planted outliers'],
      },
      {
        name: `top ${PLANTED.size} scores`,
        type: 'scatter',
        x: top.map((i) => pts[i][0]),
        y: top.map((i) => pts[i][1]),
        emphasis: true,
      },
    ]
  }, [top])

  const probe: Pt = [px.value, py.value]
  const probeScore = scorePoint(model, probe)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: probe,
      label: 'probe',
      onDrag: ([a, b]) => {
        px.set(a)
        py.set(b)
      },
    },
  ]
  const lo = score === 'knn' ? 0 : 1
  const hi = score === 'knn' ? grid.knnTop : 3

  return (
    <Interactive
      title="Global distance against local density"
      caption="A tight cluster at the origin, a diffuse cluster to the upper right and three planted outliers (two just outside the tight cluster, one far from both). The shading is the score a new point would get at each place; darker is more anomalous. Ink diamonds mark the three training points with the highest scores. Drag the probe point to read its scores. The k-NN distance darkens the diffuse cluster's fringe and misses the two local outliers; LOF stays near 1 inside both clusters and ranks all three planted points first."
      controls={
        <>
          <ParamChoice label="score" value={score} onChange={setScore} options={SCORE_OPTIONS} />
          <ParamSlider label="neighbours k" param={k} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="planted outliers in the top 3" value={`${hits} of ${PLANTED.size}`} />
          <Readout label="probe k-NN distance" value={formatNumber(probeScore.knn)} />
          <Readout label="probe LOF" value={formatNumber(probeScore.lof)} />
        </>
      }
    >
      <Heatmap
        x={X}
        y={Y}
        z={score === 'knn' ? grid.knn : grid.lof}
        range={[lo, hi]}
        overlay={overlay}
        handles={handles}
        xLabel="x₁"
        yLabel="x₂"
        valueLabel={score === 'knn' ? 'k-NN distance' : 'LOF'}
        height={420}
      />
    </Interactive>
  )
}
