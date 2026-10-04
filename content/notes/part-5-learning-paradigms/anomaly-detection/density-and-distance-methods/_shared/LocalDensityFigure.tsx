import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { fitDensity, scorePoint, twoDensityData, type Pt } from './localDensity'
import { linspace, toFlat } from 'aifn/foundation/tensor'

type Score = 'knn' | 'lof'

const X = toFlat(linspace(-3, 10, 40))
const Y = toFlat(linspace(-3.5, 8, 36))
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
  const state = useFigureState({
    score: choice<Score>(SCORE_OPTIONS, initialScore, { label: 'score' }),
    k: int(10, { min: 2, max: 40, step: 1, label: 'neighbours k', format: (v) => String(v) }),
    px: slider(-3, 10, 1.6, { step: 0.05, onChart: true }),
    py: slider(-3.5, 8, 1.4, { step: 0.05, onChart: true }),
  })

  const model = useMemo(() => fitDensity(DATA.points, state.k), [state.k])

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

  const trainScores = state.score === 'knn' ? model.knn : model.lof
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

  const overlay = useMemo(() => {
    const pts = DATA.points
    return [
      {
        name: 'points',
        x: pts.map((p) => p[0]),
        y: pts.map((p) => p[1]),
        group: pts.map((_, i) => (PLANTED.has(i) ? 1 : 0)),
        groupNames: ['data', 'planted outliers'],
      },
      {
        name: `top ${PLANTED.size} scores`,
        x: top.map((i) => pts[i][0]),
        y: top.map((i) => pts[i][1]),
        emphasis: true,
      },
    ] as const
  }, [top])

  const probe: Pt = [state.px, state.py]
  const probeScore = scorePoint(model, probe)
  const lo = state.score === 'knn' ? 0 : 1
  const hi = state.score === 'knn' ? grid.knnTop : 3

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="Global distance against local density"
      state={state}
      caption="A tight cluster at the origin, a diffuse cluster to the upper right and three planted outliers (two just outside the tight cluster, one far from both). The shading is the score a new point would get at each place; darker is more anomalous. Ink diamonds mark the three training points with the highest scores. Drag the probe point to read its scores. The k-NN distance darkens the diffuse cluster's fringe and misses the two local outliers; LOF stays near 1 inside both clusters and ranks all three planted points first."

      readouts={
        <>
          <Readout label="planted outliers in the top 3" value={`${hits} of ${PLANTED.size}`} />
          <Readout label="probe k-NN distance" value={formatNumber(probeScore.knn)} />
          <Readout label="probe LOF" value={formatNumber(probeScore.lof)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={420}>
        <Raster
          x={X}
          y={Y}
          z={state.score === 'knn' ? grid.knn : grid.lof}
          range={[lo, hi]}
          valueLabel={state.score === 'knn' ? 'k-NN distance' : 'LOF'}
        />
        <Points {...overlay[0]} live />
        <Points {...overlay[1]} live />
        <Handle
          kind="point"
          at={probe}
          label="probe"
          onDrag={([a, b]) => {
            state.set('px', a)
            state.set('py', b)
          }}
        />
      </Plot>
    </Figure>
  )
}
