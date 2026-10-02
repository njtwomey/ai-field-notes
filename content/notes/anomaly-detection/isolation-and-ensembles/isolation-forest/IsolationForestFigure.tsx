import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { avgPathLength, buildForest, isolationScore, type Box, type Pt } from './iforest'

const REGION: Box = { x0: -3, x1: 8, y0: -3, y1: 7 }
const X_RANGE: [number, number] = [REGION.x0, REGION.x1]
const Y_RANGE: [number, number] = [REGION.y0, REGION.y1]
const GX = linspace(REGION.x0, REGION.x1, 40)
const GY = linspace(REGION.y0, REGION.y1, 36)

/** Two Gaussian clusters of different spread and a dozen points scattered uniformly over the plane. */
const POINTS: Pt[] = (() => {
  const g = rng(11)
  const pts: Pt[] = []
  for (let i = 0; i < 150; i++) pts.push([0.6 * g.normal(), 0.6 * g.normal()])
  for (let i = 0; i < 130; i++) pts.push([4 + g.normal(), 3 + g.normal()])
  for (let i = 0; i < 12; i++) pts.push([-3 + 11 * g.uniform(), -3 + 10 * g.uniform()])
  return pts
})()

const OVERLAY: HeatmapOverlay[] = [
  { name: 'data', type: 'scatter', x: POINTS.map((p) => p[0]), y: POINTS.map((p) => p[1]) },
]

export function IsolationForestFigure() {
  const nTrees = useParam(100, { min: 1, max: 200, step: 1 })
  const psi = useParam(256, { min: 4, max: 256, step: 4 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const px = useParam(2.5, { min: REGION.x0, max: REGION.x1, step: 0.05 })
  const py = useParam(-1.5, { min: REGION.y0, max: REGION.y1, step: 0.05 })

  const forest = useMemo(
    () => buildForest(POINTS, nTrees.value, psi.value, seed.value, REGION),
    [nTrees.value, psi.value, seed.value],
  )
  const z = useMemo(() => GY.map((y) => GX.map((x) => isolationScore(forest, [x, y]).score)), [forest])

  const probe: Pt = [px.value, py.value]
  const probeScore = isolationScore(forest, probe)
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

  const treeSeries = useMemo(
    (): XYSeries[] => [
      { name: 'all data', type: 'scatter', x: POINTS.map((p) => p[0]), y: POINTS.map((p) => p[1]), muted: true },
      {
        name: 'subsample of tree 1',
        type: 'scatter',
        x: forest.firstSample.map((i) => POINTS[i][0]),
        y: forest.firstSample.map((i) => POINTS[i][1]),
        slot: 0,
      },
    ],
    [forest],
  )

  return (
    <Interactive
      title="Isolation forest scores"
      caption="Left: the anomaly score s(x) = 2^(−E[h(x)]/c(ψ)) of a new point at each place; darker is more anomalous. Right: the cuts of the first tree, grown on its subsample of ψ points and stopped at depth ⌈log₂ ψ⌉. Points in sparse regions sit in large cells that few cuts reach. Drag the probe to read its mean path length. One tree gives a blocky, noisy score; averaging a hundred smooths it. Lowering ψ to 32 changes the map little."
      controls={
        <>
          <ParamSlider label="trees t" param={nTrees} format={(v) => String(v)} />
          <ParamSlider label="subsample size ψ" param={psi} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="probe mean path length E[h]" value={formatNumber(probeScore.meanPath)} />
          <Readout label="c(ψ)" value={formatNumber(avgPathLength(forest.psi))} />
          <Readout label="probe score s" value={formatNumber(probeScore.score)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={GX}
          y={GY}
          z={z}
          range={[0.35, 0.8]}
          overlay={OVERLAY}
          handles={handles}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="score s"
          height={360}
        />
        <XYChart
          series={treeSeries}
          segments={forest.firstCuts}
          xRange={X_RANGE}
          yRange={Y_RANGE}
          xLabel="x₁"
          yLabel="x₂"
          height={360}
        />
      </div>
    </Interactive>
  )
}
