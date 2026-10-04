import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { avgPathLength, buildForest, isolationScore, type Box, type Pt } from './iforest'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const REGION: Box = { x0: -3, x1: 8, y0: -3, y1: 7 }
const X_RANGE: [number, number] = [REGION.x0, REGION.x1]
const Y_RANGE: [number, number] = [REGION.y0, REGION.y1]
const GX = toFlat(linspace(REGION.x0, REGION.x1, 40))
const GY = toFlat(linspace(REGION.y0, REGION.y1, 36))

/** Two Gaussian clusters of different spread and a dozen points scattered uniformly over the plane. */
const POINTS: Pt[] = (() => {
  const g = stream(11)
  const pts: Pt[] = []
  for (let i = 0; i < 150; i++) pts.push([0.6 * normal(g), 0.6 * normal(g)])
  for (let i = 0; i < 130; i++) pts.push([4 + normal(g), 3 + normal(g)])
  for (let i = 0; i < 12; i++) pts.push([-3 + 11 * uniform(g), -3 + 10 * uniform(g)])
  return pts
})()

const OVERLAY = [{ name: 'data', x: POINTS.map((p) => p[0]), y: POINTS.map((p) => p[1]) }] as const

export function IsolationForestFigure() {
  const state = useFigureState({
    nTrees: int(100, { min: 1, max: 200, step: 1, label: 'trees t', format: (v) => String(v) }),
    psi: int(256, { min: 4, max: 256, step: 4, label: 'subsample size ψ', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
    px: slider(REGION.x0, REGION.x1, 2.5, { step: 0.05, onChart: true }),
    py: slider(REGION.y0, REGION.y1, -1.5, { step: 0.05, onChart: true }),
  })

  const forest = useMemo(
    () => buildForest(POINTS, state.nTrees, state.psi, state.seed, REGION),
    [state.nTrees, state.psi, state.seed],
  )
  const z = useMemo(() => GY.map((y) => GX.map((x) => isolationScore(forest, [x, y]).score)), [forest])

  const probe: Pt = [state.px, state.py]
  const probeScore = isolationScore(forest, probe)

  const treeSeries = useMemo(
    () =>
      [
        { name: 'all data', x: POINTS.map((p) => p[0]), y: POINTS.map((p) => p[1]), muted: true },
        {
          name: 'subsample of tree 1',
          x: forest.firstSample.map((i) => POINTS[i][0]),
          y: forest.firstSample.map((i) => POINTS[i][1]),
          slot: 0,
        },
      ] as const,
    [forest],
  )

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const xAxis2 = useAxis({ label: 'x₁', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'x₂', range: Y_RANGE })
  return (
    <Figure
      title="Isolation forest scores"
      state={state}
      caption="Left: the anomaly score s(x) = 2^(−E[h(x)]/c(ψ)) of a new point at each place; darker is more anomalous. Right: the cuts of the first tree, grown on its subsample of ψ points and stopped at depth ⌈log₂ ψ⌉. Points in sparse regions sit in large cells that few cuts reach. Drag the probe to read its mean path length. One tree gives a blocky, noisy score; averaging a hundred smooths it. Lowering ψ to 32 changes the map little."

      readouts={
        <>
          <Readout label="probe mean path length E[h]" value={formatNumber(probeScore.meanPath)} />
          <Readout label="c(ψ)" value={formatNumber(avgPathLength(forest.psi))} />
          <Readout label="probe score s" value={formatNumber(probeScore.score)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={360}>
          <Raster x={GX} y={GY} z={z} range={[0.35, 0.8]} valueLabel={'score s'} />
          <Points {...OVERLAY[0]} live />
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
        <Plot x={xAxis2} y={yAxis2} height={360}>
          <Points {...treeSeries[0]} />
          <Points {...treeSeries[1]} />
          <Segments segments={forest.firstCuts} />
        </Plot>
      </div>
    </Figure>
  )
}
