import { useMemo, useState } from 'react'
import {
  Figure,
  float,
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
import {
  rhoPathDistances,
  rhoPathExtend,
  rhoPathNodes,
  rhoQueryNeighbours,
} from 'aifn-methods/learning/semi-supervised'
import { centredMoons, gridField, nearestOfClass, predictionGrid } from '../_shared/figure-helpers'

const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.8, 1.8]
const Y_RANGE: [number, number] = [-1.3, 1.3]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
const START: [[number, number], [number, number]] = [
  [-1.5, -0.25],
  [1.5, 0.25],
]

/**
 * The ρ-path distance on two moons with one labelled point per class: the nearest labelled point under the
 * distance, and the cheapest paths from a probe point to each labelled point.
 */
export function DensityPathDistance() {
  const state = useFigureState({
    rho: slider(0, 20, 8, { step: 0.5, label: 'softening ρ' }),
    noise: float(0.1, { ge: 0, le: 0.4, label: 'noise sd', suggestions: [0.05, 0.1, 0.2, 0.3] }),
    seed: int(3, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { rho, noise, seed } = state
  const [anchors, setAnchors] = useState(START)
  const [probe, setProbe] = useState<[number, number]>([0.5, -0.25])

  const data = useMemo(() => centredMoons(seed, noise), [seed, noise])
  const labelled = useMemo(() => [0, 1].map((c) => nearestOfClass(data, c, anchors[c])), [data, anchors])
  // The full graph and the grid's neighbours do not depend on ρ: compute them once per data set.
  const base = useMemo(() => rhoPathDistances(data.rows, labelled, { rho: 0 }), [data, labelled])
  const near = useMemo(() => rhoQueryNeighbours(data.rows, GRID.rows), [data])

  const result = useMemo(
    () => rhoPathDistances(data.rows, labelled, { rho, graph: base.graph }),
    [data, labelled, rho, base],
  )
  const field = useMemo(() => {
    const ext = rhoPathExtend(data.rows, GRID.rows, result, near)
    const rel = ext.distances[0].map((d0, q) => {
      const d1 = ext.distances[1][q]
      return (d0 - d1) / (d0 + d1)
    })
    return gridField(rel, NX, NY)
  }, [data, result, near])

  const accuracy = useMemo(() => {
    let right = 0
    let count = 0
    data.rows.forEach((_, i) => {
      if (labelled.includes(i)) return
      count++
      const c = result.distances[0][i] <= result.distances[1][i] ? 0 : 1
      if (c === data.y[i]) right++
    })
    return right / count
  }, [data, labelled, result])

  const probePaths = useMemo(() => {
    const ext = rhoPathExtend(data.rows, [probe], result)
    return [0, 1].map((s) => {
      const nodes = rhoPathNodes(result, s, ext.entry[s][0])
      const pts = [...nodes.map((v) => data.rows[v] as [number, number]), probe]
      return {
        distance: ext.distances[s][0],
        segments: pts.slice(1).map((p, k) => ({ from: pts[k] as [number, number], to: p as [number, number] })),
      }
    })
  }, [data, probe, result])

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const byClass = [0, 1].map((c) => data.rows.map((_, i) => i).filter((i) => data.y[i] === c && !labelled.includes(i)))
  const winner = probePaths[0].distance <= probePaths[1].distance ? 0 : 1

  return (
    <Figure
      title="A density-sensitive distance on two moons"
      state={state}
      caption="Two hundred points with one labelled point per class (large markers). Every point is classified by its nearest labelled point under the ρ-path distance on the full graph of the points: an edge of length d costs e^(ρd) − 1 and the distance is ln(1 + cheapest path cost)/ρ. The shading is (D₀ − D₁)/(D₀ + D₁), with D₀ and D₁ the distances to the two labelled points, and the boundary in ink. Drag either labelled point (it snaps to the nearest point of its class) or the probe (the ink handle): the coloured polylines are the probe's cheapest paths to the two labelled points. At ρ = 0 the distance is Euclidean, the paths are straight lines, and the boundary is the perpendicular bisector of the labels. As ρ grows, long edges become expensive, the paths hop along each moon, and the boundary moves into the gap between the moons. With noise near 0.3 the moons touch, short edges cross between them, and no ρ separates them."
      readouts={
        <>
          <Readout label="unlabelled accuracy" value={formatNumber(accuracy)} />
          <Readout label="probe: D to label 0" value={formatNumber(probePaths[0].distance)} />
          <Readout label="probe: D to label 1" value={formatNumber(probePaths[1].distance)} />
          <Readout label="probe classified as" value={`class ${winner}`} />
        </>
      }
    >
      <Plot x={xa} y={ya} height={380}>
        <Raster
          x={GRID.gx}
          y={GRID.gy}
          z={field}
          scale="diverging"
          range={[-1, 1]}
          valueLabel="(D₀ − D₁)/(D₀ + D₁)"
          boundary
          fillOpacity={0.45}
        />
        {byClass.map(
          (rows, c) =>
            rows.length > 0 && (
              <Points
                key={c}
                name={`class ${c}, unlabelled`}
                x={rows.map((i) => data.rows[i][0])}
                y={rows.map((i) => data.rows[i][1])}
                slot={c}
                thin
              />
            ),
        )}
        {probePaths.map(
          (p, s) =>
            p.segments.length > 0 && (
              <Segments key={`p${s}`} name={`cheapest path to label ${s}`} segments={p.segments} slot={s} width={2} />
            ),
        )}
        {labelled.map((i, c) => (
          <Points
            key={`l${c}`}
            name={`class ${c}, labelled`}
            x={[data.rows[i][0]]}
            y={[data.rows[i][1]]}
            slot={c}
            emphasis
            size={12}
          />
        ))}
        {labelled.map((i, c) => (
          <Handle
            key={`h${c}`}
            kind="point"
            at={[data.rows[i][0], data.rows[i][1]]}
            slot={c}
            label={`label ${c}`}
            onDrag={(p) => setAnchors((a) => (c === 0 ? [[p[0], p[1]], a[1]] : [a[0], [p[0], p[1]]]))}
          />
        ))}
        <Handle kind="point" at={probe} label="probe" onDrag={(p) => setProbe([p[0], p[1]])} />
      </Plot>
    </Figure>
  )
}
