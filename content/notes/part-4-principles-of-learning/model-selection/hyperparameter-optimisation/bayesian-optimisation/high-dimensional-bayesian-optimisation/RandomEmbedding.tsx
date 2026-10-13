import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { forrester } from 'aifn-methods/data/objectives'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { FixedHeight } from '../_shared/FixedHeight'

/** The Forrester function, rescaled from [0, 1] to the box's first coordinate on [−1, 1]. */
const F = forrester()
const f = (x1: number) => F.value(tensor([(x1 + 1) / 2]))
/** The minimiser of f in box coordinates, 2 × 0.7572 − 1, and the minimum value. */
const X_STAR = 2 * toFlat(F.minima[0])[0] - 1
const F_MIN = F.minimumValue
/** The embedded coordinate y runs over Y = [−√d, √d] with d = 1. */
const Y_POINTS = 201
const MAP_POINTS = 81
const clip = (v: number) => Math.max(-1, Math.min(1, v))

/**
 * Random embedding Bayesian optimisation's picture in two dimensions (Wang et al., 2016): f depends on x₁ only, the
 * embedding is the line x = A y through the origin, and points of the line outside the box are projected onto it.
 */
export function RandomEmbedding() {
  const state = useFigureState({
    a1: slider(-2, 2, 0.9, { onChart: true, label: 'A₁' }),
    a2: slider(-2, 2, 1.2, { onChart: true, label: 'A₂' }),
  })
  const { a1, a2 } = state

  const map = useMemo(() => {
    const axis = Array.from({ length: MAP_POINTS }, (_, i) => -1 + (2 * i) / (MAP_POINTS - 1))
    const row = axis.map(f)
    return { axis, z: axis.map(() => row) }
  }, [])

  const line = useMemo(() => {
    const y = Array.from({ length: Y_POINTS }, (_, i) => -1 + (2 * i) / (Y_POINTS - 1))
    const projected = y.map((v) => [clip(a1 * v), clip(a2 * v)] as const)
    const g = projected.map(([x1]) => f(x1))
    const best = g.reduce((arg, v, i) => (v < g[arg] ? i : arg), 0)
    return { y, projected, g, best }
  }, [a1, a2])

  const reach = Math.abs(a1)
  const x1 = useAxis({ label: 'x₁ (matters)', range: [-2, 2] })
  const x2 = useAxis({ label: 'x₂ (does not matter)', range: [-2, 2], equal: x1 })
  const yAxis = useAxis({ label: 'embedded coordinate y', range: [-1, 1] })
  const gAxis = useAxis({ label: 'g(y) = f(p(A y))', range: [-7, 16] })
  const tip = line.projected[line.best]
  return (
    <Figure
      title="A random line through a box whose objective depends on one coordinate"
      state={state}
      caption="Left: f(x₁, x₂) on the box [−1, 1]², which depends on x₁ only (the Forrester function); its minimisers form the vertical dashed line. The arrow is the embedding matrix A, here a single column; drag its tip. The thin segment is the set A y for y in Y = [−1, 1], and the orange path is that set projected onto the box, the points REMBO actually evaluates. Right: the one-dimensional function g(y) that Bayesian optimisation would search, with its minimum marked against the global minimum (dashed). The embedding contains a minimiser whenever |A₁| ≥ 0.514, the x₁ of the minimiser; a nearly horizontal or long arrow always works, and a nearly vertical one never can."
      readouts={
        <>
          <Readout label="|A₁|" value={formatNumber(reach)} />
          <Readout label="min of g" value={formatNumber(line.g[line.best])} />
          <Readout label="global minimum of f" value={formatNumber(F_MIN)} />
          <Readout label="gap" value={formatNumber(line.g[line.best] - F_MIN)} />
          <Readout label="at y" value={formatNumber(line.y[line.best])} />
        </>
      }
    >
      <FixedHeight height={380}>
        <Plots cols={2}>
          <Plot x={x1} y={x2} title="the box and the embedding">
            <Raster x={map.axis} y={map.axis} z={map.z} valueLabel="f(x)" fillOpacity={0.75} />
            <Segments
              name="the box"
              segments={[
                { from: [-1, -1], to: [1, -1] },
                { from: [1, -1], to: [1, 1] },
                { from: [1, 1], to: [-1, 1] },
                { from: [-1, 1], to: [-1, -1] },
              ]}
              muted
            />
            <Segments name="minimisers of f" segments={[{ from: [X_STAR, -1], to: [X_STAR, 1] }]} emphasis dashed />
            <Segments name="A y, y in Y" segments={[{ from: [-a1, -a2], to: [a1, a2] }]} muted />
            <Curve
              name="projected onto the box"
              x={line.projected.map((p) => p[0])}
              y={line.projected.map((p) => p[1])}
              slot={1}
            />
            <Segments name="A" segments={[{ from: [0, 0], to: [a1, a2] }]} slot={1} />
            <Points name="best point of the embedding" x={[tip[0]]} y={[tip[1]]} emphasis size={9} />
            <Handle {...state.handle(['a1', 'a2'], { label: 'A' })} />
          </Plot>
          <Plot x={yAxis} y={gAxis} title="the function searched">
            <Curve name="global minimum" x={[-1, 1]} y={[F_MIN, F_MIN]} muted dashed />
            <Curve name="g(y)" x={line.y} y={line.g} slot={1} />
            <Points name="minimum of g" x={[line.y[line.best]]} y={[line.g[line.best]]} emphasis size={9} />
          </Plot>
        </Plots>
      </FixedHeight>
    </Figure>
  )
}
