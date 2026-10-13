import { useMemo } from 'react'
import {
  Area,
  Figure,
  formatNumber,
  Handle,
  MARKER_SHAPES,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import {
  expectedHypervolumeImprovement,
  hypervolume,
  hypervolumeImprovement,
  paretoFront,
} from 'aifn-methods/optimisation/multi-objective'
import { linspace, toArray, toFlat } from 'aifn-compute/foundation/tensor'

/** A two-field handle is a point handle, which takes a marker symbol. */
const asPoint = <H extends { kind: string }>(h: H) => h as Extract<H, { kind: 'point' }>

/** Evaluated outcomes (validation error, latency in ms): the worked example's front and three dominated points. */
const EVALUATED: [number, number][] = [
  [0.1, 80],
  [0.15, 40],
  [0.25, 20],
  [0.18, 60],
  [0.27, 45],
  [0.12, 95],
]
const X_RANGE = [0, 0.35] as const
const Y_RANGE = [0, 120] as const
/** Cells per side of the EHVI map. */
const CELLS = 71

type Pt = readonly [number, number]

/**
 * The region a two-objective front dominates inside the reference box, as an area between the staircase (base) and the
 * top edge r₂: drawing only, the volume itself comes from `hypervolume`.
 */
function staircase(front: readonly Pt[], r: Pt) {
  const x: number[] = []
  const base: number[] = []
  front.forEach(([a, b], i) => {
    const next = i + 1 < front.length ? front[i + 1][0] : r[0]
    x.push(a, next)
    base.push(b, b)
  })
  return { x, base, top: x.map(() => r[1]) }
}

/**
 * The part of the box [q, r] that the front does not already dominate: for each x between q₁ and r₁, from q₂ up to
 * the lowest second objective among front points to the left of x (or r₂). Drawing only; its area is the engine's
 * `hypervolumeImprovement`.
 */
function newRegion(front: readonly Pt[], q: Pt, r: Pt) {
  const cuts = [q[0], ...front.map((p) => p[0]).filter((v) => v > q[0] && v < r[0]), r[0]]
  const ceiling = (x: number) => Math.min(r[1], ...front.filter((p) => p[0] <= x).map((p) => p[1]))
  const x: number[] = []
  const top: number[] = []
  for (let i = 0; i + 1 < cuts.length; i++) {
    const h = Math.max(q[1], ceiling(cuts[i]))
    x.push(cuts[i], cuts[i + 1])
    top.push(h, h)
  }
  return { x, top, base: x.map(() => q[1]) }
}

/**
 * A two-objective front, the region it dominates up to a reference point, and the hypervolume a new outcome would add;
 * optionally the expected hypervolume improvement of a candidate whose predicted outcome has its mean at each point.
 */
export function HypervolumeImprovement() {
  const state = useFigureState({
    q1: slider(0.02, 0.34, 0.12, { step: 0.005, onChart: true, label: 'candidate error' }),
    q2: slider(2, 118, 50, { step: 1, onChart: true, label: 'candidate latency' }),
    r1: slider(0.2, 0.34, 0.3, { step: 0.005, onChart: true, label: 'reference error' }),
    r2: slider(60, 118, 100, { step: 1, onChart: true, label: 'reference latency' }),
    ehvi: setting(false, { label: 'show expected hypervolume improvement' }),
    sd1: slider(0.005, 0.06, 0.02, { step: 0.005, label: 'posterior sd of error', when: (v) => Boolean(v.ehvi) }),
    sd2: slider(1, 30, 10, { step: 1, label: 'posterior sd of latency (ms)', when: (v) => Boolean(v.ehvi) }),
  })
  const { q1, q2, r1, r2, ehvi, sd1, sd2 } = state
  const r: Pt = [r1, r2]
  const q: Pt = [q1, q2]

  const front = useMemo(() => {
    const { indices, front } = paretoFront(EVALUATED)
    return { indices, points: toArray(front) as [number, number][] }
  }, [])
  const dominated = EVALUATED.filter((_, i) => !front.indices.includes(i))
  // Only points strictly inside the reference box dominate any of it.
  const inBox = front.points.filter(([a, b]) => a < r1 && b < r2)

  const hv = hypervolume(front.points, [r1, r2])
  const hvi = toFlat(hypervolumeImprovement(front.points, [r1, r2], [[q1, q2]]))[0]
  const ehviHere = toFlat(expectedHypervolumeImprovement([[q1, q2]], [[sd1, sd2]], front.points, [r1, r2]))[0]

  const map = useComputed(() => {
    if (!ehvi) return null
    const xs = toFlat(linspace(X_RANGE[0], X_RANGE[1], CELLS))
    const ys = toFlat(linspace(Y_RANGE[0], Y_RANGE[1], CELLS))
    const means = ys.flatMap((y) => xs.map((x) => [x, y]))
    const value = toFlat(
      expectedHypervolumeImprovement(
        means,
        means.map(() => [sd1, sd2]),
        front.points,
        [r1, r2],
      ),
    )
    return { xs, ys, z: ys.map((_, i) => Array.from(value.slice(i * CELLS, (i + 1) * CELLS))) }
  }, [ehvi, sd1, sd2, r1, r2, front])

  const stairs = staircase(inBox, r)
  const gain = hvi > 0 ? newRegion(inBox, q, r) : null

  const x = useAxis({ label: 'validation error f₁', range: X_RANGE })
  const y = useAxis({ label: 'latency f₂ (ms)', range: Y_RANGE })
  return (
    <Figure
      title="Hypervolume and the improvement a new outcome adds"
      state={state}
      caption="Six evaluated configurations in objective space, error against latency, both minimised. The three on the Pareto front (blue) dominate the shaded staircase up to the reference point r (the cross); the other three (grey) add nothing. Drag the candidate outcome q (ink) and the reference point. The orange region is the part of the box between q and r that the front does not already dominate, and its area is the hypervolume improvement. At its starting place, q = (0.12, 50), the hypervolume grows from 11.0 to 11.9. Moved behind the staircase, q adds nothing. Switch on the expected hypervolume improvement to colour each point by the EHVI of a candidate predicted there with the given posterior standard deviations: it is largest just beyond the front, and stays positive behind it, where the posterior still reaches past the staircase."
      readouts={
        <>
          <Readout label="hypervolume HV" value={formatNumber(hv)} />
          <Readout label="with q added" value={formatNumber(hv + hvi)} />
          <Readout label="improvement HVI(q)" value={formatNumber(hvi)} />
          {ehvi && <Readout label="EHVI of a candidate predicted at q" value={formatNumber(ehviHere)} />}
          <Readout label="front" value={`${front.points.length} of ${EVALUATED.length} evaluations`} />
        </>
      }
    >
      <Plot x={x} y={y} height={420}>
        {map.value && (
          <Raster
            x={map.value.xs}
            y={map.value.ys}
            z={map.value.z}
            valueLabel="EHVI"
            fillOpacity={0.75}
            stale={map.stale}
          />
        )}
        {inBox.length > 0 && (
          <Area
            name="dominated region"
            x={stairs.x}
            y={stairs.top}
            base={stairs.base}
            slot={0}
            opacity={0.22}
            line={false}
          />
        )}
        {gain && (
          <Area name="added by q" x={gain.x} y={gain.top} base={gain.base} slot={1} opacity={0.45} line={false} />
        )}
        <Points name="dominated evaluations" x={dominated.map((p) => p[0])} y={dominated.map((p) => p[1])} muted />
        <Points
          name="Pareto front"
          x={front.points.map((p) => p[0])}
          y={front.points.map((p) => p[1])}
          labels={front.points.map((_, i) => `p${'₁₂₃₄₅₆'[i]}`)}
          slot={0}
          size={10}
        />
        <Handle {...asPoint(state.handle(['r1', 'r2'], { label: 'r' }))} symbol={MARKER_SHAPES[4]} />
        <Handle {...state.handle(['q1', 'q2'], { label: 'q' })} />
      </Plot>
    </Figure>
  )
}
