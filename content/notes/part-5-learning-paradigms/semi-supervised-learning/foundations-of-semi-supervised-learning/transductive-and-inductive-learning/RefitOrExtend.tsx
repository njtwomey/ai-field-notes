import { useMemo } from 'react'
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
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { moons } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { harmonicExtension, harmonicLabels, pointAffinity } from 'aifn-compute/graph/propagation'
import { predictionGrid } from '../_shared/models'

const N = 150
const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.8, 1.8]
const Y_RANGE: [number, number] = [-1.3, 1.3]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
/** Two labelled points per class: the points nearest these positions (the tips and middles of the moons). */
const ANCHORS: [number, number, 0 | 1][] = [
  [-1.5, -0.25, 0],
  [-0.5, 0.75, 0],
  [1.5, 0.25, 1],
  [0.5, -0.75, 1],
]

/** Class-1 scores of harmonic label propagation on a k-NN graph; null when the system is singular. */
function propagate(rows: number[][], labels: number[], gamma: number, k: number): number[] | null {
  try {
    const W = pointAffinity(fromRows(rows), { gamma, neighbours: k })
    return (toArray(harmonicLabels(W, labels, { classes: 2 }).scores) as number[][]).map((r) => r[1])
  } catch {
    return null
  }
}

/**
 * Label propagation on two moons with one new point: refitting the transductive solution with the point in the graph,
 * against the inductive extension that scores the point from the existing scores without refitting.
 */
export function RefitOrExtend() {
  const state = useFigureState({
    tx: float(0.6, { ge: X_RANGE[0], le: X_RANGE[1], label: 'new point x₁', onChart: true }),
    ty: float(-0.4, { ge: Y_RANGE[0], le: Y_RANGE[1], label: 'new point x₂', onChart: true }),
    width: float(0.15, { gt: 0, le: 1, label: 'kernel width σ', suggestions: [0.1, 0.15, 0.25, 0.4] }),
    k: int(8, { ge: 2, le: 40, label: 'neighbours k', suggestions: [4, 8, 16] }),
    seed: int(3, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { tx, ty, width, k, seed } = state
  const gamma = 1 / (2 * width * width)

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.1, spacing: 'random' })
    const rows = (toArray(d.x) as number[][]).map(([a, b]) => [a - 0.5, b - 0.25])
    const y = toFlat(d.y)
    const labels = y.map(() => -1)
    for (const [ax, ay, c] of ANCHORS) {
      let best = -1
      let bestD = Infinity
      rows.forEach((r, i) => {
        if (y[i] !== c || labels[i] >= 0) return
        const dd = (r[0] - ax) ** 2 + (r[1] - ay) ** 2
        if (dd < bestD) {
          bestD = dd
          best = i
        }
      })
      labels[best] = c
    }
    return { rows, y, labels, px: rows.map((r) => r[0]), py: rows.map((r) => r[1]) }
  }, [seed])

  // The transductive solution on the original points, and the inductive rule it defines everywhere.
  const base = useMemo(() => {
    const f = propagate(data.rows, data.labels, gamma, k)
    if (!f) return null
    const scores = f.map((p) => [1 - p, p])
    const ext = toArray(
      harmonicExtension(data.rows, scores, GRID.points, { gamma, neighbours: k }).scores,
    ) as number[][]
    return {
      f,
      scores,
      field: Array.from({ length: NY }, (_, r) => Array.from({ length: NX }, (_, c) => ext[r * NX + c][1] - 0.5)),
    }
  }, [data, gamma, k])

  // Refit with the new point in the graph, against the extension formula at the new point.
  const refit = useComputed(() => {
    if (!base) return null
    const f = propagate([...data.rows, [tx, ty]], [...data.labels, -1], gamma, k)
    if (!f) return null
    const extended = (
      toArray(harmonicExtension(data.rows, base.scores, [[tx, ty]], { gamma, neighbours: k }).scores) as number[][]
    )[0][1]
    const unlabelled = data.labels.map((l, i) => (l < 0 ? i : -1)).filter((i) => i >= 0)
    const flipped = unlabelled.filter((i) => f[i] > 0.5 !== base.f[i] > 0.5)
    let largest = 0
    for (const i of unlabelled) largest = Math.max(largest, Math.abs(f[i] - base.f[i]))
    return { refitted: f[N], extended, flipped, largest, f }
  }, [data, base, tx, ty, gamma, k])

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const r = refit.value
  const flippedSet = new Set(r?.flipped ?? [])
  const shownF = r?.f ?? base?.f ?? null
  const rest = data.y
    .map((_, i) => i)
    .filter((i) => data.labels[i] < 0 && !flippedSet.has(i) && shownF && shownF[i] !== 0.5)
  const byClass = [0, 1].map((c) => rest.filter((i) => (shownF![i] > 0.5 ? 1 : 0) === c))
  const L = data.y.map((_, i) => i).filter((i) => data.labels[i] >= 0)

  return (
    <Figure
      title="Refit or extend: a new point for label propagation"
      state={state}
      caption="One hundred and fifty points on two moons, two labelled per class (large markers). Label propagation on a k-nearest-neighbour graph is transductive: it scores only the points in the graph. The shading is the inductive rule obtained from it without refitting, the kernel-weighted average of the neighbours' scores at every position, with its boundary in ink. Drag the new point. The transductive answer adds the point to the graph and solves the whole system again; the readouts compare its score with the inductive extension's. Small markers show each old point's class after the refit, and red markers are old points whose class the new point changed. In the gap between the moons a new point joins both moons, and refitting can move the labels of points already scored."
      readouts={
        <>
          <Readout label="refit P(class 1) at the new point" value={r ? formatNumber(r.refitted) : '—'} />
          <Readout label="extension P(class 1)" value={r ? formatNumber(r.extended) : '—'} />
          <Readout label="old points relabelled" value={r ? String(r.flipped.length) : '—'} />
          <Readout label="largest change of an old score" value={r ? formatNumber(r.largest) : '—'} />
        </>
      }
    >
      <Plot x={xa} y={ya} height={380}>
        {base && (
          <Raster
            x={GRID.gx}
            y={GRID.gy}
            z={base.field}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="inductive P(class 1) − ½"
            boundary
            fillOpacity={0.45}
          />
        )}
        {byClass.map(
          (rows, c) =>
            rows.length > 0 && (
              <Points
                key={c}
                name={`class ${c} after refit`}
                x={rows.map((i) => data.px[i])}
                y={rows.map((i) => data.py[i])}
                slot={c}
                thin
                stale={refit.stale}
              />
            ),
        )}
        {flippedSet.size > 0 && (
          <Points
            name="relabelled by the new point"
            x={[...flippedSet].map((i) => data.px[i])}
            y={[...flippedSet].map((i) => data.py[i])}
            tone="destructive"
            size={9}
          />
        )}
        {[0, 1].map((c) => {
          const rows = L.filter((i) => data.labels[i] === c)
          return (
            <Points
              key={`l${c}`}
              name={`class ${c}, labelled`}
              x={rows.map((i) => data.px[i])}
              y={rows.map((i) => data.py[i])}
              slot={c}
              emphasis
              size={12}
            />
          )
        })}
        <Handle {...state.handle(['tx', 'ty'], { label: 'new point' })} />
      </Plot>
    </Figure>
  )
}
