import { useMemo, useState } from 'react'
import { empiricalRademacher, shatteringTable, type ShatterClass } from 'aifn-methods/theory/capacity'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

type Point = [number, number]

/** Half-planes need one linear program per labelling, so they stop at 10 points (1,024 programs). */
const HALF_PLANE_LIMIT = 10
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const clamp = (v: number) => Math.max(0.02, Math.min(0.98, v))

/**
 * A Monte Carlo estimate of the empirical Rademacher complexity of a class's projection on a sample the reader can
 * move, as sign vectors are drawn, beside Massart's bound.
 */
export function RademacherEstimate() {
  const state = useFigureState({
    family: choice(
      [
        { value: 'intervals', label: 'intervals on x₁ (VC 2)' },
        { value: 'half-planes', label: 'half-planes (VC 3)' },
        { value: 'rectangles', label: 'axis-aligned rectangles (VC 4)' },
      ],
      'rectangles',
      { label: 'class' },
    ),
    n: int(8, { ge: 2, le: 12, suggestions: [4, 8, 10, 12], label: 'points n' }),
    draws: int(1000, { ge: 10, le: 20_000, suggestions: [100, 1000, 10_000], label: 'sign draws K' }),
    seed: int(1, { ge: 0, le: 999, label: 'seed' }),
  })
  const { family, draws, seed } = state
  const n = family === 'half-planes' ? Math.min(state.n, HALF_PLANE_LIMIT) : state.n
  const key = `${n}/${seed}`
  const drawn = useMemo(
    () =>
      toRows(
        uniform(child(stream('measuring-capacity'), 'points', seed), 0, 1, { shape: [n, 2] }) as Tensor,
      ) as Point[],
    [n, seed],
  )
  // Dragged points belong to one draw of the sample; a new n or seed starts from the new draw.
  const [moved, setMoved] = useState<{ key: string; points: Point[] }>({ key: '', points: [] })
  const points = moved.key === key ? moved.points : drawn
  const drag = (i: number, v: Point) =>
    setMoved({ key, points: points.map((q, j): Point => (j === i ? [clamp(v[0]), clamp(v[1])] : q)) })

  const result = useComputed(
    () => {
      const table = shatteringTable(points, family as ShatterClass)
      const realised = table.labellings.filter((_, j) => table.realised[j])
      const r = empiricalRademacher(child(stream('measuring-capacity'), 'signs', seed), realised, draws)
      return { ...r, projections: realised.length, all: table.labellings.length }
    },
    [points, family, draws, seed],
    { mode: 'release' },
  )
  const r = result.value
  const steps = useMemo(() => Float64Array.from({ length: r.running.length }, (_, i) => i + 1), [r.running.length])

  const x1 = useAxis({ label: 'x₁', range: [0, 1] })
  const x2 = useAxis({ label: 'x₂', range: [0, 1], equal: x1 })
  const kAxis = useAxis({ label: 'sign draws K', range: [1, Math.max(10, draws)], log: true })
  const rAxis = useAxis({ label: 'estimated complexity', range: [0, 1.3] })
  return (
    <Figure
      title="Estimating the empirical Rademacher complexity"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="labellings realised" value={`${r.projections} of ${r.all}`} />
          <Readout label="estimate ± standard error" value={`${fmt(r.estimate)} ± ${fmt(r.standardError)}`} />
          <Readout label="Massart √(2 ln N / n)" value={fmt(r.massart)} />
        </>
      }
      caption="Left: the sample. Drag the points; bunching them or lining them up removes labellings the class can realise. Right: the running Monte Carlo estimate of the empirical Rademacher complexity, the average over sign draws of the best correlation (1/n) Σ σᵢ h(xᵢ) over the realised labellings, which equals 1 − 2 × the smallest error on the random signs. The dashed line is Massart's bound for the N realised labellings, and the top of the chart is 1, the value for a class that fits every sign pattern. The engine enumerates the labellings exactly (shatteringTable), so the estimate is that of an exact ERM."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} title="sample">
          <Points name="points" x={points.map((q) => q[0])} y={points.map((q) => q[1])} slot={0} size={14} />
          {points.map((q, i) => (
            <Handle key={i} kind="point" at={q} label={`${i + 1}`} onDrag={(v) => drag(i, v)} />
          ))}
        </Plot>
        <Plot x={kAxis} y={rAxis} title="estimate as signs are drawn">
          <Curve name="running estimate" slot={1} x={steps} y={r.running} stale={result.stale} />
          <Curve name="Massart bound" x={[1, Math.max(10, draws)]} y={[r.massart, r.massart]} emphasis dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
