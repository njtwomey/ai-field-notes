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
  type Segment,
  Segments,
  seriesLayers,
  type SeriesSpec,
  setting,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'
import { costMatrix, hungarian, sinkhorn, type Pt } from '../_shared/ot'
import { normal, stream } from 'aifn/foundation/random'

const N = 7
const RANGE: [number, number] = [-2.5, 2.5]
const IDX = Array.from({ length: N }, (_, i) => i + 1)
const UNIFORM = new Array(N).fill(1 / N)

function initial(): { src: Pt[]; tgt: Pt[] } {
  const r = stream(11)
  const src: Pt[] = IDX.map(() => [-1.1 + 0.55 * normal(r), 0.2 + 0.7 * normal(r)])
  const tgt: Pt[] = IDX.map(() => [1.1 + 0.45 * normal(r), -0.1 + 0.8 * normal(r)])
  const clip = (p: Pt): Pt => [clampR(p[0]), clampR(p[1])]
  return { src: src.map(clip), tgt: tgt.map(clip) }
}

const clampR = (v: number) => Math.min(Math.max(v, RANGE[0] + 0.1), RANGE[1] - 0.1)
const START = initial()

/**
 * Seven source and seven target points with equal masses. The exact plan is a matching (Hungarian algorithm). The
 * entropic plan after k Sinkhorn iterations spreads each point's mass over several targets as ε grows.
 */
export function SinkhornClouds() {
  const src = START.src
  const [tgt, setTgt] = useState<Pt[]>(START.tgt)
  const state = useFigureState({
    eps: float(0.1, { min: 0.01, max: 3, scale: 'log10', suggestions: [0.01, 0.03, 0.1, 0.3, 1], label: 'ε' }),
    iters: int(30, { min: 0, max: 100, suggestions: [0, 1, 5, 30, 100], label: 'Sinkhorn iterations' }),
    showExact: setting(true, 'show exact matching'),
  })
  const { eps, showExact } = state
  const iters = { value: state.iters }

  const C = useMemo(() => costMatrix(src, tgt, 2), [src, tgt])
  const exact = useMemo(() => {
    const col = hungarian(C)
    return { col, cost: col.reduce((s, j, i) => s + C[i][j], 0) / N }
  }, [C])
  const sk = useMemo(() => sinkhorn(UNIFORM, UNIFORM, C, eps, iters.value), [C, eps, iters.value])

  // Rows of the heatmap show where each source point's mass goes, as fractions of that point's mass.
  const z = useMemo(() => sk.P.map((row) => row.map((v) => v * N)), [sk])

  const series = useMemo(
    () => [{ name: 'source points', x: src.map((p) => p[0]), y: src.map((p) => p[1]), slot: 0 }] as const,
    [src],
  )
  // Sinkhorn links: every pair that carries at least 10 % of a source point's mass.
  const segments = useMemo((): Segment[] => {
    const out: Segment[] = []
    sk.P.forEach((row, i) =>
      row.forEach((v, j) => {
        if (v * N >= 0.1) out.push({ from: src[i], to: tgt[j] })
      }),
    )
    return out
  }, [sk, src, tgt])
  const vectors = useMemo(
    (): Segment[] => (showExact ? exact.col.map((j, i) => ({ from: src[i], to: tgt[j] })) : []),
    [showExact, exact, src, tgt],
  )
  const overlay = useMemo(
    (): SeriesSpec[] =>
      showExact
        ? [{ name: 'exact matching', type: 'scatter', x: exact.col.map((j) => j + 1), y: IDX, emphasis: true }]
        : [],
    [showExact, exact],
  )

  // Only the targets are draggable: handles draw in ink, so the sources keep their colour.
  const handles: Handle[] = tgt.map((p, i): Handle => ({
    kind: 'point',
    at: p,
    onDrag: ([x, y]) => setTgt((prev) => prev.map((q, k) => (k === i ? [clampR(x), clampR(y)] : q))),
  }))

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: RANGE, equal: xAxis })
  const xAxis2 = useAxis({ label: 'target j' })
  const yAxis2 = useAxis({ label: 'source i' })
  return (
    <Figure
      title="Exact matching against the entropic plan"
      state={state}
      caption="Blue points are sources; dark points are targets, and each can be dragged. Arrows show the exact optimal matching; grey lines join pairs that carry at least 10% of a source point's mass under the Sinkhorn plan. The matrix shows the Sinkhorn plan, row i being where source point i sends its mass; diamonds mark the exact matching. Small ε gives a plan close to the matching but needs more iterations; large ε blurs every row towards uniform. Step the iterations up from 0 to watch the row sums converge."
      readouts={
        <>
          <Readout label="exact cost W₂²" value={formatNumber(exact.cost)} />
          <Readout label="Sinkhorn transport cost ⟨C, P⟩" value={formatNumber(sk.cost)} />
          <Readout label="row-sum error ‖P1 − a‖₁" value={formatNumber(sk.marginalError)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <Plot x={xAxis} y={yAxis}>
          <Points {...series[0]} />
          <Segments segments={segments} />
          <Vectors vectors={vectors} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Raster x={IDX} y={IDX} z={z} range={[0, 1]} valueLabel={'share of source mass'} />
          {seriesLayers(overlay, { live: true })}
        </Plot>
      </div>
    </Figure>
  )
}
