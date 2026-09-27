import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { costMatrix, hungarian, sinkhorn, type Pt } from '../_shared/ot'

const N = 7
const RANGE: [number, number] = [-2.5, 2.5]
const IDX = Array.from({ length: N }, (_, i) => i + 1)
const UNIFORM = new Array(N).fill(1 / N)

function initial(): { src: Pt[]; tgt: Pt[] } {
  const r = rng(11)
  const src: Pt[] = IDX.map(() => [-1.1 + 0.55 * r.normal(), 0.2 + 0.7 * r.normal()])
  const tgt: Pt[] = IDX.map(() => [1.1 + 0.45 * r.normal(), -0.1 + 0.8 * r.normal()])
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
  const logEps = useParam(-1, { min: -2, max: 0.5, step: 0.05 })
  const iters = useParam(30, { min: 0, max: 100, step: 1 })
  const [showExact, setShowExact] = useState(true)
  const eps = 10 ** logEps.value

  const C = useMemo(() => costMatrix(src, tgt, 2), [src, tgt])
  const exact = useMemo(() => {
    const col = hungarian(C)
    return { col, cost: col.reduce((s, j, i) => s + C[i][j], 0) / N }
  }, [C])
  const sk = useMemo(() => sinkhorn(UNIFORM, UNIFORM, C, eps, iters.value), [C, eps, iters.value])

  // Rows of the heatmap show where each source point's mass goes, as fractions of that point's mass.
  const z = useMemo(() => sk.P.map((row) => row.map((v) => v * N)), [sk])

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'source points', type: 'scatter', x: src.map((p) => p[0]), y: src.map((p) => p[1]), slot: 0 },
    ],
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
    (): HeatmapOverlay[] =>
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

  return (
    <Interactive
      title="Exact matching against the entropic plan"
      caption="Blue points are sources; dark points are targets, and each can be dragged. Arrows show the exact optimal matching; grey lines join pairs that carry at least 10% of a source point's mass under the Sinkhorn plan. The matrix shows the Sinkhorn plan, row i being where source point i sends its mass; diamonds mark the exact matching. Small ε gives a plan close to the matching but needs more iterations; large ε blurs every row towards uniform. Step the iterations from 0 to watch the row sums converge."
      controls={
        <>
          <ParamSlider label={`log₁₀ ε (ε = ${formatNumber(eps)})`} param={logEps} />
          <ParamSlider label="Sinkhorn iterations" param={iters} withArrows />
          <ParamSwitch label="show exact matching" checked={showExact} onChange={setShowExact} />
        </>
      }
      readout={
        <>
          <Readout label="exact cost W₂²" value={formatNumber(exact.cost)} />
          <Readout label="Sinkhorn transport cost ⟨C, P⟩" value={formatNumber(sk.cost)} />
          <Readout label="row-sum error ‖P1 − a‖₁" value={formatNumber(sk.marginalError)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <XYChart
          series={series}
          segments={segments}
          vectors={vectors}
          xRange={RANGE}
          yRange={RANGE}
          equalAspect
          xLabel="x₁"
          yLabel="x₂"
          handles={handles}
        />
        <Heatmap
          height={320}
          x={IDX}
          y={IDX}
          z={z}
          range={[0, 1]}
          overlay={overlay}
          xLabel="target j"
          yLabel="source i"
          valueLabel="share of source mass"
        />
      </div>
    </Interactive>
  )
}
