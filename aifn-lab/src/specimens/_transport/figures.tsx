import { useMemo, useState } from 'react'
import { blobs } from 'aifn-applied/data/synthetic'
import {
  costMatrix,
  exactTransport,
  sinkhornSteps,
  uniformWeights,
  wasserstein1d,
  type SinkhornState,
} from 'aifn/transport'
import { normals, stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type Segment, type XYSeries } from '@lab/viz'

const SOURCE = blobs(stream('ot-source'), { n: 12, centers: [[-2, 0]], sd: 0.8 }).x
const TARGET = blobs(stream('ot-target'), {
  n: 15,
  centers: [
    [2, 0.5],
    [1.5, -1.5],
  ],
  sd: 0.4,
}).x
const C = costMatrix(SOURCE, TARGET)
const A = uniformWeights(12)
const B = uniformWeights(15)
const EXACT = exactTransport(A, B, C)
const src = toFlat(SOURCE)
const tgt = toFlat(TARGET)

function couplings(plan: number[], m: number, threshold: number): Segment[] {
  const out: Segment[] = []
  plan.forEach((p, k) => {
    if (p < threshold) return
    const i = Math.floor(k / m)
    const j = k % m
    out.push({ from: [src[2 * i], src[2 * i + 1]], to: [tgt[2 * j], tgt[2 * j + 1]] })
  })
  return out
}

export function SinkhornSpecimen() {
  const [logEps, setLogEps] = useState(-0.5)
  const eps = 10 ** logEps
  const run = useMemo(
    () =>
      trace(sinkhornSteps(A, B, C, { epsilon: eps, tolerance: 1e-8 }), {}, 400, {
        record: { error: (s) => Math.max(s.marginalError, 1e-16) },
      }),
    [eps],
  )
  const [step, setStep] = useState(1000)
  const at = Math.min(step, run.steps.length - 1)
  const s: SinkhornState = run.steps[at]
  const plan = toFlat(s.plan)
  const maxP = Math.max(...plan)
  const rows = Array.from({ length: 12 }, (_, i) => plan.slice(i * 15, i * 15 + 15))
  const points: XYSeries[] = [
    {
      name: 'source a',
      type: 'scatter',
      x: src.filter((_, i) => i % 2 === 0),
      y: src.filter((_, i) => i % 2 === 1),
      slot: 0,
    },
    {
      name: 'target b',
      type: 'scatter',
      x: tgt.filter((_, i) => i % 2 === 0),
      y: tgt.filter((_, i) => i % 2 === 1),
      slot: 1,
    },
  ]
  const errors: XYSeries[] = [
    { name: 'row-marginal L1 error', type: 'line', x: run.index, y: toFlat(run.series.error), slot: 2 },
  ]
  return (
    <Figure
      title="Sinkhorn plans as ε changes"
      description="Entropic transport spreads each source point's mass over nearby targets; as ε shrinks the plan sharpens towards the exact (sparse) optimal plan and Sinkhorn needs more iterations to converge."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · regularisation">
            <Slider label="log₁₀ ε" value={logEps} onChange={setLogEps} min={-2} max={1} step={0.1} />
          </ControlRow>
          <ControlRow label="2 · iteration">
            <Player label="iteration" value={at} onChange={setStep} count={run.steps.length} defaultSpeed={10} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ε" value={eps.toPrecision(3)} />
          <Readout label="⟨C, P⟩" value={s.transportCost.toFixed(4)} />
          <Readout label="exact cost" value={EXACT.cost.toFixed(4)} />
          <Readout label="dual" value={s.dual.toFixed(4)} />
          <Readout label="converged at" value={run.meta.stopped === 'done' ? run.meta.steps : `> ${run.meta.steps}`} />
        </>
      }
      caption="aifn/ot sinkhornSteps (log domain) on 12 and 15 equally weighted points with squared-distance cost; lines join pairs carrying at least 20% of the largest entry of P. Left: the couplings; middle: the plan P (source rows × target columns); right: the row-marginal error by iteration."
    >
      <Subplots cols={3} widthRatios={[1.3, 1, 1]}>
        <Panel>
          <XYChart series={points} segments={couplings(plan, 15, 0.2 * maxP)} aspect="equal" xLabel="x₁" yLabel="x₂" />
        </Panel>
        <Panel>
          <Heatmap
            x={Array.from({ length: 15 }, (_, j) => j)}
            y={Array.from({ length: 12 }, (_, i) => i)}
            z={rows}
            xLabel="target j"
            yLabel="source i"
            valueLabel="P_ij"
          />
        </Panel>
        <Panel>
          <XYChart
            series={errors}
            xLabel="iteration"
            yLabel="error"
            yLog
            handles={[{ kind: 'x', at, label: 'iteration', onDrag: (v) => setStep(Math.round(v)) }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

export function Wasserstein1dSpecimen() {
  const [shift, setShift] = useState(1)
  const [scale, setScale] = useState(1.5)
  const u = useMemo(() => toFlat(normals(stream('w1-u'), 200)).sort((a, b) => a - b), [])
  const v = useMemo(
    () =>
      toFlat(normals(stream('w1-v'), 200))
        .map((z) => shift + scale * z)
        .sort((a, b) => a - b),
    [shift, scale],
  )
  const q = u.map((_, i) => (i + 0.5) / 200)
  const series: XYSeries[] = [
    { name: 'F⁻¹ of u', type: 'line', x: q, y: u, slot: 0 },
    { name: 'G⁻¹ of v', type: 'line', x: q, y: v, slot: 1 },
  ]
  const w1 = wasserstein1d(u, v)
  const w2 = wasserstein1d(u, v, { p: 2 })
  return (
    <Figure
      title="Wasserstein distance on the line"
      description="On the line the optimal plan matches quantiles, so W_p is the L^p distance between quantile functions: W₁ is the area between them."
      controls={
        <ControlRow label="target sample">
          <Slider label="shift" value={shift} onChange={setShift} min={-3} max={3} />
          <Slider label="scale" value={scale} onChange={setScale} min={0.2} max={3} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="W₁" value={w1.toFixed(4)} />
          <Readout label="W₂" value={w2.toFixed(4)} />
          <Readout label="W₂ for Gaussians" value={Math.sqrt(shift ** 2 + (scale - 1) ** 2).toFixed(4)} />
        </>
      }
      caption="aifn/ot wasserstein1d on two samples of 200 normal draws; the Gaussian formula W₂² = (μ₁ − μ₂)² + (σ₁ − σ₂)² is the population value."
    >
      <XYChart series={series} xLabel="quantile level q" yLabel="value" rescaleOnChange={false} holdFit="union" />
    </Figure>
  )
}
