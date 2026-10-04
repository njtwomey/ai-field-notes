import { useMemo, useState } from 'react'
import { blobs } from 'aifn-methods/data/synthetic'
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
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { row, slider, useFigureState } from '@lab/state'
import { Area, Curve, Handle, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from '@lab/viz'

type Segment = { from: [number, number]; to: [number, number] }

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
const SRC_X = src.filter((_, i) => i % 2 === 0)
const SRC_Y = src.filter((_, i) => i % 2 === 1)
const TGT_X = tgt.filter((_, i) => i % 2 === 0)
const TGT_Y = tgt.filter((_, i) => i % 2 === 1)
const COLS = Array.from({ length: 15 }, (_, j) => j)
const ROWS = Array.from({ length: 12 }, (_, i) => i)

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
  const state = useFigureState({
    reg: row('1 · regularisation', { logEps: slider(-2, 1, -0.5, { label: 'log₁₀ ε', step: 0.1 }) }),
  })
  const eps = 10 ** state.reg.logEps
  const run = useMemo(
    () =>
      trace(sinkhornSteps(A, B, C, { epsilon: eps, tolerance: 1e-8 }), {}, 400, {
        record: { error: (s) => Math.max(s.marginalError, 1e-16) },
      }),
    [eps],
  )
  const [step, setStep] = useState(1)
  const at = Math.min(step, run.steps.length - 1)
  const s: SinkhornState = run.steps[at]
  const plan = useMemo(() => toFlat(s.plan), [s])
  const segments = useMemo(() => couplings(plan, 15, 0.2 * Math.max(...plan)), [plan])
  const rows = useMemo(() => ROWS.map((i) => plan.slice(i * 15, i * 15 + 15)), [plan])
  const errors = useMemo(() => toFlat(run.series.error), [run])
  const x1 = useAxis({ label: 'x₁' })
  const x2 = useAxis({ label: 'x₂', equal: x1 })
  const tj = useAxis({ label: 'target j' })
  const si = useAxis({ label: 'source i' })
  const it = useAxis({ label: 'iteration', key: eps })
  const err = useAxis({ label: 'error', log: true, key: eps })
  return (
    <Figure
      title="Sinkhorn plans as ε changes"
      purpose="Entropic transport spreads each source point's mass over nearby targets; as ε shrinks the plan sharpens towards the exact (sparse) optimal plan and Sinkhorn needs more iterations to converge."
      defaultSize="XL"
      state={state}
      controls={
        <Player
          label="2 · iteration"
          value={at}
          onChange={setStep}
          count={run.steps.length}
          startReason="iteration 0 is the unscaled kernel exp(−C/ε) with almost no mass; iteration 1 is the first plan"
        />
      }
      readouts={{
        costs: (
          <>
            <Readout label="ε" value={eps.toPrecision(3)} />
            <Readout label="⟨C, P⟩" value={s.transportCost.toFixed(4)} />
            <Readout label="exact cost" value={EXACT.cost.toFixed(4)} />
            <Readout label="dual" value={s.dual.toFixed(4)} />
          </>
        ),
        run: (
          <Readout label="converged at" value={run.meta.stopped === 'done' ? run.meta.steps : `> ${run.meta.steps}`} />
        ),
      }}
      caption="aifn/transport sinkhornSteps (log domain) on 12 and 15 equally weighted points with squared-distance cost; lines join pairs carrying at least 20% of the largest entry of P. Left: the couplings; middle: the plan P (source rows × target columns); right: the row-marginal error by iteration. Play from the first iteration, or drag the vertical line on the error panel."
    >
      <Plots cols={3} widths={[1.6, 1, 1]}>
        <Plot x={x1} y={x2}>
          <Segments segments={segments} live />
          <Points name="source a" x={SRC_X} y={SRC_Y} slot={0} />
          <Points name="target b" x={TGT_X} y={TGT_Y} slot={1} />
        </Plot>
        <Plot x={tj} y={si}>
          <Raster x={COLS} y={ROWS} z={rows} valueLabel="P_ij" />
        </Plot>
        <Plot x={it} y={err}>
          <Curve name="row-marginal L1 error" x={run.index} y={errors} slot={2} />
          <Handle kind="x" at={at} label="iteration" onDrag={(v) => setStep(Math.round(v))} />
        </Plot>
      </Plots>
    </Figure>
  )
}

const N = 200
const LEVELS = Array.from({ length: N }, (_, i) => (i + 0.5) / N)

export function Wasserstein1dSpecimen() {
  const state = useFigureState({
    target: row('target sample', {
      shift: slider(-3, 3, 1, { label: 'shift' }),
      scale: slider(0.2, 3, 1.5, { label: 'scale' }),
    }),
  })
  const { shift, scale } = state.target
  const u = useMemo(() => toFlat(normals(stream('w1-u'), N)).sort((a, b) => a - b), [])
  const v = useMemo(
    () =>
      toFlat(normals(stream('w1-v'), N))
        .map((z) => shift + scale * z)
        .sort((a, b) => a - b),
    [shift, scale],
  )
  const w1 = wasserstein1d(u, v)
  const w2 = wasserstein1d(u, v, { p: 2 })
  const q = useAxis({ label: 'quantile level q', range: [0, 1] })
  const y = useAxis({ label: 'value', hold: 'union' })
  return (
    <Figure
      title="Wasserstein distance on the line"
      purpose="On the line the optimal plan matches quantiles, so W_p is the L^p distance between quantile functions: W₁ is the shaded area between them."
      state={state}
      readouts={{
        distances: (
          <>
            <Readout label="W₁ (shaded area)" value={w1.toFixed(4)} />
            <Readout label="W₂" value={w2.toFixed(4)} />
            <Readout label="W₂ for Gaussians" value={Math.sqrt(shift ** 2 + (scale - 1) ** 2).toFixed(4)} />
          </>
        ),
      }}
      caption="aifn/transport wasserstein1d on two samples of 200 normal draws; the Gaussian formula W₂² = (μ₁ − μ₂)² + (σ₁ − σ₂)² is the population value. The shaded band between the two quantile functions has area W₁."
    >
      <Plot x={q} y={y}>
        <Area name="|F⁻¹ − G⁻¹|" x={LEVELS} y={v} base={u} line={false} muted />
        <Curve name="F⁻¹ of u" x={LEVELS} y={u} slot={0} />
        <Curve name="G⁻¹ of v" x={LEVELS} y={v} slot={1} />
      </Plot>
    </Figure>
  )
}
