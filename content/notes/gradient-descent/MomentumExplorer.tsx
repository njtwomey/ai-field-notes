import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type HeatmapOverlay,
} from '@/components/viz'
import type { LogisticValley } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { descend, excess, snap, type LossAndGrad, type Vec } from './optim'

/** Mean cross-entropy of P(y = 1) = σ(w·x + b), with its gradient. */
function crossEntropy(x: number[], y: number[]): LossAndGrad {
  return ([w, b]) => {
    let loss = 0
    let gw = 0
    let gb = 0
    for (let i = 0; i < x.length; i++) {
      const z = w * x[i] + b
      // log(1 + e^z) − y·z, written to avoid overflow for large |z|.
      loss += Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z))) - y[i] * z
      const r = 1 / (1 + Math.exp(-z)) - y[i]
      gw += r * x[i]
      gb += r
    }
    const n = x.length
    return { loss: loss / n, grad: [gw / n, gb / n] }
  }
}

/** First step at which the excess loss falls below `tol`, or undefined if it never does. */
function stepsTo(excessLoss: number[], tol: number): number | undefined {
  const i = excessLoss.findIndex((e) => e < tol)
  return i === -1 ? undefined : i
}

export function MomentumExplorer() {
  const { data, error } = useFigure<LogisticValley>('gradient-descent/logistic-valley')
  const [logLr, setLogLr] = useState(-1)
  const [beta, setBeta] = useState(0.9)
  const [steps, setSteps] = useState(200)
  const [start, setStart] = useState<Vec>([4, 0])

  const f = useMemo(() => (data ? crossEntropy(data.data.x, (data.data.group ?? []).map(Number)) : undefined), [data])
  const lr = 10 ** logLr
  const runs = useMemo(
    () => (f ? { plain: descend(f, start, lr, steps), heavy: descend(f, start, lr, steps, beta) } : undefined),
    [f, start, lr, steps, beta],
  )

  const overlay = useMemo((): HeatmapOverlay[] => {
    if (!runs || !data) return []
    const line = (name: string, path: Vec[], slot: number): HeatmapOverlay => ({
      name,
      type: 'line',
      x: path.map((p) => p[0]),
      y: path.map((p) => p[1]),
      slot,
    })
    return [
      line('plain', runs.plain.path, 1),
      line('momentum', runs.heavy.path, 2),
      { name: 'minimum', type: 'scatter', x: [data.optimum.x], y: [data.optimum.y], emphasis: true },
    ]
  }, [runs, data])

  const curves = useMemo(() => {
    if (!runs || !data) return undefined
    const plain = excess(runs.plain.losses, data.optimum_loss)
    const heavy = excess(runs.heavy.losses, data.optimum_loss)
    return {
      plain,
      heavy,
      series: [
        { name: 'plain', type: 'line' as const, x: plain.map((_, i) => i), y: plain, slot: 1 },
        {
          name: `momentum β = ${formatNumber(beta)}`,
          type: 'line' as const,
          x: heavy.map((_, i) => i),
          y: heavy,
          slot: 2,
        },
      ],
    }
  }, [runs, data, beta])

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !runs || !curves) return null

  const fmtSteps = (n: number | undefined) => (n === undefined ? `> ${steps}` : String(n))

  return (
    <Interactive
      title="Momentum in a narrow valley"
      caption="Left: log₁₀ cross-entropy of a 1-D logistic regression over weight w and bias b. The feature is not centred, so the valley is long and thin. Both optimisers use the same step size and start; click a cell to move the start. Plain gradient descent reaches the valley floor quickly, then crawls along it. Momentum keeps its speed along the floor. With β near 1 it overshoots, and its loss rises and falls."
      controls={
        <>
          <ParamSlider
            label="step size η"
            value={logLr}
            onChange={setLogLr}
            min={-2.5}
            max={0}
            step={0.01}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider label="momentum β" value={beta} onChange={setBeta} min={0} max={0.99} step={0.01} />
          <ParamSlider label="steps" value={steps} onChange={setSteps} min={10} max={400} step={5} />
        </>
      }
      readout={
        <>
          <Readout label="plain excess loss" value={runs.plain.diverged ? '∞' : formatNumber(curves.plain.at(-1)!)} />
          <Readout
            label="momentum excess loss"
            value={runs.heavy.diverged ? '∞' : formatNumber(curves.heavy.at(-1)!)}
          />
          <Readout label="steps to 10⁻², plain" value={fmtSteps(stepsTo(curves.plain, 1e-2))} />
          <Readout label="steps to 10⁻², momentum" value={fmtSteps(stepsTo(curves.heavy, 1e-2))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={data.surface.x}
          y={data.surface.y}
          z={data.surface.z}
          xLabel="w"
          yLabel="b"
          valueLabel="log₁₀ cross-entropy"
          overlay={overlay}
          marker={start}
          onCellClick={(w, b) => setStart([snap(w), snap(b)])}
          height={340}
        />
        <XYChart height={340} series={curves.series} yLog xLabel="step" yLabel="L − L*" />
      </div>
    </Interactive>
  )
}
