import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import type { LogisticValley } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { descend, excess, inGrid, type LossAndGrad, type Vec } from './optim'

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
  const state = useFigureState({
    logLr: float(-1, {
      min: -2.5,
      max: 0,
      step: 0.01,
      label: 'step size η',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    beta: float(0.9, { min: 0, max: 0.99, step: 0.01, label: 'momentum β' }),
    steps: int(200, { min: 10, max: 400, step: 5, label: 'steps' }),
  })
  const [start, setStart] = useState<Vec>([4, 0])

  const f = useMemo(() => (data ? crossEntropy(data.data.x, (data.data.group ?? []).map(Number)) : undefined), [data])
  const lr = 10 ** state.logLr
  const runs = useMemo(
    () =>
      f
        ? { plain: descend(f, start, lr, state.steps), heavy: descend(f, start, lr, state.steps, state.beta) }
        : undefined,
    [f, start, lr, state.steps, state.beta],
  )

  const overlay = useMemo((): SeriesSpec[] => {
    if (!runs || !data) return []
    const line = (name: string, path: Vec[], slot: number): SeriesSpec => ({
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
          name: `momentum β = ${formatNumber(state.beta)}`,
          type: 'line' as const,
          x: heavy.map((_, i) => i),
          y: heavy,
          slot: 2,
        },
      ],
    }
  }, [runs, data, state.beta])

  const startHandle: Handle[] | undefined = data && [
    {
      kind: 'point',
      at: start,
      label: 'start',
      onDrag: ([w, b]) => setStart([inGrid(w, data.surface.x), inGrid(b, data.surface.y)]),
    },
  ]

  // Axes before any early return: hooks run in the same order on every render.
  const xAxis = useAxis({ label: 'w' })
  const yAxis = useAxis({ label: 'b' })
  const xAxis2 = useAxis({ label: 'step', hold: 'union' })
  const yAxis2 = useAxis({ label: 'L − L*', hold: 'union', log: true })

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !runs || !curves) return null

  const fmtSteps = (n: number | undefined) => (n === undefined ? `> ${state.steps}` : String(n))

  return (
    <Figure
      title="Momentum in a narrow valley"
      state={state}
      caption="Left: log₁₀ cross-entropy of a 1-D logistic regression over weight w and bias b. The feature is not centred, so the valley is long and thin. Both optimisers use the same step size and start; drag the dot to move the start. Plain gradient descent reaches the valley floor quickly, then crawls along it. Momentum keeps its speed along the floor. With β near 1 it overshoots, and its loss rises and falls."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={data.surface.x} y={data.surface.y} z={data.surface.z} valueLabel={'log₁₀ cross-entropy'} />
          {seriesLayers(overlay, { live: true })}
          {(startHandle ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(curves.series)}
        </Plot>
      </div>
    </Figure>
  )
}
