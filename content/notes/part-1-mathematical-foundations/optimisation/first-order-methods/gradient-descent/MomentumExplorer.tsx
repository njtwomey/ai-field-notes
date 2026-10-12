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
  useComputed,
  useFigureState,
} from 'aifn-render'
import { descend, excess, inGrid } from './optim'
import { logisticSurface, type Vec } from './surfaces'

/** First step at which the excess loss falls below `tol`, or undefined if it never does. */
function stepsTo(excessLoss: number[], tol: number): number | undefined {
  const i = excessLoss.findIndex((e) => e < tol)
  return i === -1 ? undefined : i
}

export function MomentumExplorer() {
  // The data, the loss surface and the optimum are fixed: computed once, on the engine.
  const { value: surface } = useComputed(() => logisticSurface(), [], { mode: 'release' })
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

  const lr = 10 ** state.logLr
  const { value: runs } = useComputed(
    () => ({
      plain: descend(surface.f, start, lr, state.steps),
      heavy: descend(surface.f, start, lr, state.steps, state.beta),
    }),
    [surface, start, lr, state.steps, state.beta],
  )

  const overlay = useMemo((): SeriesSpec[] => {
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
      { name: 'minimum', type: 'scatter', x: [surface.optimum[0]], y: [surface.optimum[1]], emphasis: true },
    ]
  }, [runs, surface])

  const curves = useMemo(() => {
    const plain = excess(runs.plain.losses, surface.optimumLoss)
    const heavy = excess(runs.heavy.losses, surface.optimumLoss)
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
  }, [runs, surface, state.beta])

  const startHandle: Handle = {
    kind: 'point',
    at: start,
    label: 'start',
    onDrag: ([w, b]) => setStart([inGrid(w, surface.w), inGrid(b, surface.b)]),
  }

  const xAxis = useAxis({ label: 'w' })
  const yAxis = useAxis({ label: 'b' })
  const xAxis2 = useAxis({ label: 'step', hold: 'union' })
  const yAxis2 = useAxis({ label: 'L − L*', hold: 'union', log: true })

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
          <Raster x={surface.w} y={surface.b} z={surface.z} valueLabel={'log₁₀ cross-entropy'} />
          {seriesLayers(overlay, { live: true })}
          <Handle {...startHandle} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(curves.series)}
        </Plot>
      </div>
    </Figure>
  )
}
