import { useMemo, useState } from 'react'
import {
  Button,
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
import { curvature, regressionSurface, type Vec } from './surfaces'

const log10 = Math.log10

export function StepSizeExplorer() {
  // The data, the loss surface, the optimum and the Hessian are fixed: computed once, on the engine.
  const { value: problem } = useComputed(
    () => {
      const surface = regressionSurface()
      return { surface, ...curvature(surface) }
    },
    [],
    { mode: 'release' },
  )
  const state = useFigureState({
    logLr: float(-1.3, {
      min: -3,
      max: -0.7,
      step: 0.01,
      label: 'step size η',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    steps: int(30, { min: 5, max: 100, step: 1, label: 'steps' }),
  })
  const [start, setStart] = useState<Vec>([-1, 3])

  const lr = 10 ** state.logLr
  const { value: run } = useComputed(
    () => descend(problem.surface.f, start, lr, state.steps),
    [problem, start, lr, state.steps],
  )

  const { surface } = problem
  const overlay = useMemo(
    (): SeriesSpec[] => [
      { name: 'path', type: 'line', x: run.path.map((p) => p[0]), y: run.path.map((p) => p[1]), showPoints: true },
      { name: 'minimum', type: 'scatter', x: [surface.optimum[0]], y: [surface.optimum[1]], emphasis: true },
    ],
    [run, surface],
  )

  const curve = useMemo(() => {
    const e = excess(run.losses, surface.optimumLoss)
    return [{ name: 'MSE − MSE*', type: 'line' as const, x: e.map((_, i) => i), y: e, slot: 1 }]
  }, [run, surface])

  const startHandle: Handle = {
    kind: 'point',
    at: start,
    label: 'start',
    onDrag: ([w, b]) => setStart([inGrid(w, surface.w), inGrid(b, surface.b)]),
  }

  const xAxis = useAxis({ label: 'w' })
  const yAxis = useAxis({ label: 'b' })
  const xAxis2 = useAxis({ label: 'step', hold: 'union' })
  const yAxis2 = useAxis({ label: 'MSE − MSE*', hold: 'union', log: true })

  const limit = 2 / problem.lambdaMax
  const ratio = lr / limit
  const regime =
    run.diverged || ratio >= 1 ? 'diverges' : lr * problem.lambdaMax > 1 ? 'overshoots, converges' : 'no overshoot'
  const preset = (fraction: number) => state.set('logLr', Math.round(log10(fraction * limit) * 100) / 100)

  return (
    <Figure
      title="Step size on a quadratic loss"
      state={state}
      caption="Left: log₁₀ MSE of a line fit over slope w and intercept b, with the gradient-descent path from the start point. Drag the dot to move the start. Right: the excess loss per step on a log scale, so a straight line means a constant rate. Try the presets, then push η past 2/λ_max."
      controls={
        <>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" onClick={() => preset(0.1)}>
              Too small
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() =>
                state.set('logLr', Math.round(log10(2 / (problem.lambdaMin + problem.lambdaMax)) * 100) / 100)
              }
            >
              Best fixed
            </Button>
            <Button variant="outline" size="sm" onClick={() => preset(0.9)}>
              Zig-zag
            </Button>
            <Button variant="outline" size="sm" onClick={() => preset(1.1)}>
              Diverge
            </Button>
          </div>
        </>
      }
      readouts={
        <>
          <Readout label="η" value={formatNumber(lr)} />
          <Readout label="η / (2/λ_max)" value={formatNumber(ratio)} />
          <Readout label="κ = λ_max/λ_min" value={formatNumber(problem.lambdaMax / problem.lambdaMin)} />
          <Readout label="regime" value={regime} />
          <Readout
            label="final excess loss"
            value={run.diverged ? '∞' : formatNumber(run.losses.at(-1)! - surface.optimumLoss)}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={surface.w} y={surface.b} z={surface.z} valueLabel={'log₁₀ MSE'} />
          {seriesLayers(overlay, { live: true })}
          <Handle {...startHandle} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(curve)}
        </Plot>
      </div>
    </Figure>
  )
}
