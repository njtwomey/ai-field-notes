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
  useFigureState,
} from 'aifn-render'
import type { RegressionSurface } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { descend, excess, inGrid, type LossAndGrad, type Vec } from './optim'

const mean = (xs: number[]) => xs.reduce((a, v) => a + v, 0) / xs.length

/** MSE of y ≈ w·x + b from sufficient statistics: O(1) per evaluation, however many samples. */
function mseProblem(x: number[], y: number[]) {
  const mx = mean(x)
  const my = mean(y)
  const mxx = mean(x.map((v) => v * v))
  const mxy = mean(x.map((v, i) => v * y[i]))
  const myy = mean(y.map((v) => v * v))
  const f: LossAndGrad = ([w, b]) => ({
    loss: myy + w * w * mxx + b * b - 2 * w * mxy - 2 * b * my + 2 * w * b * mx,
    grad: [2 * (w * mxx + b * mx - mxy), 2 * (w * mx + b - my)],
  })
  // Hessian 2·[[E x², E x], [E x, 1]]; eigenvalues of a symmetric 2×2 matrix in closed form.
  const tr = 2 * (mxx + 1)
  const det = 4 * (mxx - mx * mx)
  const disc = Math.sqrt(tr * tr - 4 * det)
  return { f, lambdaMax: (tr + disc) / 2, lambdaMin: (tr - disc) / 2 }
}

const log10 = Math.log10

export function StepSizeExplorer() {
  const { data, error } = useFigure<RegressionSurface>('gradient-descent/regression-surface')
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

  const problem = useMemo(() => (data ? mseProblem(data.data.x, data.data.y) : undefined), [data])
  const lr = 10 ** state.logLr
  const run = useMemo(
    () => (problem ? descend(problem.f, start, lr, state.steps) : undefined),
    [problem, start, lr, state.steps],
  )

  const overlay = useMemo((): SeriesSpec[] => {
    if (!run || !data) return []
    return [
      { name: 'path', type: 'line', x: run.path.map((p) => p[0]), y: run.path.map((p) => p[1]), showPoints: true },
      { name: 'minimum', type: 'scatter', x: [data.optimum.x], y: [data.optimum.y], emphasis: true },
    ]
  }, [run, data])

  const curve = useMemo(() => {
    if (!run || !data) return []
    const e = excess(run.losses, data.optimum_loss)
    return [{ name: 'MSE − MSE*', type: 'line' as const, x: e.map((_, i) => i), y: e, slot: 1 }]
  }, [run, data])

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
  const yAxis2 = useAxis({ label: 'MSE − MSE*', hold: 'union', log: true })

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !problem || !run) return null

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
            value={run.diverged ? '∞' : formatNumber(run.losses.at(-1)! - data.optimum_loss)}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={data.surface.x} y={data.surface.y} z={data.surface.z} valueLabel={'log₁₀ MSE'} />
          {seriesLayers(overlay, { live: true })}
          {(startHandle ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(curve)}
        </Plot>
      </div>
    </Figure>
  )
}
