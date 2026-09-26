import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  type HeatmapOverlay,
} from '@/components/viz'
import type { RegressionSurface } from '@/generated/contracts'
import { useFigure } from '@/lib/generated'
import { mean } from '@/lib/math'
import { descend, excess, snap, type LossAndGrad, type Vec } from './optim'

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
  const [logLr, setLogLr] = useState(-1.3)
  const [steps, setSteps] = useState(30)
  const [start, setStart] = useState<Vec>([-1, 3])

  const problem = useMemo(() => (data ? mseProblem(data.data.x, data.data.y) : undefined), [data])
  const lr = 10 ** logLr
  const run = useMemo(() => (problem ? descend(problem.f, start, lr, steps) : undefined), [problem, start, lr, steps])

  const overlay = useMemo((): HeatmapOverlay[] => {
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

  if (error) return <p className="text-sm text-destructive">{error.message}</p>
  if (!data || !problem || !run) return null

  const limit = 2 / problem.lambdaMax
  const ratio = lr / limit
  const regime =
    run.diverged || ratio >= 1 ? 'diverges' : lr * problem.lambdaMax > 1 ? 'overshoots, converges' : 'no overshoot'
  const preset = (fraction: number) => setLogLr(Math.round(log10(fraction * limit) * 100) / 100)

  return (
    <Interactive
      title="Step size on a quadratic loss"
      caption="Left: log₁₀ MSE of a line fit over slope w and intercept b, with the gradient-descent path from the start point. Click a cell to move the start. Right: the excess loss per step on a log scale, so a straight line means a constant rate. Try the presets, then push η past 2/λ_max."
      controls={
        <>
          <ParamSlider
            label="step size η"
            value={logLr}
            onChange={setLogLr}
            min={-3}
            max={-0.7}
            step={0.01}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider label="steps" value={steps} onChange={setSteps} min={5} max={100} step={1} />
          <div className="flex flex-wrap gap-2">
            <ParamButton onClick={() => preset(0.1)}>Too small</ParamButton>
            <ParamButton
              onClick={() => setLogLr(Math.round(log10(2 / (problem.lambdaMin + problem.lambdaMax)) * 100) / 100)}
            >
              Best fixed
            </ParamButton>
            <ParamButton onClick={() => preset(0.9)}>Zig-zag</ParamButton>
            <ParamButton onClick={() => preset(1.1)}>Diverge</ParamButton>
          </div>
        </>
      }
      readout={
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
        <Heatmap
          x={data.surface.x}
          y={data.surface.y}
          z={data.surface.z}
          xLabel="w"
          yLabel="b"
          valueLabel="log₁₀ MSE"
          overlay={overlay}
          marker={start}
          onCellClick={(w, b) => setStart([snap(w), snap(b)])}
          height={340}
        />
        <XYChart height={340} series={curve} yLog xLabel="step" yLabel="MSE − MSE*" />
      </div>
    </Interactive>
  )
}
