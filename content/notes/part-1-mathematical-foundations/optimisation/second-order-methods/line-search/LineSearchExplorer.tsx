import { useMemo, useState } from 'react'
import { gradientDescent, type FirstOrderState } from 'aifn/optim/first-order'
import { rosenbrock, type TestFunction } from 'aifn-methods/data/objectives'
import { tensor, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import {
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Select,
  Slider,
  useAxis,
} from 'aifn-render'

type Surface = { x: number[]; y: number[]; z: number[][] }

function surfaceOf(fn: TestFunction, n = 80): Surface {
  const [x0, y0] = toFlat(fn.lo)
  const [x1, y1] = toFlat(fn.hi)
  const x = Array.from({ length: n }, (_, i) => x0 + ((x1 - x0) * i) / (n - 1))
  const y = Array.from({ length: n }, (_, i) => y0 + ((y1 - y0) * i) / (n - 1))
  const z = y.map((yi) => x.map((xj) => Math.log10(1 + fn.value(tensor([xj, yi])))))
  return { x, y, z }
}

function pathOf(series: Tensor): { xs: number[]; ys: number[] } {
  const data = toFlat(series)
  const kept = series.shape[0]
  return {
    xs: Array.from({ length: kept }, (_, k) => data[2 * k]),
    ys: Array.from({ length: kept }, (_, k) => data[2 * k + 1]),
  }
}

const ROSEN = rosenbrock()

export function LineSearchExplorer() {
  const [kind, setKind] = useState<'backtracking' | 'strong-wolfe'>('backtracking')
  const [alpha0, setAlpha0] = useState(0.05)
  const [step, setStep] = useState(1)

  const surface = useMemo(() => surfaceOf(ROSEN), [])

  const t = useMemo(
    () =>
      trace(gradientDescent(ROSEN.objective, { lineSearch: kind, stepSize: alpha0 }), { x0: ROSEN.start }, 150, {
        record: {
          'f(x)': (s) => s.value,
          'step α': (s) => s.stepSize,
          trials: (s) => s.lineSearch?.trials.length ?? 0,
          x: (s) => s.x,
        },
      }),
    [kind, alpha0],
  )

  const totalSteps = t.steps.length
  const currentStep = Math.min(step, totalSteps - 1)
  const currentState = t.steps[currentStep] as FirstOrderState
  const previousState = currentStep > 0 ? (t.steps[currentStep - 1] as FirstOrderState) : null

  const path = useMemo(() => pathOf(t.series.x), [t])
  const clip = (v: number, a: readonly number[]) => Math.min(Math.max(v, a[0]), a[a.length - 1])

  const soFar = useMemo(
    () => ({
      x: path.xs.slice(0, currentStep + 1).map((v) => clip(v, surface.x)),
      y: path.ys.slice(0, currentStep + 1).map((v) => clip(v, surface.y)),
    }),
    [path, currentStep, surface],
  )

  const minimaXY = useMemo(
    () => ({
      x: ROSEN.minima.map((m) => toFlat(m)[0]),
      y: ROSEN.minima.map((m) => toFlat(m)[1]),
    }),
    [],
  )

  const profile = useMemo(() => {
    const ls = currentState.lineSearch
    if (!ls || !previousState) return null
    const px = toFlat(previousState.x)
    const pg = toFlat(previousState.grad)
    const alphaMax = Math.max(0.01, ...ls.trials.map((q) => q.alpha)) * 1.2
    const alphas = Array.from({ length: 120 }, (_, i) => (alphaMax * i) / 119)
    return {
      alphas,
      phi: alphas.map((a) => ROSEN.value(tensor([px[0] - a * pg[0], px[1] - a * pg[1]]))),
      armijo: alphas.map((a) => previousState.value + 1e-4 * a * ls.initialSlope),
      trials: { x: ls.trials.map((q) => q.alpha), y: ls.trials.map((q) => q.value) },
      accepted: { x: [ls.alpha], y: [ls.value] },
    }
  }, [currentState, previousState])

  const xPlot = useAxis({ label: 'x₀', range: [surface.x[0], surface.x[surface.x.length - 1]] })
  const yPlot = useAxis({ label: 'x₁', equal: xPlot, range: [surface.y[0], surface.y[surface.y.length - 1]] })
  const alphaAxis = useAxis({ label: 'step length α' })
  const phiAxis = useAxis({ label: 'f(x − α∇f)' })

  return (
    <Figure
      title="Line search trials along descent direction"
      purpose="Visualize step acceptance and rejection along the negative gradient under the Armijo sufficient-decrease and Wolfe curvature conditions."
      caption="Left: the Rosenbrock surface with evaluated trial points (blue dots) along the negative gradient and the trajectory up to the current iteration. Right: 1D slice φ(α) = f(x − α∇f) showing the Armijo sufficient-decrease line, evaluated trial steps, and the accepted step."
    >
      <ControlRow>
        <Select
          label="Line search rule"
          value={kind}
          onChange={(v) => setKind(v as 'backtracking' | 'strong-wolfe')}
          options={[
            { value: 'backtracking', label: 'Backtracking (Armijo)' },
            { value: 'strong-wolfe', label: 'Strong Wolfe' },
          ]}
        />
        <Slider label="Initial trial step α₀" value={alpha0} min={0.005} max={0.2} step={0.005} onChange={setAlpha0} />
      </ControlRow>

      <ControlRow>
        <Player label="Iteration" value={currentStep} count={totalSteps} onChange={setStep} />
      </ControlRow>

      <div className="my-2 flex flex-wrap gap-4 font-mono text-xs text-muted-foreground">
        <Readout label="step" value={`${currentStep} / ${totalSteps - 1}`} />
        <Readout label="f(x)" value={formatNumber(currentState.value)} />
        <Readout label="accepted α" value={currentState.stepSize ? formatNumber(currentState.stepSize) : '—'} />
        <Readout label="trials" value={currentState.lineSearch?.trials.length ?? 0} />
      </div>

      <Plots cols={2}>
        <Plot x={xPlot} y={yPlot} title="Surface & descent path">
          <Raster x={surface.x} y={surface.y} z={surface.z} valueLabel="log₁₀(1 + f)" />
          <Points name="minimum" x={minimaXY.x} y={minimaXY.y} emphasis />
          <Curve name="path" x={soFar.x} y={soFar.y} slot={1} showPoints />
          {currentState.lineSearch && (
            <Points
              name="trials"
              x={currentState.lineSearch.trials.map((q) => toFlat(q.x)[0])}
              y={currentState.lineSearch.trials.map((q) => toFlat(q.x)[1])}
              slot={2}
            />
          )}
          <Points
            name="current"
            x={[clip(path.xs[currentStep], surface.x)]}
            y={[clip(path.ys[currentStep], surface.y)]}
            slot={1}
            size={8}
          />
        </Plot>

        <Plot x={alphaAxis} y={phiAxis} title="1D line search φ(α) & Armijo line">
          {profile ? (
            <>
              <Curve name="φ(α)" x={profile.alphas} y={profile.phi} slot={0} />
              <Curve name="Armijo decrease line" x={profile.alphas} y={profile.armijo} dashed slot={3} />
              <Points name="trials" x={profile.trials.x} y={profile.trials.y} slot={2} />
              <Points name="accepted α" x={profile.accepted.x} y={profile.accepted.y} emphasis size={8} />
            </>
          ) : (
            <Points name="start" x={[0]} y={[currentState.value]} emphasis />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
