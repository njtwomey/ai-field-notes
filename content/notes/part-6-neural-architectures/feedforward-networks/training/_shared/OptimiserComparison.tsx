import { useMemo } from 'react'
import {
  Figure,
  float,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  row,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import {
  KAPPA,
  OPTIMISERS,
  SURFACES,
  optimise,
  stepsTo,
  type OptimiserId,
  type SurfaceId,
  type Vec,
} from './optimisers'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const GRID = 81
const TOL = 1e-3
/** Floor for the loss on a log axis once a run has converged to machine precision. */
const FLOOR = 1e-12

type Props = {
  /** Optimisers switched on at first render; the others can be switched on by the reader. */
  initial?: OptimiserId[]
  initialSurface?: SurfaceId
}

const clamp = (v: number, [lo, hi]: Vec) => Math.round(Math.min(Math.max(v, lo), hi) * 100) / 100

/** One surface's own settings: each surface opens at its own step sizes and start, and keeps them when revisited. */
function surfaceCase(id: SurfaceId) {
  const s = SURFACES.find((x) => x.id === id)!
  return {
    label: s.label,
    params: {
      eta: float(s.eta, {
        min: 1e-3,
        max: 0.3,
        scale: 'log10',
        suggestions: [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        label: 'SGD-family step size η',
      }),
      alpha: float(s.alpha, {
        min: 3e-3,
        max: 1,
        scale: 'log10',
        suggestions: [0.003, 0.01, 0.03, 0.1, 0.3, 1],
        label: 'adaptive step size α',
      }),
      x0: slider(s.xRange[0], s.xRange[1], s.start[0], { step: 0.01, onChart: true }),
      y0: slider(s.yRange[0], s.yRange[1], s.start[1], { step: 0.01, onChart: true }),
    },
  }
}

export function OptimiserComparison({ initial = ['sgd', 'nesterov', 'adam'], initialSurface = 'valley' }: Props) {
  const state = useFigureState({
    surface: variants(
      { valley: surfaceCase('valley'), rotated: surfaceCase('rotated'), rosenbrock: surfaceCase('rosenbrock') },
      { choiceLabel: 'surface', initial: initialSurface },
    ),
    steps: int(150, { min: 20, max: 400, suggestions: [50, 150, 400], label: 'steps' }),
    show: row(
      'optimisers',
      Object.fromEntries(OPTIMISERS.map((o) => [o.id, setting(initial.includes(o.id), o.label)])) as Record<
        OptimiserId,
        ReturnType<typeof setting>
      >,
    ),
  })
  const surface = SURFACES.find((s) => s.id === state.surface.key)!
  const { eta, alpha, x0, y0 } = state.surface.values
  const steps = state.steps
  const start = useMemo<Vec>(() => [x0, y0], [x0, y0])
  const showValues = state.show
  const shown = useMemo(() => OPTIMISERS.map((o) => showValues[o.id]), [showValues])

  const grid = useMemo(() => {
    const x = toFlat(linspace(surface.xRange[0], surface.xRange[1], GRID))
    const y = toFlat(linspace(surface.yRange[0], surface.yRange[1], GRID))
    const z = y.map((yv) => x.map((xv) => Math.log10(surface.f([xv, yv]).loss + 0.01)))
    return { x, y, z }
  }, [surface])

  const runs = useMemo(
    () =>
      OPTIMISERS.map((o, i) =>
        shown[i] ? optimise(surface.f, start, o.id, o.adaptive ? alpha : eta, steps) : undefined,
      ),
    [surface, start, eta, alpha, steps, shown],
  )

  const overlay = useMemo((): SeriesSpec[] => {
    const paths = OPTIMISERS.flatMap((o, i): SeriesSpec[] => {
      const run = runs[i]
      if (!run) return []
      return [{ name: o.label, type: 'line', x: run.path.map((p) => p[0]), y: run.path.map((p) => p[1]), slot: i + 1 }]
    })
    return [
      ...paths,
      { name: 'minimum', type: 'scatter', x: [surface.minimum[0]], y: [surface.minimum[1]], emphasis: true },
    ]
  }, [runs, surface])

  const curves = useMemo(
    (): SeriesSpec[] =>
      OPTIMISERS.flatMap((o, i): SeriesSpec[] => {
        const run = runs[i]
        if (!run) return []
        const y = run.losses.map((l) => Math.max(l, FLOOR))
        return [{ name: o.label, type: 'line', x: y.map((_, t) => t), y, slot: i + 1 }]
      }),
    [runs],
  )

  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'y' })
  const xAxis2 = useAxis({ label: 'step', hold: 'union' })
  const yAxis2 = useAxis({ label: 'loss L', hold: 'union', log: true })
  return (
    <Figure
      title="Optimisers on three surfaces"
      caption={`Left: log₁₀(L + 0.01) over the parameters, with each optimiser's path from the same start; drag the start point to move it. Right: the loss per step on a log scale. The valley is L = ½(x² + ${KAPPA}y²), with condition number ${KAPPA}; the rotated valley is the same function turned by 45°, so its axes of curvature are no longer the coordinate axes. SGD, momentum (β = 0.9) and Nesterov share the step size η; AdaGrad, RMSProp (ρ = 0.9) and Adam (β₁ = 0.9, β₂ = 0.999) share the step size α. Rotation leaves the SGD family unchanged and slows the per-coordinate methods.`}
      state={state}
      readouts={OPTIMISERS.flatMap((o, i) => {
        const run = runs[i]
        if (!run) return []
        const hit = stepsTo(run.losses, TOL)
        const value = run.diverged ? 'diverged' : hit === undefined ? `> ${steps}` : String(hit)
        return [<Readout key={o.id} label={`${o.label}: steps to L < 10⁻³`} value={value} />]
      })}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={grid.x} y={grid.y} z={grid.z} valueLabel={'log₁₀(L + 0.01)'} />
          {seriesLayers(overlay, { live: true })}
          <Handle
            kind="point"
            at={start}
            label="start"
            onDrag={([x, y]) => {
              state.set('surface.x0', clamp(x, surface.xRange))
              state.set('surface.y0', clamp(y, surface.yRange))
            }}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(curves)}
        </Plot>
      </div>
    </Figure>
  )
}
