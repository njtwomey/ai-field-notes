import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
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

export function OptimiserComparison({ initial = ['sgd', 'nesterov', 'adam'], initialSurface = 'valley' }: Props) {
  const [surfaceId, setSurfaceId] = useState<SurfaceId>(initialSurface)
  const surface = SURFACES.find((s) => s.id === surfaceId)!
  const [logEta, setLogEta] = useState(Math.log10(surface.eta))
  const [logAlpha, setLogAlpha] = useState(Math.log10(surface.alpha))
  const [steps, setSteps] = useState(150)
  const [start, setStart] = useState<Vec>(surface.start)
  const [shown, setShown] = useState<boolean[]>(OPTIMISERS.map((o) => initial.includes(o.id)))

  const chooseSurface = (id: SurfaceId) => {
    const s = SURFACES.find((x) => x.id === id)!
    setSurfaceId(id)
    setStart(s.start)
    setLogEta(Math.log10(s.eta))
    setLogAlpha(Math.log10(s.alpha))
  }

  const grid = useMemo(() => {
    const x = linspace(surface.xRange[0], surface.xRange[1], GRID)
    const y = linspace(surface.yRange[0], surface.yRange[1], GRID)
    const z = y.map((yv) => x.map((xv) => Math.log10(surface.f([xv, yv]).loss + 0.01)))
    return { x, y, z }
  }, [surface])

  const eta = 10 ** logEta
  const alpha = 10 ** logAlpha
  const runs = useMemo(
    () =>
      OPTIMISERS.map((o, i) =>
        shown[i] ? optimise(surface.f, start, o.id, o.adaptive ? alpha : eta, steps) : undefined,
      ),
    [surface, start, eta, alpha, steps, shown],
  )

  const overlay = useMemo((): HeatmapOverlay[] => {
    const paths = OPTIMISERS.flatMap((o, i): HeatmapOverlay[] => {
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
    (): XYSeries[] =>
      OPTIMISERS.flatMap((o, i): XYSeries[] => {
        const run = runs[i]
        if (!run) return []
        const y = run.losses.map((l) => Math.max(l, FLOOR))
        return [{ name: o.label, type: 'line', x: y.map((_, t) => t), y, slot: i + 1 }]
      }),
    [runs],
  )

  const handles: Handle[] = [
    {
      kind: 'point',
      at: start,
      label: 'start',
      onDrag: ([x, y]) => setStart([clamp(x, surface.xRange), clamp(y, surface.yRange)]),
    },
  ]

  return (
    <Interactive
      title="Optimisers on three surfaces"
      caption={`Left: log₁₀(L + 0.01) over the parameters, with each optimiser's path from the same start; drag the start point to move it. Right: the loss per step on a log scale. The valley is L = ½(x² + ${KAPPA}y²), with condition number ${KAPPA}; the rotated valley is the same function turned by 45°, so its axes of curvature are no longer the coordinate axes. SGD, momentum (β = 0.9) and Nesterov share the step size η; AdaGrad, RMSProp (ρ = 0.9) and Adam (β₁ = 0.9, β₂ = 0.999) share the step size α. Rotation leaves the SGD family unchanged and slows the per-coordinate methods.`}
      controls={
        <>
          <ParamChoice
            label="surface"
            value={surfaceId}
            onChange={chooseSurface}
            options={SURFACES.map((s) => ({ value: s.id, label: s.label }))}
          />
          <ParamSlider
            label="SGD-family step size η"
            value={logEta}
            onChange={setLogEta}
            min={-3}
            max={-0.5}
            step={0.01}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider
            label="adaptive step size α"
            value={logAlpha}
            onChange={setLogAlpha}
            min={-2.5}
            max={0}
            step={0.01}
            format={(v) => formatNumber(10 ** v)}
          />
          <ParamSlider label="steps" value={steps} onChange={setSteps} min={20} max={400} step={10} />
          <div className="col-span-full flex flex-wrap gap-x-5 gap-y-2">
            {OPTIMISERS.map((o, i) => (
              <ParamSwitch
                key={o.id}
                label={o.label}
                checked={shown[i]}
                onChange={(v) => setShown((s) => s.map((old, j) => (j === i ? v : old)))}
              />
            ))}
          </div>
        </>
      }
      readout={OPTIMISERS.flatMap((o, i) => {
        const run = runs[i]
        if (!run) return []
        const hit = stepsTo(run.losses, TOL)
        const value = run.diverged ? 'diverged' : hit === undefined ? `> ${steps}` : String(hit)
        return [<Readout key={o.id} label={`${o.label}: steps to L < 10⁻³`} value={value} />]
      })}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={grid.x}
          y={grid.y}
          z={grid.z}
          xLabel="x"
          yLabel="y"
          valueLabel="log₁₀(L + 0.01)"
          overlay={overlay}
          handles={handles}
          height={340}
        />
        <XYChart height={340} series={curves} yLog xLabel="step" yLabel="loss L" />
      </div>
    </Interactive>
  )
}
