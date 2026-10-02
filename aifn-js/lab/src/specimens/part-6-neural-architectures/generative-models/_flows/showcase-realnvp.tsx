/**
 * RealNVP trained in the worker (`aifn-applied/generative/flows`' `realNvpRun`): the model density against the true
 * one, samples, and the data pushed through each coupling layer towards the Gaussian base, played over checkpoints.
 */
import { useMemo, useState } from 'react'
import type { RealNvpCheckpoint, RealNvpRun, RealNvpRunOptions } from 'aifn-applied/generative/flows'
import { Player, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Curve, formatNumber, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const DATASETS = {
  moons: { label: 'two moons', key: 'moons', knobs: { n: 1000, noise: 0.06, spacing: 'random' } },
  ring: { label: '8 Gaussians on a ring', key: 'gaussianRing', knobs: { n: 1000 } },
  pinwheel: { label: 'pinwheel', key: 'pinwheel', knobs: { n: 1000 } },
  spirals: { label: 'two spirals', key: 'spirals', knobs: { n: 1000, arms: 2, noise: 0.03 } },
} as const
type DataKey = keyof typeof DATASETS
type Settings = { data: DataKey; seed: number; options: RealNvpRunOptions }

const task = ({ data, seed, options }: Settings): Task<RealNvpRun> => {
  const d = DATASETS[data]
  return call<RealNvpRun>(
    'applied/generative/flows/realNvpRun',
    call(`applied/data/synthetic/${d.key}`, call('foundation/random/stream', seed), d.knobs),
    options,
  )
}

const columns = (a: ArrayLike<number>, n = a.length / 2) => ({
  x: Array.from({ length: n }, (_, i) => a[2 * i]),
  y: Array.from({ length: n }, (_, i) => a[2 * i + 1]),
})
const rows = (v: Float64Array, g: number) =>
  Array.from({ length: g }, (_, i) => Array.from(v.subarray(i * g, (i + 1) * g)))

export function RealNvpShowcase() {
  const state = useFigureState({
    setup: row('1 · data and flow', {
      data: choice(
        (Object.keys(DATASETS) as DataKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
        'moons',
        { label: 'data' },
      ),
      layers: int(6, { ge: 1, le: 16, suggestions: [2, 4, 6, 8], label: 'coupling layers' }),
      width: choice([16, 32, 64], 32, { label: 'conditioner width (2 layers)' }),
    }),
    run: row('2 · training run', {
      steps: int(1500, { ge: 1, suggestions: [500, 1500, 3000], label: 'Adam steps' }),
      rate: float(1e-3, { gt: 0, scale: 'log10', suggestions: [3e-4, 1e-3, 3e-3], label: 'step size' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const settings: Settings = {
    data: state.setup.data as DataKey,
    seed: state.run.seed,
    options: {
      layers: state.setup.layers,
      hidden: [state.setup.width, state.setup.width],
      steps: state.run.steps,
      stepSize: state.run.rate,
      seed: state.run.seed,
    },
  }
  const trained = useTrainedRun(settings, task)
  const run = trained.run.value
  const shots = useMemo(() => run?.checkpoints ?? [], [run])
  const [picked, setPicked] = useState<{ run: Settings | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const shot: RealNvpCheckpoint | undefined = shots[index]
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const box = run?.box ?? 3
  const x1 = useAxis({ label: 'x₁', range: [-box, box], key: box })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], key: box, equal: x1 })
  const z1 = useAxis({ label: 'coordinate 1', range: [-4, 4] })
  const z2 = useAxis({ label: 'coordinate 2', range: [-4, 4], equal: z1 })
  const stepAxis = useAxis({ label: 'Adam step', range: [0, run?.steps ?? 1], key: run?.steps })
  const nllAxis = useAxis({
    label: 'negative log-likelihood per point',
    hold: 'union',
    key: JSON.stringify(trained.trained),
  })
  const g = run?.gridX.length ?? 0
  const scaleMax = useMemo(() => {
    let m = 0
    const last = shots[shots.length - 1]
    if (last) for (const v of last.density) m = Math.max(m, v)
    return m || 1
  }, [shots])
  // The layer slider is bounded by the trained model (by the setup before training); null shows the last layer.
  const [layerPick, setLayerPick] = useState<number | null>(null)
  const layers = shot ? shot.layers.length - 1 : state.setup.layers
  const layer = Math.min(layerPick ?? layers, layers)
  const pushed = shot ? columns(shot.layers[layer]) : null
  const labels = useMemo(() => Array.from(run?.layerLabels ?? []), [run])
  const at = run && shot ? run.nll.step.findIndex((t) => t >= shot.step) : -1
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} /> : null
  const done = run?.done ?? 0
  const total = trained.trained?.options.steps ?? settings.options.steps ?? 1
  return (
    <Figure
      title="RealNVP: a coupling flow on 2-d data"
      purpose="A normalising flow is an invertible map from data to a Gaussian whose log-Jacobian is cheap: each affine coupling layer rescales and shifts one coordinate by a network of the other, so the density of a point is the Gaussian density of its image times the product of the scales, and samples are Gaussian draws mapped back."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} steps`} />
          <ControlRow label="3 · show">
            <Slider
              label="layer shown (0 = data, last = base)"
              value={layer}
              onChange={(v) => setLayerPick(Math.round(v) >= layers ? null : Math.round(v))}
              min={0}
              max={layers}
              step={1}
              disabled={!shot}
            />
          </ControlRow>
          <ControlRow label="4 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `step ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="NLL" value={run && at >= 0 ? fmt(run.nll.value[at]) : '—'} />
          <Readout label="true density's NLL" value={run ? fmt(run.trueNll) : '—'} />
          <Readout label="layer shown" value={shot ? `${layer} of ${layers}` : '—'} />
        </>
      }
      caption="aifn realNvpRun: alternating affine coupling layers (core's affineCouplingBijector), each conditioner an MLP with log-scales bounded to ±2 by tanh and its output layer zeroed, so training starts from the identity map and the standard normal density. Left: the model density p(x) on one colour scale for the run (the final model's maximum), with samples from fixed Gaussian draws. Middle: 500 data points after the chosen number of layers, coloured by mode; after the last they should look like N(0, I). Right: the training negative log-likelihood, with the true density's for reference (its entropy estimate; the flow cannot go much below it). Play the checkpoints or drag the step marker."
    >
      <Plots cols={3}>
        <Plot x={x1} y={x2} title={!trained.trained ? 'press Train to start' : 'model density and samples'}>
          {shot && run && (
            <Raster
              x={run.gridX}
              y={run.gridY}
              z={rows(shot.density, g)}
              scale="sequential"
              range={[0, scaleMax]}
              valueLabel="p(x)"
              fillOpacity={0.85}
            />
          )}
          {shot && <Points name="samples" {...columns(shot.samples)} slot={1} thin size={2} />}
        </Plot>
        <Plot x={z1} y={z2} title={`data after ${layer} coupling layer${layer === 1 ? '' : 's'}`}>
          {pushed && <Points name="data" x={pushed.x} y={pushed.y} group={labels} thin />}
        </Plot>
        <Plot x={stepAxis} y={nllAxis} title="negative log-likelihood">
          {run && Number.isFinite(run.trueNll) && (
            <Curve name="true density" x={[0, run.steps]} y={[run.trueNll, run.trueNll]} emphasis dashed />
          )}
          {run && <Curve name="flow" slot={0} x={run.nll.step} y={run.nll.value} />}
          {marker}
        </Plot>
      </Plots>
    </Figure>
  )
}
