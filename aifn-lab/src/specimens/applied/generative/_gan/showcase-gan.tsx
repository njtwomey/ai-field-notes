import { useMemo, useState } from 'react'
import type { GanCheckpoint, GanRun, GanRunOptions } from 'aifn-applied/generative/gan'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, type Task } from '@lab/state'
import { Button } from '@lab/ui/button'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis, Vectors, type Vector } from '@lab/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')

/** The datasets, each a registered generator of `aifn-applied/data/synthetic` with a known density. */
const DATASETS = {
  ring: { label: '8 Gaussians on a ring', key: 'gaussianRing', knobs: { n: 1000 } },
  grid: { label: '5 × 5 grid of Gaussians', key: 'gaussianGrid', knobs: { n: 1000 } },
  pinwheel: { label: 'pinwheel (5 arms)', key: 'pinwheel', knobs: { n: 1000 } },
  roll: { label: 'Swiss-roll slice', key: 'swissRoll2d', knobs: { n: 1000 } },
  moons: { label: 'two moons', key: 'moons', knobs: { n: 1000, noise: 0.08, spacing: 'random' } },
  spirals: { label: 'two spirals', key: 'spirals', knobs: { n: 1000, arms: 2, noise: 0.04 } },
} as const
type DataKey = keyof typeof DATASETS

const GAMES = [
  { value: 'minimax', label: 'minimax (original)' },
  { value: 'non-saturating', label: 'non-saturating' },
  { value: 'wasserstein', label: 'Wasserstein + gradient penalty' },
  { value: 'hinge', label: 'hinge' },
] as const
type Game = (typeof GAMES)[number]['value']

const OPTIMISERS = [
  { value: 'adam', label: 'Adam (β₁ 0.5)' },
  { value: 'rmsprop', label: 'RMSProp' },
  { value: 'sgd', label: 'SGD' },
] as const

/**
 * The presets: each sets the whole configuration (Train runs it). Against a discriminator taking five steps per generator
 * step, the minimax generator's gradient vanishes where D rejects its points and it stalls off the data; the
 * non-saturating loss keeps a strong gradient there and finds the modes (Goodfellow et al., 2014, §3). WGAN-GP on two
 * moons first settles on one moon (a collapsed generator) and then slowly spreads to the other.
 */
const STRONG_D = {
  'setup.data': 'ring',
  'setup.criticSteps': 5,
  'optim.criticRate': 0.002,
  'optim.generatorRate': 0.001,
  'run.steps': 1000,
  'run.seed': 4,
}
const PRESETS: Record<string, { label: string; values: Record<string, string | number> }> = {
  minimax: { label: 'minimax, strong D', values: { ...STRONG_D, 'setup.game': 'minimax' } },
  nonSaturating: { label: 'non-saturating, strong D', values: { ...STRONG_D, 'setup.game': 'non-saturating' } },
  wgan: {
    label: 'WGAN-GP on moons',
    values: { 'setup.data': 'moons', 'setup.game': 'wasserstein', 'setup.criticSteps': 2, 'run.steps': 1000 },
  },
}

type Settings = { data: DataKey; seed: number; options: GanRunOptions }

const ganTask = ({ data, seed, options }: Settings): Task<GanRun> => {
  const d = DATASETS[data]
  return call<GanRun>(
    'applied/generative/gan/ganRun',
    call(`applied/data/synthetic/${d.key}`, call('foundation/random/stream', seed), d.knobs),
    options,
  )
}

/** Rows of a row-major [n × 2] array as columns. */
const columns = (a: Float64Array, n = a.length / 2) => ({
  x: Array.from({ length: n }, (_, i) => a[2 * i]),
  y: Array.from({ length: n }, (_, i) => a[2 * i + 1]),
})

/** A row-major g × g field as raster rows. */
const rows = (v: Float64Array, g: number) =>
  Array.from({ length: g }, (_, i) => Array.from(v.subarray(i * g, (i + 1) * g)))

export function GanShowcase() {
  const state = useFigureState({
    setup: row('1 · data and game', {
      data: choice(
        (Object.keys(DATASETS) as DataKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
        'ring',
        { label: 'data' },
      ),
      game: choice(GAMES, 'non-saturating', { label: 'game' }),
      criticSteps: int(1, { label: 'discriminator steps per generator step', ge: 1, le: 10, suggestions: [1, 2, 5] }),
      hidden: choice([32, 64, 128], 64, { label: 'hidden width (both nets, 2 layers)' }),
    }),
    optim: row('2 · optimisers', {
      generator: choice(OPTIMISERS, 'adam', { label: 'generator' }),
      generatorRate: float(0.001, {
        label: 'generator rate',
        gt: 0,
        le: 0.1,
        scale: 'log10',
        suggestions: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
      }),
      critic: choice(OPTIMISERS, 'adam', { label: 'discriminator' }),
      criticRate: float(0.001, {
        label: 'discriminator rate',
        gt: 0,
        le: 0.1,
        scale: 'log10',
        suggestions: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
      }),
    }),
    run: row('3 · training run', {
      steps: int(1500, { ge: 1, suggestions: [500, 1000, 1500, 2000, 3000], label: 'generator steps' }),
      batch: choice([32, 64, 128], 64, { label: 'batch' }),
      seed: int(4, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const { setup, optim, run: runRow } = state
  const settings: Settings = {
    data: setup.data as DataKey,
    seed: runRow.seed,
    options: {
      game: setup.game as Game,
      steps: Number(runRow.steps),
      criticSteps: setup.criticSteps,
      batchSize: Number(runRow.batch),
      hidden: [Number(setup.hidden), Number(setup.hidden)],
      generator: { name: optim.generator, stepSize: optim.generatorRate },
      critic: { name: optim.critic, stepSize: optim.criticRate },
      seed: runRow.seed,
    },
  }
  const trained = useTrainedRun(settings, ganTask)
  const result = trained.run.value
  const shots = useMemo(() => result?.checkpoints ?? [], [result])

  // The checkpoint shown belongs to the run it was picked on; a new run opens at step 0.
  const [picked, setPicked] = useState<{ run: Settings | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const shot: GanCheckpoint | undefined = shots[index]
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }

  const box = result?.box ?? 3
  const x1 = useAxis({ label: 'x₁', range: [-box, box], key: box })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], key: box, equal: x1 })
  const stepAxis = useAxis({ label: 'generator step', range: [0, result?.steps ?? 1], key: result?.steps })
  const lossAxis = useAxis({ label: 'loss', hold: 'union', key: result?.steps })
  const shareAxis = useAxis({ label: 'share', range: [0, 1] })
  const probability = result ? result.game === 'minimax' || result.game === 'non-saturating' : true

  const data = useMemo(() => (result ? columns(result.data) : null), [result])
  const modes = useMemo(() => (result ? columns(result.modes) : null), [result])
  const samples = useMemo(() => (shot ? columns(shot.samples) : null), [shot])
  const g = result?.gridX.length ?? 0
  const field = useMemo(() => (shot && g ? rows(shot.field, g) : null), [shot, g])
  const optimal = useMemo(() => (shot?.optimal && g ? rows(shot.optimal, g) : null), [shot, g])
  // One colour range and one arrow scale for the whole run, so checkpoints compare.
  const scales = useMemo(() => {
    if (!result) return { critic: 1, arrow: 0 }
    let critic = 0
    const lengths: number[] = []
    for (const c of result.checkpoints) {
      for (const v of c.field) critic = Math.max(critic, Math.abs(v))
      for (let i = 0; i < c.push.length / 2; i++) lengths.push(Math.hypot(c.push[2 * i], c.push[2 * i + 1]))
    }
    lengths.sort((a, b) => a - b)
    const typical = lengths[Math.floor(0.95 * (lengths.length - 1))] || 1
    return { critic: critic || 1, arrow: (0.18 * result.box) / typical }
  }, [result])
  const arrows = useMemo((): Vector[] => {
    if (!shot) return []
    const n = shot.push.length / 2
    return Array.from({ length: n }, (_, i) => {
      const from: [number, number] = [shot.samples[2 * i], shot.samples[2 * i + 1]]
      return { from, to: [from[0] + scales.arrow * shot.push[2 * i], from[1] + scales.arrow * shot.push[2 * i + 1]] }
    })
  }, [shot, scales])
  const losses = useMemo(() => {
    if (!result) return null
    const n = result.criticLoss.length
    const stride = Math.max(1, Math.floor(n / 600))
    const x: number[] = []
    const c: number[] = []
    const gl: number[] = []
    for (let i = 0; i < n; i += stride) {
      x.push(i + 1)
      c.push(result.criticLoss[i])
      gl.push(result.generatorLoss[i])
    }
    return { x, c, g: gl }
  }, [result])
  const coverage = useMemo(() => {
    const withCoverage = shots.filter((c) => c.coverage)
    return {
      x: withCoverage.map((c) => c.step),
      hit: withCoverage.map((c) => c.coverage!.hit / c.coverage!.modes),
      quality: withCoverage.map((c) => c.coverage!.quality),
    }
  }, [shots])

  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} /> : null
  const done = result?.done ?? 0
  const total = trained.trained?.options.steps ?? settings.options.steps ?? 1
  const presets = (
    <div className="flex flex-wrap items-center gap-1">
      {Object.entries(PRESETS).map(([key, p]) => (
        <Button
          key={key}
          size="sm"
          variant="outline"
          aria-label={`preset: ${p.label}`}
          onClick={() => {
            state.reset()
            for (const [path, v] of Object.entries(p.values)) state.set(path, v)
          }}
        >
          {p.label}
        </Button>
      ))}
    </div>
  )
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  return (
    <Figure
      title="A GAN on 2-d data"
      purpose="A generator learns to place points where a discriminator cannot tell them from data; the discriminator's field shows where it pushes, and the known density shows which modes the generator found."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls
          run={trained as never}
          progress={done / total}
          progressText={`${done} / ${total} steps`}
          actions={presets}
        />
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="modes hit" value={shot?.coverage ? `${shot.coverage.hit} / ${shot.coverage.modes}` : '—'} />
          <Readout label="high-quality share" value={shot?.coverage ? f3(shot.coverage.quality) : '—'} />
          <Readout label="D loss" value={shot && shot.step > 0 ? f3(result!.criticLoss[shot.step - 1]) : '—'} />
          <Readout label="G loss" value={shot && shot.step > 0 ? f3(result!.generatorLoss[shot.step - 1]) : '—'} />
        </>
      }
      caption={
        <>
          aifn ganRun: an MLP generator (2-d latent noise → {settings.options.hidden?.[0]} →{' '}
          {settings.options.hidden?.[0]} → 2) and discriminator, trained in the worker by adversarialTraining with
          discriminatorLoss / generatorLoss (and the gradient penalty for WGAN-GP). Left: real points (grey), mode
          centres (ink) and 512 points from fixed latents. Middle: the discriminator, D(x) = σ(logit) for the minimax
          and non-saturating games or the critic&apos;s score, with arrows where the generator&apos;s loss pushes 64 of
          the points (one scale for the whole run, so vanishing gradients show as short arrows). Right: the optimal
          discriminator D* = p_data/(p_data + p_g), with p_data exact and p_g a Gaussian KDE of the generated points
          (probability games only). Bottom: both losses, and the share of modes hit and of high-quality points (log
          p_data above its 1% quantile on the data). Presets: against a discriminator with five steps per generator
          step, the minimax generator&apos;s push vanishes and it stalls off the data, while the non-saturating loss
          finds the modes; WGAN-GP on two moons first collapses onto one moon. Play the checkpoints, or drag the step
          marker on either bottom chart.
        </>
      }
    >
      <Plots cols={3}>
        <Plot x={x1} y={x2} title={empty ?? 'generated against real'}>
          {data && <Points name="data" x={data.x} y={data.y} muted thin />}
          {samples && <Points name="generated" x={samples.x} y={samples.y} slot={0} thin />}
          {modes && <Points name="mode centres" x={modes.x} y={modes.y} emphasis size={5} />}
        </Plot>
        <Plot
          x={x1}
          y={x2}
          title={probability ? 'discriminator D(x) and the push on G' : 'critic score and the push on G'}
        >
          {field && result && (
            <Raster
              x={result.gridX}
              y={result.gridY}
              z={field}
              scale={probability ? 'sequential' : 'diverging'}
              range={probability ? [0, 1] : [-scales.critic, scales.critic]}
              valueLabel={probability ? 'D(x)' : 'critic'}
              fillOpacity={0.8}
            />
          )}
          {samples && <Points name="generated" x={samples.x} y={samples.y} slot={0} thin />}
          <Vectors vectors={arrows} />
        </Plot>
        <Plot x={x1} y={x2} title={probability ? 'optimal discriminator D*(x)' : 'D* applies to the probability games'}>
          {optimal && result && (
            <Raster
              x={result.gridX}
              y={result.gridY}
              z={optimal}
              scale="sequential"
              range={[0, 1]}
              valueLabel="D*(x)"
              fillOpacity={0.8}
            />
          )}
          {samples && optimal && <Points name="generated" x={samples.x} y={samples.y} slot={0} thin />}
        </Plot>
      </Plots>
      <Plots cols={2}>
        <Plot x={stepAxis} y={lossAxis} title="losses">
          {losses && <Curve name="discriminator" x={losses.x} y={losses.c} slot={2} />}
          {losses && <Curve name="generator" x={losses.x} y={losses.g} slot={3} />}
          {marker}
        </Plot>
        <Plot x={stepAxis} y={shareAxis} title="mode coverage">
          {coverage.x.length > 0 && <Curve name="modes hit" x={coverage.x} y={coverage.hit} slot={4} showPoints />}
          {coverage.x.length > 0 && (
            <Curve name="high-quality share" x={coverage.x} y={coverage.quality} slot={5} showPoints />
          )}
          {marker}
        </Plot>
      </Plots>
      <Player
        label="checkpoint"
        value={index}
        onChange={pick}
        count={Math.max(1, shots.length)}
        format={(i) => `step ${shots[i]?.step ?? 0}`}
      />
    </Figure>
  )
}
